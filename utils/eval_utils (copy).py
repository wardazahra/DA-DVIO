
import os
import torch
from devo.utils import Timer
from pathlib import Path
import datetime
import numpy as np
import yaml
import glob
from itertools import chain
from natsort import natsorted
import copy
import math
import shutil
from scipy.spatial.transform import Rotation as R
from tabulate import tabulate

from devo.plot_utils import plot_trajectory, fig_trajectory
from devo.plot_utils import save_trajectory_tum_format

from utils.viz_utils import show_image, visualize_voxel

from evo.tools import file_interface
import evo.main_ape as main_ape
from evo.core import sync, metrics
from evo.core.trajectory import PoseTrajectory3D
from evo.core.geometry import GeometryException


def detect_static_frames(trajectory, threshold=1e-5):
    """
    Detect initial static frames in trajectory where poses don't change significantly.
    
    Args:
        trajectory: PoseTrajectory3D object
        threshold: minimum distance threshold to consider movement
    
    Returns:
        int: number of initial static frames to skip
    """
    positions = trajectory.positions_xyz
    if len(positions) < 2:
        return 0
    
    # Calculate distances between consecutive positions
    distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    
    # Find first frame where movement exceeds threshold
    moving_frames = np.where(distances > threshold)[0]
    
    if len(moving_frames) == 0:
        # Entire trajectory is static
        return len(positions) - 1
    
    # Return the first moving frame index + 1 (to include some static context)
    first_moving_frame = moving_frames[0] + 1
    
    # Skip at least the first few static frames, but keep some for alignment
    skip_frames = max(0, min(first_moving_frame - 5, first_moving_frame // 2))
    
    print(f"Detected {first_moving_frame} static frames, skipping {skip_frames} for alignment")
    return skip_frames


from multiprocessing import Process, Queue
from dpvo.dpvo import DPVO
from dpvo.stream import image_stream
from dpvo.utils import Timer

from devo.devo import DEVO

from devo.devo2 import DEVO as DEVO_GBA

from devo.dba import DBA as DEIO2

import gtsam
from tqdm import tqdm

# 下面的执行为DEIO2
@torch.no_grad()
def run_DEIO2(voxeldir, cfg, network, viz=False, iterator=None, _all_imu=None, _all_gt=None, _all_gt_keys=None, timing=False, H=480, W=640, viz_flow=False, scale=1.0, **kwargs): 
    
    # IMU内参及与camera的外参的初始化
    Ti1c=np.array(cfg.Ti1c).reshape(4,4)
    if cfg.ENALBE_INV:
        Ti1c=np.linalg.inv(Ti1c)
    IMU_noise = np.array([ cfg.accel_noise_sigma, cfg.gyro_noise_sigma, cfg.accel_bias_sigma, cfg.gyro_bias_sigma])

    slam = None #初始化slam
    slam = DEIO2(cfg, network, evs=True, ht=H, wd=W, viz=viz, viz_flow=viz_flow, **kwargs)

    slam.Ti1c=Ti1c
    slam.Tbc = gtsam.Pose3(slam.Ti1c) #将矩阵转换为gtsam.Pose3类型
    slam.state.set_imu_params((IMU_noise*1.0).tolist())#设置IMU的噪声参数

    # 注意，此时进来全部的时间都为us，在进入程序前统一变为秒
    _all_imu[:,0] /= 1e6 #将us转换为秒
    slam.all_imu=_all_imu
    
    for i, (voxel, intrinsics, t) in enumerate(tqdm(iterator)):
        # 将t从us转换为秒
        t=t/1e6
        if timing and i == 0:
            i=i+1
            t0 = torch.cuda.Event(enable_timing=True)
            t1 = torch.cuda.Event(enable_timing=True)
            t0.record()

        if viz: 
            # import matplotlib.pyplot as plt
            # plt.switch_backend('Qt5Agg')
            visualize_voxel(voxel.detach().cpu())
        
        with Timer("DEVO", enabled=timing):
            slam(t, voxel, intrinsics, scale=scale) #调用__call__函数进行跟踪

    for _ in range(12):#跑完后，再跑12次
        slam.update()

    poses, tstamps = slam.terminate()

    # tstamps将秒转换为us
    tstamps=tstamps*1e6

    if timing:
        t1.record()
        torch.cuda.synchronize()
        dt = t0.elapsed_time(t1)/1e3
        avg_fps = (i+1)/dt
        print(f"{voxeldir}\nDEVO Network {i+1} frames in {dt} sec, e.g. {avg_fps} FPS")
    else:
        avg_fps = None
    
    flowdata = slam.flow_data if viz_flow else None
    return poses, tstamps, flowdata, avg_fps


# 下面的执行为DEVO 带GBA的
@torch.no_grad()
def EVO_run_GBA(voxeldir, cfg, network, viz=False, iterator=None, timing=False, H=480, W=640, viz_flow=False, scale=1.0, **kwargs): 

    slam = None #初始化slam
    slam = DEVO_GBA(cfg, network, evs=True, ht=H, wd=W, viz=viz, viz_flow=viz_flow, **kwargs)
    
    # for i, (voxel, intrinsics, t) in enumerate(iterator):
    for i, (voxel, intrinsics, t) in enumerate(tqdm(iterator)):
        if timing and i == 0:
            t0 = torch.cuda.Event(enable_timing=True)
            t1 = torch.cuda.Event(enable_timing=True)
            t0.record()

        if viz: 
            # import matplotlib.pyplot as plt
            # plt.switch_backend('Qt5Agg')
            visualize_voxel(voxel.detach().cpu())
        
        with Timer("DEVO", enabled=timing):
            slam(t, voxel, intrinsics, scale=scale)

    for _ in range(12):#跑完后，再跑12次
        slam.update()

    poses, tstamps = slam.terminate()

    if timing:
        t1.record()
        torch.cuda.synchronize()
        dt = t0.elapsed_time(t1)/1e3
        print(f"{voxeldir}\nDEVO Network {i+1} frames in {dt} sec, e.g. {(i+1)/dt} FPS")
    
    flowdata = slam.flow_data if viz_flow else None
    return poses, tstamps, flowdata

# 下面的执行为DEVO，相当于原本的DEVO中的run_voxel
@torch.no_grad()
def EVO_run(voxeldir, cfg, network, viz=False, iterator=None, timing=False, H=480, W=640, viz_flow=False, scale=1.0, **kwargs): 
    slam = None #初始化slam
    slam = DEVO(cfg, network, evs=True, ht=H, wd=W, viz=viz, viz_flow=viz_flow, **kwargs)
    
    for i, (voxel, intrinsics, t) in enumerate(tqdm(iterator)):
        if timing and i == 0:
            i=i+1
            t0 = torch.cuda.Event(enable_timing=True)
            t1 = torch.cuda.Event(enable_timing=True)
            t0.record()

        if viz: 
            # import matplotlib.pyplot as plt
            # plt.switch_backend('Qt5Agg')
            visualize_voxel(voxel.detach().cpu())
        
        with Timer("DEVO", enabled=timing):
            slam(t, voxel, intrinsics, scale=scale)

    for _ in range(12):
        slam.update()

    poses, tstamps = slam.terminate()

    if timing:
        t1.record()
        torch.cuda.synchronize()
        dt = t0.elapsed_time(t1)/1e3
        avg_fps = (i+1)/dt
        print(f"{voxeldir}\nDEVO Network {i+1} frames in {dt} sec, e.g. {(i+1)/dt} FPS")
    else:
        avg_fps = None
    
    flowdata = slam.flow_data if viz_flow else None
    return poses, tstamps, flowdata, avg_fps

# 下面执行的为DPVO
@torch.no_grad()
def VO_run(cfg, network, imagedir, calib, stride=1, viz=False, show_img=False):

    slam = None

    queue = Queue(maxsize=8)
    reader = Process(target=image_stream, args=(queue, imagedir, calib, stride, 0))#多线程调用image_stream函数
    reader.start()

    # 初始化 tqdm 进度条
    pbar = tqdm(desc="Processing frames")

    while 1:
        (t, image, intrinsics) = queue.get()
        if t < 0: break

        image = torch.from_numpy(image).permute(2,0,1).cuda()
        intrinsics = torch.from_numpy(intrinsics).cuda()

        if show_img:
            show_image(image, 1)

        if slam is None:
            slam = DPVO(cfg, network, ht=image.shape[1], wd=image.shape[2], viz=viz)

        with Timer("SLAM", enabled=False):
            slam(t, image, intrinsics)
        
        # 更新进度条
        pbar.update(1)

    reader.join()

    # 关闭进度条
    pbar.close()

    return slam.terminate()


def assert_eval_config(args):
    assert os.path.isfile(args.weights) and (".pth" in args.weights or ".pt" in args.weights)
    assert os.path.isfile(args.val_split)
    assert args.trials > 0

def ate(traj_ref, traj_est, timestamps):
    import evo
    import evo.main_ape as main_ape
    from evo.core.trajectory import PoseTrajectory3D
    from evo.core.metrics import PoseRelation

    traj_est = PoseTrajectory3D(
        positions_xyz=traj_est[:,:3],
        orientations_quat_wxyz=traj_est[:,3:], # TODO wrong format: EVO uses wxyz, we use xyzw
        timestamps=timestamps)

    traj_ref = PoseTrajectory3D(
        positions_xyz=traj_ref[:,:3],
        orientations_quat_wxyz=traj_ref[:,3:],  # TODO wrong format: EVO uses wxyz, we use xyzw
        timestamps=timestamps)
    
    result = main_ape.ape(traj_ref, traj_est, est_name='traj', 
        pose_relation=PoseRelation.translation_part, align=True, correct_scale=True)

    return result.stats["rmse"]

def get_alg(n):
    if n == "eds" or n == "tumvie" or n == "tartanair":
        return "rgb"
    elif n == "eds_evs" or n == "tumvie_evs" or n == "tartanair_evs":
        return "evs"
    elif n == "eds_evs_viz" or n == "tumvie_evs_viz" or n == "tartanair_evs_viz":
        return "evs_viz"

def make_outfolder(outdir, dataset_name, expname, scene_name, trial, train_step, stride, calib1_eds, camID_tumvie):
    date = datetime.datetime.today().strftime('%Y-%m-%d') # TODO improve output folder
    outfolder = os.path.join(f"{outdir}/{dataset_name}/{date}/{expname}/{scene_name}_trial_{trial}_step_{train_step}")
    if stride != 1:
        outfolder = outfolder + f"_stride_{stride}"
    if calib1_eds != None:
        outfolder = outfolder + f"_calib1" if calib1_eds else outfolder + f"_calib0"
    if camID_tumvie != None:
        outfolder = outfolder + f"_camID_{camID_tumvie}"
    outfolder = os.path.abspath(outfolder)
    os.makedirs(outfolder, exist_ok=True)
    return outfolder

def run_rpg_eval(outfolder, traj_ref, tss_ref_us, traj_est, tstamps):
    p = f"{outfolder}/"
    p = os.path.abspath(p)
    os.makedirs(p, exist_ok=True)

    fnameGT = os.path.join(p, "stamped_groundtruth.txt")
    f = open(fnameGT, "w")
    f.write("# timestamp[secs] tx ty tz qx qy qz qw\n")
    for i in range(len(traj_ref)):
        f.write(f"{tss_ref_us[i]/1e6} {traj_ref[i,0]} {traj_ref[i,1]} {traj_ref[i,2]} {traj_ref[i,3]} {traj_ref[i,4]} {traj_ref[i,5]} {traj_ref[i,6]}\n")
    f.close()

    fnameEst = os.path.join(p, "stamped_traj_estimate.txt")
    f = open(fnameEst, "w")
    f.write("# timestamp[secs] tx ty tz qx qy qz qw\n")
    for i in range(len(traj_est)):
        f.write(f"{tstamps[i]/1e6} {traj_est[i,0]} {traj_est[i,1]} {traj_est[i,2]} {traj_est[i,3]} {traj_est[i,4]} {traj_est[i,5]} {traj_est[i,6]}\n")
    f.close()
    
    # cmd = f"python thirdparty/rpg_trajectory_evaluation/scripts/analyze_trajectory_single.py --result_dir {p} --recalculate_errors --png --plot"
    cmd = f"python thirdparty/rpg_trajectory_evaluation/scripts/analyze_trajectory_single.py {p} --recalculate_errors --png --plot"
    os.system(cmd)

    return fnameGT, fnameEst

def load_stats_rpg_results(outfolder):
    rpg_fspath = os.path.join(outfolder, "saved_results/traj_est")

    absfile = natsorted(glob.glob(os.path.join(rpg_fspath, "absolute_err_stat*.yaml")))[-1]
    with open(absfile, 'r') as file:
        abs_stats = yaml.safe_load(file)

    last_relfile = natsorted(glob.glob(os.path.join(rpg_fspath, "relative_error_statistics_*.yaml")))[-1]
    with open(last_relfile, 'r') as file:
        rel_stats = yaml.safe_load(file)

    # last_relfile_time = natsorted(glob.glob(os.path.join(rpg_fspath, "Time_relative_error_statistics_*.yaml")))[-1]
    # with open(last_relfile_time, 'r') as file:
    #     rel_stats_time = yaml.safe_load(file)
    rel_stats_time = copy.deepcopy(rel_stats) 
    
    return abs_stats, rel_stats, rel_stats_time

def remove_all_patterns_from_str(s, patterns):
    for pattern in patterns:
        if pattern in s:
            s = s.replace(pattern, "")
    return s

def remove_row_from_table(table_string, row_index):
    rows = table_string.split('\n')
    if row_index < len(rows):
        del rows[row_index]
    return '\n'.join(rows)

def dict_to_table(data, scene, header=True):
    table_data = [["Scene", *data.keys()], [f"{scene}", *data.values()]]
    table_data = [row + ["\\\\"] for row in table_data]

    table = tabulate(table_data, tablefmt="plain")

    if not header:
        table = remove_row_from_table(table, 0)

    return table

def write_res_table(outfolder, res_str, scene_name, trial):
    res = res_str.split("|")
    res_dict = {}
    for r in res:
        k = r.split(":")[0]
        patterns_to_remove = ["\n", " ", ")", "("]
        k = remove_all_patterns_from_str(k, patterns_to_remove)

        v = r.split(":")[1]
        v = remove_all_patterns_from_str(v, patterns_to_remove)
        res_dict[k] = float(v)

    summtable_fnmae = os.path.join(outfolder, "../0_res.txt")
    if not os.path.isfile(summtable_fnmae): 
        f = open(summtable_fnmae, "w")
    else:
        f = open(summtable_fnmae, "a")
    if trial == 0:
        f.write("\n")

    table = dict_to_table(res_dict, scene_name, trial==0)
    f.write(table)
    f.write("\n")
    f.close()


def ate_real(traj_ref, tss_ref_us, traj_est, tstamps):
    evoGT = PoseTrajectory3D(
        positions_xyz=traj_ref[:,:3],
        # orientations_quat_wxyz=traj_ref[:,3:], # TODO wrong format: EVO uses wxyz, we use xyzw
        orientations_quat_wxyz=traj_ref[:, [6,3,4,5]],#pose存储的都是xyz xyzw
        timestamps=tss_ref_us/1e6)

    evoEst = PoseTrajectory3D(
        positions_xyz=traj_est[:,:3],
        # orientations_quat_wxyz=traj_est[:,3:], # TODO wrong format: EVO uses wxyz, we use xyzw
        orientations_quat_wxyz=traj_est[:, [6,3,4,5]],#pose存储的都是xyz xyzw
        timestamps=tstamps/1e6)

    if traj_ref.shape == traj_est.shape:
        assert np.all(tss_ref_us == tstamps)
        return ate(traj_ref, traj_est, tstamps)*100, evoGT, evoEst
    
    evoGT, evoEst = sync.associate_trajectories(evoGT, evoEst, max_diff=1)
    try:
        # Auto-detect and skip static frames for alignment
        skip_frames = detect_static_frames(evoEst)
        if skip_frames > 0:
            # Create new trajectories skipping initial static frames
            evoGT_truncated = PoseTrajectory3D(
                positions_xyz=evoGT.positions_xyz[skip_frames:],
                orientations_quat_wxyz=evoGT.orientations_quat_wxyz[skip_frames:],
                timestamps=evoGT.timestamps[skip_frames:]
            )
            evoEst_truncated = PoseTrajectory3D(
                positions_xyz=evoEst.positions_xyz[skip_frames:],
                orientations_quat_wxyz=evoEst.orientations_quat_wxyz[skip_frames:],
                timestamps=evoEst.timestamps[skip_frames:]
            )
            ape_trans = main_ape.ape(evoGT_truncated, evoEst_truncated, pose_relation=metrics.PoseRelation.translation_part, align=True, correct_scale=True)
        else:
            ape_trans = main_ape.ape(evoGT, evoEst, pose_relation=metrics.PoseRelation.translation_part, align=True, correct_scale=True)
    except:
        print("Alignment failed, computing ATE without alignment")
        ape_trans = main_ape.ape(evoGT, evoEst, pose_relation=metrics.PoseRelation.translation_part, align=False, correct_scale=False)
    evoATE = ape_trans.stats["rmse"]*100
    return evoATE, evoGT, evoEst


def make_evo_traj(poses_N_x_7, tss_us):
    assert poses_N_x_7.shape[1] == 7
    assert poses_N_x_7.shape[0] > 10
    assert tss_us.shape[0] == poses_N_x_7.shape[0]

    traj_evo = PoseTrajectory3D(
        positions_xyz=poses_N_x_7[:,:3],
        # orientations_quat_wxyz=poses_N_x_7[:,3:],
        orientations_quat_wxyz = poses_N_x_7[:, [6,3,4,5]],#pose存储的都是xyz xyzw
        timestamps=tss_us/1e6)#转换为秒
    return traj_evo


@torch.no_grad()
#将rpg_eval置为false            
def log_results(data, hyperparam, all_results, results_dict_scene, figures, 
                plot=False, save=True, return_figure=False, rpg_eval=False, stride=1, 
                calib1_eds=None, camID_tumvie=None, outdir=None, expname="", max_diff_sec=0.01,_n_to_align=-1, avg_fps=None):
    # results: dict of (scene, list of results)
    # all_results: list of all raw_results

    # unpack data
    traj_GT, tss_GT_us, traj_est, tss_est_us = data
    print("Saving raw traj_est before ATE check...")
    save_trajectory_tum_format((traj_est, tss_est_us), "results/debug/precheck_traj_est.txt")

    train_step, net, dataset_name, scene, trial, cfg, args = hyperparam

    # create folders
    if train_step is None:
        if isinstance(net, str) and ".pth" in net:
            train_step = os.path.basename(net.split(".")[0])
        else:
            train_step = -1
    scene_name = '_'.join(scene.split('/')[1:]).title() if "/P0" in scene else scene.title()
    if outdir is None:
        outdir = "results"
    outfolder = make_outfolder(outdir, dataset_name, expname, scene_name, trial, train_step, stride, calib1_eds, camID_tumvie)

    # save cfg & args to outfolder
    if cfg is not None:
        with open(f"{outfolder}/cfg.yaml", 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False)
    #将所有的参数文件写下来
    if args is not None:
        if args is not None:
            with open(f"{outfolder}/args.yaml", 'w') as f:
                yaml.dump(vars(args), f, default_flow_style=False)

    # compute ATE
    ate_score, evoGT, evoEst = ate_real(traj_GT, tss_GT_us, traj_est, tss_est_us)#已经更改里面的xyzw顺序
    # all_results.append(ate_score)
    # results_dict_scene[scene].append(ate_score)
    
    # following https://github.com/arclab-hku/Event_based_VO-VIO-SLAM/issues/5
    evoGT = make_evo_traj(traj_GT, tss_GT_us)#已经更改里面的xyzw顺序
    evoEst = make_evo_traj(traj_est, tss_est_us)
    gtlentraj = evoGT.get_infos()["path length (m)"]#获取轨迹长度
    evoGT, evoEst = sync.associate_trajectories(evoGT, evoEst, max_diff=1)
    # ape_trans = main_ape.ape(copy.deepcopy(evoGT), copy.deepcopy(evoEst), pose_relation=metrics.PoseRelation.translation_part, align=True, correct_scale=True)
    # 新增参数_n_to_align
    try:
        # Auto-detect static frames if _n_to_align is -1
        if _n_to_align == -1:
            skip_frames = detect_static_frames(evoEst)
            if skip_frames > 0:
                # Create truncated trajectories skipping initial static frames
                evoGT_truncated = PoseTrajectory3D(
                    positions_xyz=evoGT.positions_xyz[skip_frames:],
                    orientations_quat_wxyz=evoGT.orientations_quat_wxyz[skip_frames:],
                    timestamps=evoGT.timestamps[skip_frames:]
                )
                evoEst_truncated = PoseTrajectory3D(
                    positions_xyz=evoEst.positions_xyz[skip_frames:],
                    orientations_quat_wxyz=evoEst.orientations_quat_wxyz[skip_frames:],
                    timestamps=evoEst.timestamps[skip_frames:]
                )
                ape_trans = main_ape.ape(evoGT_truncated, evoEst_truncated, pose_relation=metrics.PoseRelation.translation_part, align=True, correct_scale=True)
            else:
                ape_trans = main_ape.ape(copy.deepcopy(evoGT), copy.deepcopy(evoEst), pose_relation=metrics.PoseRelation.translation_part, align=True, correct_scale=True)
        else:
            ape_trans = main_ape.ape(copy.deepcopy(evoGT), copy.deepcopy(evoEst), pose_relation=metrics.PoseRelation.translation_part, align=True, n_to_align=_n_to_align, correct_scale=True)
    except GeometryException as e:
        print(f"Alignment failed, computing ATE without alignment: {e}")
        ape_trans = main_ape.ape(copy.deepcopy(evoGT), copy.deepcopy(evoEst), pose_relation=metrics.PoseRelation.translation_part, align=False, correct_scale=False)
    # 用红色字体显示
    print(f"\033[31m EVO结果：{ape_trans}\033[0m");
    if _n_to_align!=-1:
        print(f"align {_n_to_align} frames")
    MPE = ape_trans.stats["mean"] / gtlentraj * 100
    print(f"MPE is {MPE:.02f}") #注意只保留两位小数
    evoATE = ape_trans.stats["rmse"]*100
    if _n_to_align==-1:#只有为-1时才进行assert
        assert abs(evoATE-ate_score) < 1e-5
    R_rmse_deg = -1.0

    # 用于最后输出所有的结果
    all_results.append(MPE)
    results_dict_scene[scene].append(MPE)

    if save:#将结果保存，必然为true
        Path(f"{outfolder}").mkdir(exist_ok=True)
        print("Saving trajectory to:", f"{outfolder}/{scene_name}_Trial{trial:02d}.txt")
        print("traj_est shape:", traj_est.shape)
        print("tss_est_us shape:", tss_est_us.shape)
        print("First few timestamps:", tss_est_us[:3])
        print("First few poses:", traj_est[:3])

        save_trajectory_tum_format((traj_est, tss_est_us), f"{outfolder}/{scene_name}_Trial{trial:02d}.txt")
    
    if save and avg_fps is not None:  # save fps
        with open(f"{outfolder}/fps.txt", "w") as f:
            f.write(f"{avg_fps:.02f}")

    if rpg_eval:#传入的都是false，不用管

        fnamegt, fnameest = run_rpg_eval(outfolder, traj_GT, tss_GT_us, traj_est, tss_est_us)
        abs_stats, rel_stats, _ = load_stats_rpg_results(outfolder)

        # abs errs
        ate_rpg = abs_stats["trans"]["rmse"]*100
        print(f"ate_rpg: {ate_rpg:.04f}, ate_real (EVO): {ate_score:.04f}")
        # assert abs(ate_rpg-ate_score)/ate_rpg < 0.1 # 10%
        R_rmse_deg = abs_stats["rot"]["rmse"]
        MTE_m = abs_stats["trans"]["mean"]

        # traj_GT_inter = interpolate_traj_at_tss(traj_GT, tss_GT_us, tss_est_us)
        # ate_inter, _, _ = ate_real(traj_GT_inter, tss_est_us, traj_est, tss_est_us)
        
        res_str = f"\nATE[cm]: {ate_score:.03f} | R_rmse[deg]: {R_rmse_deg:.03f} | MPE[%/m]: {MPE:.03f} \n"
        # res_str += f"MTE[m]: {MTE_m:.03f} | (ATE_int[cm]: {ate_inter:.02f} | ATE_rpg[cm]: {ate_rpg:.02f}) \n"

        write_res_table(outfolder, res_str, scene_name, trial)
    else:
        res_str = f"\nATE[cm]: {ate_score:.02f} | MPE[%/m]: {MPE:.02f}"

    if plot:
        Path(f"{outfolder}/").mkdir(exist_ok=True)
        pdfname = f"{outfolder}/../{scene_name}_Trial{trial+1:02d}_exp_{expname}_step_{train_step}_stride_{stride}.pdf"
        plot_trajectory((traj_est, tss_est_us/1e6), (traj_GT, tss_GT_us/1e6), 
                        f"{dataset_name} {expname} {scene_name.replace('_', ' ')} Trial #{trial} {res_str}",
                        pdfname, align=True, correct_scale=True, max_diff_sec=max_diff_sec)
        shutil.copy(pdfname, f"{outfolder}/{scene_name}_Trial{trial+1:02d}_step_{train_step}_stride_{stride}.pdf")#将pdf文件复制到outfolder文件夹下

        # [DEBUG]
        pdfname = f"{outfolder}/GT_{scene_name}_Trial{trial+1:02d}_exp_{expname}_step_{train_step}_stride_{stride}.pdf"
        plot_trajectory((traj_GT, tss_GT_us/1e6), (traj_GT, tss_GT_us/1e6), 
                        f"{dataset_name} {expname} {scene_name.replace('_', ' ')} Trial #{trial} {res_str}",
                        pdfname, align=True, correct_scale=True, max_diff_sec=max_diff_sec)

    if return_figure:
        fig = fig_trajectory((traj_est, tss_est_us/1e6), (traj_GT, tss_GT_us/1e6), f"{dataset_name} {scene_name.replace('_', ' ')} {res_str})",
                            return_figure=True, max_diff_sec=max_diff_sec)
        figures[f"{dataset_name}_{scene_name}"] = fig

    return all_results, results_dict_scene, figures, outfolder



@torch.no_grad()
def write_raw_results(all_results, outfolder):
    # all_results: list of all raw_results
    os.makedirs(os.path.join(f"{outfolder}/../raw_results"), exist_ok=True)
    with open(os.path.join(f"{outfolder}/../raw_results", datetime.datetime.now().strftime('%m-%d-%I%p.txt')), "w") as f:
        f.write(','.join([str(x) for x in all_results]))

@torch.no_grad()
def compute_median_results(results, all_results, dataset_name, outfolder=None):
    # results: dict of (scene, list of results)
    # all_results: list of all raw_results
        
    results_dict = dict([(f"{dataset_name}/{k}", np.median(v)) for (k, v) in results.items()])
    results_dict["AUC"] = np.maximum(1 - np.array(all_results), 0).mean()

    xs = []
    for scene in results:
        x = np.median(results[scene])
        xs.append(x)
    results_dict["AVG"] = np.mean(xs) / 100.0 # cm -> m

    if outfolder is not None:
        with open(os.path.join(f"{outfolder}/../../results_dict_{datetime.datetime.now().strftime('%m-%d-%I%p.txt')}"), 'w') as f:
            k0 = list(results.keys())[0]
            num_runs = len(results[k0])#获取运行的次数，也就是行数，每个结果运行了多少次
            f.write(' & '.join([str(k) for k in results.keys()]))
            f.write('\n')

 
            for i in range(num_runs):
                print(f"{[str(v[i]) for v in results.values()]}")
                f.write(' & '.join([str(v[i]) for v in results.values()]))
                f.write('\n')

            f.write(f"Medians\n")
            # for i in range(num_runs):
                # print(f"{[str(v[i]) for v in results.values()]}")
            f.write(' & '.join([str(np.median(v)) for v in results.values()]))
            f.write('\n')

            f.write('\n\n')

    return results_dict


@torch.no_grad()
def compute_min_results(results, all_results, dataset_name, outfolder=None):
    # results: dict of (scene, list of results)
    # all_results: list of all raw_results
        
    results_dict = dict([(f"{dataset_name}/{k}", np.min(v)) for (k, v) in results.items()])
    results_dict["AUC"] = np.maximum(1 - np.array(all_results), 0).mean()

    xs = []
    for scene in results:
        x = np.min(results[scene])
        xs.append(x)
    results_dict["AVG"] = np.mean(xs) / 100.0 # cm -> m

    if outfolder is not None:
        with open(os.path.join(f"{outfolder}/../../results_dict_{datetime.datetime.now().strftime('%m-%d-%I%p.txt')}"), 'w') as f:
            k0 = list(results.keys())[0]
            num_runs = len(results[k0])#获取运行的次数，也就是行数，每个结果运行了多少次
            f.write(' & '.join([str(k) for k in results.keys()]))
            f.write('\n')

 
            for i in range(num_runs):
                print(f"{[str(v[i]) for v in results.values()]}")
                f.write(' & '.join([str(v[i]) for v in results.values()]))
                f.write('\n')

            f.write(f"min\n")
            # for i in range(num_runs):
                # print(f"{[str(v[i]) for v in results.values()]}")
            f.write(' & '.join([str(np.min(v)) for v in results.values()]))
            f.write('\n')

            f.write('\n\n')

    return results_dict


from devo.plot_utils import make_traj, best_plotmode
from evo.tools import plot
from evo.core.geometry import GeometryException
import matplotlib.pyplot as plt
def plot_four_trajectory(pred_traj, ESVO_AA_traj, ESVO_traj, gt_traj=None, title="", filename="", align=True, correct_scale=True, max_diff_sec=0.01):
    pred_traj = make_traj(pred_traj)

    # 转换为PoseTrajectory3D
    ESVO_AA_traj = make_traj(ESVO_AA_traj) # ESVO_AA的轨迹
    ESVO_traj= make_traj(ESVO_traj) # ESVO的轨迹

    if gt_traj is not None:
        gt_traj = make_traj(gt_traj)
        gt_traj, pred_traj = sync.associate_trajectories(gt_traj, pred_traj, max_diff=max_diff_sec)

        if align:
            try:
                pred_traj.align(gt_traj, correct_scale=correct_scale)
            except GeometryException as e:
                print("Plotting error:", e)

    plot_collection = plot.PlotCollection("PlotCol")
    fig = plt.figure(figsize=(8, 8))
    # plot_mode = best_plotmode(gt_traj if (gt_traj is not None) else pred_traj)
    plot_mode = plot.PlotMode.xy
    ax = plot.prepare_axis(fig, plot_mode)
    ax.set_title(title)
    if gt_traj is not None:
        plot.traj(ax, plot_mode, gt_traj, '--', 'gray', "Ground Truth")
    plot.traj(ax, plot_mode, pred_traj, '-', 'blue', "DEVO")
    # 下面两个是新增的
    plot.traj(ax, plot_mode, ESVO_AA_traj, '-', 'green', "ESVO_AA")
    plot.traj(ax, plot_mode, ESVO_traj, '-', 'red', "ESVO")
    
    plot_collection.add_figure("traj (error)", fig)
    plot_collection.export(filename, confirm_overwrite=False)
    plt.close(fig=fig)
    # print(f"Saved {filename}")

def show_trajectory_comparison(pred_traj, ESVO_AA_traj=None, ESVO_traj=None, ESVIO_traj=None, ESIO_traj=None, gt_traj=None, align=True, _n_to_align=-1,correct_scale=True, max_diff_sec=0.01,title="", filename=""):
    
    pred_traj = make_traj(pred_traj)#输入的结果

    # 转换为PoseTrajectory3D
    if ESVO_AA_traj is not None:
        ESVO_AA_traj = make_traj(ESVO_AA_traj) # ESVO_AA的轨迹
    if ESVO_traj is not None:
        ESVO_traj= make_traj(ESVO_traj) # ESVO的轨迹
    if ESVIO_traj is not None:
        ESVIO_traj= make_traj(ESVIO_traj)
    if ESIO_traj is not None:
        ESIO_traj= make_traj(ESIO_traj)

    if gt_traj is not None:
        gt_traj = make_traj(gt_traj)
        gt_traj, pred_traj = sync.associate_trajectories(gt_traj, pred_traj, max_diff=max_diff_sec)#将两个轨迹对齐(时间维度上的)

        if align:#进行轨迹对齐
            try:
                pred_traj.align(gt_traj, correct_scale=correct_scale, n=_n_to_align)
            except GeometryException as e:
                print("Plotting error:", e)

    if ESVIO_traj is not None:
        gt_traj_esvio, ESVIO_traj = sync.associate_trajectories(gt_traj, ESVIO_traj, max_diff=max_diff_sec)

        ape_trans = main_ape.ape(copy.deepcopy(gt_traj_esvio), copy.deepcopy(ESVIO_traj), pose_relation=metrics.PoseRelation.translation_part, align=True,n_to_align=_n_to_align, correct_scale=True)
        print(f"\033[31m ESVIO结果：{ape_trans}\033[0m");

        ESVIO_traj.align(gt_traj_esvio, correct_scale=correct_scale, n=_n_to_align)
    if ESIO_traj is not None:
        gt_traj_esio, ESIO_traj = sync.associate_trajectories(gt_traj, ESIO_traj, max_diff=max_diff_sec)

        ape_trans = main_ape.ape(copy.deepcopy(gt_traj_esio), copy.deepcopy(ESIO_traj), pose_relation=metrics.PoseRelation.translation_part, align=True,n_to_align=_n_to_align, correct_scale=True)
        print(f"\033[31m ESIO结果：{ape_trans}\033[0m");

        ESIO_traj.align(gt_traj_esio, correct_scale=correct_scale, n=_n_to_align)

    plot_collection = plot.PlotCollection("PlotCol")
    fig = plt.figure(figsize=(8, 8))
    plot_mode = best_plotmode(gt_traj if (gt_traj is not None) else pred_traj)
    # plot_mode = plot.PlotMode.xy
    ax = plot.prepare_axis(fig, plot_mode)
    ax.grid(False)  # 禁用网格

    # 设置背景为白色
    fig.patch.set_facecolor('white')  # 图像整体背景
    ax.set_facecolor('white')  # 坐标轴区域背景

    # 设置坐标轴颜色为黑色
    ax.spines['top'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['right'].set_color('black')

    # ax.set_title(title)

    if gt_traj is not None:
        plot.traj(ax, plot_mode, gt_traj, ':', 'black', "Ground Truth")

    plot.traj(ax, plot_mode, pred_traj, '-', 'green', "DEIO")
    # 下面是新增的
    if ESVO_AA_traj is not None:
        plot.traj(ax, plot_mode, ESVO_AA_traj, '--', 'pink', "ESVO_AA")
    if ESVO_traj is not None:
        plot.traj(ax, plot_mode, ESVO_traj, '--', 'red', "ESVO")
    if ESVIO_traj is not None:
        plot.traj(ax, plot_mode, ESVIO_traj, '--', 'gray', "ESVIO")
    if ESIO_traj is not None:
        plot.traj(ax, plot_mode, ESIO_traj, '--', 'blue', "ESIO")
    
    plot_collection.add_figure("traj (error)", fig)

    # 如果filename不是空的，就保存
    if filename!="":
        plot_collection.export(filename, confirm_overwrite=False)
        plt.close(fig=fig)
        # print(f"Saved {filename}")
    
    plt.show() #显示图像
