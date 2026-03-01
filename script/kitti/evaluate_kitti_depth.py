import glob
import os
from pathlib import Path
import cv2
import evo.main_ape as main_ape
import numpy as np
import torch
import quaternion
import math

# 处理服务器中evo的可视化问题
import evo
from evo.tools.settings import SETTINGS
SETTINGS['plot_backend'] = 'Agg'

from evo.core import sync
from evo.core.metrics import PoseRelation
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface

from dpvo.config import cfg
from dpvo.utils import Timer

from utils.load_utils import load_gt_us
from utils.eval_utils import log_results, compute_median_results, VO_run

SKIP = 0

def show_image(image, t=0):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(t)

def load_psmnet_depth_for_frame(psmnet_dir, frame_id, target_height, target_width, sequence_name=None):
    """
    Load PSMNet depth map for specific frame
    Handles multiple directory structures and naming conventions
    """
    if psmnet_dir is None:
        return None
    
    # ✅ Try ALL possible path combinations
    possible_files = []
    
    # Pattern 1: depth_maps/sequence/frame
    if sequence_name:
        possible_files.extend([
            psmnet_dir / "depth_maps" / sequence_name / f"{frame_id:06d}.npy",
            psmnet_dir / "depth_maps" / sequence_name / f"{frame_id:06d}_depth.npy",
            psmnet_dir / "depth_maps" / sequence_name / f"{frame_id:06d}.pfm",
        ])
    
    # Pattern 2: sequence/depth_maps/frame
    if sequence_name:
        possible_files.extend([
            psmnet_dir / sequence_name / "depth_maps" / f"{frame_id:06d}.npy",
            psmnet_dir / sequence_name / "depth_maps" / f"{frame_id:06d}_depth.npy",
            psmnet_dir / sequence_name / "depth_maps" / f"{frame_id:06d}.pfm",
        ])
    
    # Pattern 3: sequence/frame (no depth_maps subdir)
    if sequence_name:
        possible_files.extend([
            psmnet_dir / sequence_name / f"{frame_id:06d}.npy",
            psmnet_dir / sequence_name / f"{frame_id:06d}_depth.npy",
            psmnet_dir / sequence_name / f"{frame_id:06d}.pfm",
        ])
    
    # Pattern 4: depth_maps/frame (no sequence subdir)
    possible_files.extend([
        psmnet_dir / "depth_maps" / f"{frame_id:06d}.npy",
        psmnet_dir / "depth_maps" / f"{frame_id:06d}_depth.npy",
        psmnet_dir / "depth_maps" / f"{frame_id:06d}.pfm",
    ])
    
    # Pattern 5: flat structure
    possible_files.extend([
        psmnet_dir / f"{frame_id:06d}.npy",
        psmnet_dir / f"{frame_id:06d}_depth.npy",
        psmnet_dir / f"{frame_id:06d}.pfm",
    ])
    
    # ✅ DEBUG for first 3 frames
    if frame_id < 3:
        print(f"\n[LOAD_DEPTH] Frame {frame_id}: Trying {len(possible_files)} paths...")
    
    # Find first existing file
    depth_file = None
    for file_path in possible_files:
        if file_path.exists():
            depth_file = file_path
            if frame_id < 3:
                print(f"  ✅ FOUND: {depth_file}")
            break
    
    if depth_file is None:
        if frame_id < 3:
            print(f"  ❌ NOT FOUND. Tried:")
            for f in possible_files[:5]:
                print(f"     - {f}")
            print(f"     ... and {len(possible_files)-5} more")
        return None
    
    try:
        # Load depth
        if str(depth_file).endswith('.pfm'):
            # Handle PFM format
            import re
            with open(depth_file, 'rb') as f:
                header = f.readline().decode('utf-8').rstrip()
                if header not in ['PF', 'Pf']:
                    raise Exception('Not a PFM file.')
                
                dim_match = re.match(r'^(\d+)\s(\d+)\s$', f.readline().decode('utf-8'))
                if dim_match:
                    width, height = map(int, dim_match.groups())
                else:
                    raise Exception('Malformed PFM header.')
                
                scale = float(f.readline().decode('utf-8').rstrip())
                data = np.fromfile(f, '<f')
                
            depth = np.reshape(data, (height, width))
            depth = np.flipud(depth)  # PFM is bottom-to-top
        else:
            depth = np.load(depth_file).astype(np.float32)
        
        # Check if disparity (large values)
        if np.median(depth[depth > 0]) > 100:
            # Convert disparity to depth
            baseline_focal = 379.8145  # KITTI calibration
            depth = np.where(depth > 1.0, baseline_focal / depth, 0.0)
            depth = np.clip(depth, 1.0, 80.0)
            
            if frame_id < 3:
                print(f"  🔧 Converted disparity→depth, median: {np.median(depth[depth > 1.0]):.2f}m")
        
        # Resize if needed
        if depth.shape != (target_height, target_width):
            import cv2
            depth = cv2.resize(depth, (target_width, target_height))
        
        if frame_id < 3:
            valid = depth[depth > 1.0]
            print(f"  ✅ Loaded: shape={depth.shape}, range=[{valid.min():.2f}, {valid.max():.2f}]m")
        
        return depth
        
    except Exception as e:
        print(f"⚠️ Error loading {depth_file}: {e}")
        import traceback
        traceback.print_exc()
        return None


# KITTI Images+IMU DEIO2 runner with PSMNet integration
# Replace the problematic section in your evaluation script (around line 119)


@torch.no_grad()
def run_DEIO2_KITTI(datapath_val, cfg, network, viz=False, iterator=None, _all_imu=None, 
                    _all_gt=None, _all_gt_keys=None, timing=False, H=376, W=1024, 
                    viz_flow=False, scale=1.0, psmnet_base_dir=None, sequence_name=None, **kwargs):
    """KITTI Images+IMU version using DEIO (DBA) with optional PSMNet depth maps"""

    from dpvo.utils import Timer
    from devo.dba import DBA as DEIO2
    import gtsam
    from tqdm import tqdm

    # Instantiate DBA (images path; events=False)
    slam = DEIO2(cfg, network, evs=False, ht=H, wd=W, viz=viz, viz_flow=viz_flow, **kwargs)


    # ---- IMU setup ----
    # Check if IMU is enabled in config AND IMU data is available
    imu_enabled = getattr(cfg, "ENALBE_IMU", False) and (_all_imu is not None)
    
    if imu_enabled:
        print("🚀 IMU-enabled mode: Setting up IMU parameters")
        
        Ti1c = np.array(cfg.Ti1c).reshape(4, 4)
        if getattr(cfg, "ENALBE_INV", False):
            Ti1c = np.linalg.inv(Ti1c)

        IMU_noise = np.array([
            cfg.accel_noise_sigma, cfg.gyro_noise_sigma,
            cfg.accel_bias_sigma, cfg.gyro_bias_sigma
        ])

        slam.Ti1c = Ti1c
        #slam.Tbc  = gtsam.Pose3(slam.Ti1c)
        slam.Tbc  = gtsam.Pose3(np.linalg.inv(Ti1c))
        slam.state.set_imu_params((IMU_noise * 1.0).tolist())

        # IMU data → set on the DBA
        # Ensure strictly non-decreasing timestamps
        time_diffs = np.diff(_all_imu[:, 0])
        if (time_diffs < 0).any():
            mask = np.concatenate([[True], time_diffs >= 0])
            _all_imu = _all_imu[mask]

        slam.all_imu     = _all_imu
        slam.all_gt      = _all_gt
        slam.all_gt_keys = _all_gt_keys
        
        # Configure for IMU mode
        slam.visual_only = False  # Enable IMU processing
        slam.ignore_imu = False   # Don't ignore IMU
        slam.imu_enabled = False  # Will be enabled after VI initialization
        
        print(f"✅ IMU setup complete: {len(_all_imu)} IMU measurements from {_all_imu[0,0]:.3f}s to {_all_imu[-1,0]:.3f}s")
        
    else:
        if getattr(cfg, "ENALBE_IMU", False):
            print("⚠️ Warning: IMU enabled in config but IMU data not available, falling back to visual-only")
        else:
            print("ℹ️ Running in visual-only mode (IMU disabled in config)")
            
        # Configure for visual-only mode
        slam.visual_only = True
        slam.imu_enabled = False
        slam.ignore_imu = True
        
        # Set IMU data to None to avoid errors
        slam.all_imu = None
        slam.all_gt = _all_gt
        slam.all_gt_keys = _all_gt_keys
        
        # Initialize Ti1c for visual-only mode (identity transform)
        slam.Ti1c = np.eye(4)

    # ---- Depth / RoMeO knobs ----
    if psmnet_base_dir:
        print(f"🎯 PSMNet depths enabled from: {psmnet_base_dir}")
        slam.depth_weight = float(getattr(cfg, "DEPTH_WEIGHT", 0.05))
        print(f"   RoMeO depth weight λ = {slam.depth_weight}")
    else:
        print("⚠️  No PSMNet depths provided - using standard depth init")

    frame_count = 0
    depth_success_count = 0

    for i, (image, intrinsics, t) in enumerate(tqdm(iterator)):
        if timing and i == 0:
            i = i + 1
            t0 = torch.cuda.Event(enable_timing=True)
            t1 = torch.cuda.Event(enable_timing=True)
            t0.record()

        # Load PSMNet map for this frame if available
        psmnet_depth = None
        if psmnet_base_dir:
            psmnet_depth = load_psmnet_depth_for_frame(
                psmnet_base_dir, frame_count, H, W, sequence_name
            )
            if psmnet_depth is not None:
                depth_success_count += 1
                if frame_count % 100 == 0:
                    valid = psmnet_depth[psmnet_depth > 1.0]
                    if valid.size > 0:
                        print(f"📊 Frame {frame_count}: PSMNet depth {valid.min():.2f}-{valid.max():.2f}m, median={np.median(valid):.2f}m")

        # Call DBA (passes depth to the integrated injection/regularization)
        with Timer("DEIO_KITTI", enabled=timing):
            slam(t, image, intrinsics, scale=scale, psmnet_depth=psmnet_depth)

        frame_count += 1

    # A few extra local BA iterations to settle
    for _ in range(12):
        slam.update()

    #poses, tstamps = slam.terminate()
    poses, tstamps = slam.terminate()

    # Report results
    if psmnet_base_dir:
        success_rate = 100.0 * depth_success_count / max(frame_count, 1)
        print("🎯 PSMNet Integration Summary:")
        print(f"   Processed frames: {frame_count}")
        print(f"   Successful depth loads: {depth_success_count}")
        print(f"   Success rate: {success_rate:.1f}%")
        print(f"   Depth prior frames cached: {len(getattr(slam, 'depth_priors', {}))}")

    if timing:
        t1.record()
        torch.cuda.synchronize()
        dt = t0.elapsed_time(t1) / 1e3
        avg_fps = (i + 1) / dt
        mode_str = "IMU" if imu_enabled and not slam.visual_only else "Visual-only"
        psmnet_str = "+PSMNet" if psmnet_base_dir else ""
        print(f"{datapath_val}\nDEIO KITTI Images+{mode_str}{psmnet_str} {i+1} frames in {dt:.2f} sec, {avg_fps:.1f} FPS")
    else:
        avg_fps = None

    flowdata = slam.flow_data if viz_flow else None
    return poses, tstamps, flowdata, avg_fps

def load_kitti_preprocessed_data(sequence_path, side="left"):
    """Load preprocessed KITTI data with consistent timestamp handling"""
    
    gt_file = os.path.join(sequence_path, f"gt_stamped_{side}.txt")
    imu_file = os.path.join(sequence_path, f"imu_data.csv")
    
    print(f"Loading preprocessed KITTI data from {sequence_path}")
    
    # Load GT data (timestamps already in seconds)
    gt_data = np.loadtxt(gt_file)
    tss_traj_sec = gt_data[:, 0]  # Timestamps in seconds
    traj_hf = gt_data[:, 1:]      # Pose data
    
    # Load IMU data (convert ns to seconds)
    all_imu = np.loadtxt(imu_file, delimiter=',', skiprows=1)
    all_imu[:, 0] = all_imu[:, 0] / 1e9  # ns → seconds
    
    print(f"Loaded data ranges:")
    print(f"  GT: {tss_traj_sec[0]:.6f} to {tss_traj_sec[-1]:.6f} seconds ({len(tss_traj_sec)} poses)")
    print(f"  IMU: {all_imu[0, 0]:.6f} to {all_imu[-1, 0]:.6f} seconds ({len(all_imu)} measurements)")
    
    # Verify timestamps start near zero (relative timestamps)
    if all_imu[0, 0] > 1000:  # If first timestamp > 1000 seconds, likely absolute
        raise ValueError("Timestamps appear to be absolute, not relative. Re-run preprocessing.")
    
    # Sort IMU data by timestamp
    all_imu = all_imu[all_imu[:, 0].argsort()]
    
    # Create GT dictionary 
    all_gt = {}
    for ts_sec, pose in zip(tss_traj_sec, traj_hf):
        x, y, z = pose[0], pose[1], pose[2]
        qx, qy, qz, qw = pose[3], pose[4], pose[5], pose[6]
        
        # Create transformation matrix
        q = quaternion.from_float_array([float(qw), float(qx), float(qy), float(qz)])
        R_mat = quaternion.as_rotation_matrix(q)
        T = np.eye(4)
        T[0:3, 0:3] = R_mat
        T[0:3, 3] = [float(x), float(y), float(z)]
        
        all_gt[float(ts_sec)] = {'T': T}
    
    all_gt_keys = np.array(sorted(all_gt.keys()))
    
    print(f"Data verification:")
    print(f"  Sample IMU timestamp: {all_imu[0, 0]:.6f}")
    print(f"  Sample GT timestamp: {all_gt_keys[0]:.6f}")
    print(f"  Sample IMU data: {all_imu[0, 1:4]} (gyro), {all_imu[0, 4:7]} (accel)")
    
    # Convert GT timestamps to microseconds for compatibility with existing code
    tss_traj_us = tss_traj_sec * 1e6
    
    return all_gt, all_gt_keys, traj_hf, tss_traj_us, all_imu

def create_kitti_image_iterator(sequence_path, side="left", stride=1, target_H=376, target_W=1024):
    """
    Yield (image_tensor[C,H,W], intrinsics_tensor[fx,fy,cx,cy], timestamp_seconds)
    from preprocessed KITTI folder:
      - images_undistorted_{side}/*.png
      - tss_imgs_sec_{side}.txt   (timestamps in seconds)
      - calib_undist_{side}.txt   (fx fy cx cy)
    """
    import os, glob, cv2
    import numpy as np
    import torch

    ts_file = os.path.join(sequence_path, f"tss_imgs_sec_{side}.txt")
    if not os.path.exists(ts_file):
        raise FileNotFoundError(f"Preprocessed timestamps not found: {ts_file}")

    timestamps_seconds = np.loadtxt(ts_file)[::stride]

    imagedir = os.path.join(sequence_path, f"images_undistorted_{side}")
    image_files = sorted(glob.glob(os.path.join(imagedir, "*.png")))[::stride]
    if len(image_files) == 0:
        raise FileNotFoundError(f"No images found in {imagedir}")

    calib_file = os.path.join(sequence_path, f"calib_undist_{side}.txt")
    fx, fy, cx, cy = np.loadtxt(calib_file)

    # Scale intrinsics to target resolution
    sample = cv2.imread(image_files[0])
    if sample is None:
        raise RuntimeError(f"Failed to read sample image: {image_files[0]}")
    orig_H, orig_W = sample.shape[:2]
    sx = target_W / float(orig_W)
    sy = target_H / float(orig_H)
    fx_s = fx * sx
    fy_s = fy * sy
    cx_s = cx * sx
    cy_s = cy * sy

    for img_path, t in zip(image_files, timestamps_seconds):
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[0] != target_H or img.shape[1] != target_W:
            img = cv2.resize(img, (target_W, target_H), interpolation=cv2.INTER_LINEAR)

        image_t = torch.from_numpy(img).permute(2, 0, 1).float().cuda()      # [3,H,W], float
        K_t     = torch.tensor([fx_s, fy_s, cx_s, cy_s], dtype=torch.float32, device=image_t.device)
        yield image_t, K_t, float(t)



def get_kitti_camera_imu_extrinsics():
    """Get KITTI camera-IMU extrinsics (synthetic/estimated)"""
    # KITTI doesn't have real IMU, so we use an estimated transformation
    # This is a typical camera-IMU transformation for automotive applications
    
    # Camera is roughly aligned with vehicle, IMU typically in center
    # Assume camera is ~1.65m high, ~0.3m back from front
    Ti1c = np.array([
        [ 1.0,  0.0,  0.0,  0.0],    # No rotation (aligned)
        [ 0.0,  1.0,  0.0,  0.0],
        [ 0.0,  0.0,  1.0,  0.0],
        [ 0.0,  0.0,  0.0,  1.0]
    ])
    
    return Ti1c.flatten().tolist()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed_kittidir', default="datasets/KITTI_processed", help="Processed KITTI dataset root directory")
    parser.add_argument('--network', type=str, default='dpvo.pth')
    parser.add_argument('--config', default="config/kitti_dpvo_imu.yaml")
    parser.add_argument('--stride', type=int, default=2)
    parser.add_argument('--viz', action="store_true")
    parser.add_argument('--trials', type=int, default=1)
    parser.add_argument('--backend_thresh', type=float, default=64.0)
    parser.add_argument('--plot', action="store_true")
    parser.add_argument('--opts', nargs='+', default=[])
    parser.add_argument('--save_trajectory', action="store_true")
    parser.add_argument('--sequences', nargs='+', type=int, default=None, help="Specific sequences to run")
    
    # PSMNet depth integration arguments
    parser.add_argument('--psmnet_dir', type=Path, default=None,
                       help='Base directory containing PSMNet depths (e.g., /path/to/PSMNet/output)')
    parser.add_argument('--depth_weight', type=float, default=0.05,
                       help='RoMeO depth constraint weight (lambda parameter)')
    parser.add_argument('--enable_romeo', action="store_true", default=True,
                       help='Enable RoMeO depth constraints (default: True)')
    
    args = parser.parse_args()

    cfg.merge_from_file(args.config)
    cfg.BACKEND_THRESH = args.backend_thresh
    cfg.merge_from_list(args.opts)
    cfg.resnet = False  # or True if you want to use ResNetFPN


    # PSMNet configuration
    if args.psmnet_dir:
        print(f"\033[42m🎯 PSMNet depths enabled from: {args.psmnet_dir}\033[0m")
        cfg.DEPTH_WEIGHT = args.depth_weight
        cfg.ROMEO_ENHANCED = args.enable_romeo
    else:
        print("\033[43m⚠️  No PSMNet depths provided - using standard initialization\033[0m")

    print("\033[42m Running KITTI Images+IMU DEIO with PSMNet integration...\033[0m ")
    print(cfg, "\n")

    # KITTI sequences (00-10 have ground truth)
    all_kitti_sequences = list(range(11))  # 0-10
    
    # Use specified sequences or all available
    if args.sequences:
        test_sequences = args.sequences
    else:
        test_sequences = [seq for seq in all_kitti_sequences 
                         if os.path.exists(os.path.join(args.processed_kittidir, f"{seq:02d}"))]
    
    print(f"Testing sequences: {[f'{seq:02d}' for seq in test_sequences]}")

    dataset_name = "KITTI/Images_IMU_PSMNet" if args.psmnet_dir else "KITTI/Images_IMU"
    train_step = None

    results_dict_scene, figures = {}, {}
    all_results = []
    
    for i, sequence_id in enumerate(test_sequences):
        sequence_name = f"{sequence_id:02d}"
        sequence_path = os.path.join(args.processed_kittidir, sequence_name)
        
        if not os.path.exists(sequence_path):
            print(f"Sequence {sequence_name} not found, skipping...")
            continue
            
        print(f"Eval on KITTI sequence {sequence_name}")
        results_dict_scene[sequence_name] = []

        for trial in range(args.trials):
            print(f"\nRunning trial {trial} of {sequence_name}...")
            
            try:
                # Load preprocessed data
                all_gt, all_gt_keys, traj_hf, tss_traj_us, all_imu = load_kitti_preprocessed_data(sequence_path)
                
                # Create image iterator - use compatible resolution
                iterator = create_kitti_image_iterator(sequence_path, stride=args.stride, 
                                                      target_H=376, target_W=1024)
                
                # Run KITTI Images+IMU DEIO with PSMNet
                traj_est, tstamps, flowdata, avg_fps = run_DEIO2_KITTI(
                    sequence_path, cfg, args.network, viz=args.viz,
                    iterator=iterator, _all_imu=all_imu, _all_gt=all_gt, 
                    _all_gt_keys=all_gt_keys, timing=True, 
                    H=376, W=1024, viz_flow=False, 
                    psmnet_base_dir=args.psmnet_dir, sequence_name=sequence_name
                )
                
                # Evaluation
                data = (traj_hf, tss_traj_us, traj_est, tstamps)
                hyperparam = (train_step, args.network, dataset_name, sequence_name, trial, cfg, args)
                
                # Log results
                all_results, results_dict_scene, figures, outfolder = log_results(
                    data, hyperparam, all_results, results_dict_scene, figures, 
                    plot=True, save=True, return_figure=False, stride=args.stride,
                    _n_to_align=1000, expname=sequence_name,
                    avg_fps=avg_fps
                )
                
            except Exception as e:
                print(f"Error in trial {trial} of {sequence_name}: {e}")
                import traceback
                traceback.print_exc()
                continue

        if sequence_name in results_dict_scene:
            print(f"{sequence_name}: {sorted(results_dict_scene[sequence_name])}")

    # Final results
    if results_dict_scene:
        results_dict = compute_median_results(results_dict_scene, all_results, dataset_name, outfolder=None)

        print("\n" + "="*50)
        print("FINAL KITTI RESULTS:")
        if args.psmnet_dir:
            print("WITH PSMNET DEPTH INTEGRATION")
        print("="*50)
        
        for k in results_dict:
            print(f"{k}: {results_dict[k]:.4f}")

        if len(results_dict) > 1:
            avg_result = np.mean(list(results_dict.values()))
            print(f"\nAVERAGE: {avg_result:.4f}")
    else:
        print("No successful results")

    print("Done!")
