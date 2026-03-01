from copy import deepcopy

import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import numpy as np
from evo.core import sync
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import plot
from evo.core.geometry import GeometryException
from pathlib import Path


def make_traj(args) -> PoseTrajectory3D:
    if isinstance(args, tuple):
        traj, tstamps = args
        # return PoseTrajectory3D(positions_xyz=traj[:,:3], orientations_quat_wxyz=traj[:,3:], timestamps=tstamps)
        #存的格式为xyz xyzw
        return PoseTrajectory3D(positions_xyz=traj[:,:3], orientations_quat_wxyz=traj[:, [6,3,4,5]], timestamps=tstamps)#就是把xyzw，改为wxyz来用
    assert isinstance(args, PoseTrajectory3D), type(args)
    return deepcopy(args)

def best_plotmode(traj):
    _, i1, i2 = np.argsort(np.var(traj.positions_xyz, axis=0))
    plot_axes = "xyz"[i2] + "xyz"[i1]
    # plot_axes="xy" #单纯画xy平面
    return getattr(plot.PlotMode, plot_axes)

def plot_trajectory(pred_traj, gt_traj=None, title="", filename="", align=True, correct_scale=True, max_diff_sec=0.01):
    pred_traj = make_traj(pred_traj)

    if gt_traj is not None:
        gt_traj = make_traj(gt_traj)
        gt_traj, pred_traj = sync.associate_trajectories(gt_traj, pred_traj, max_diff=max_diff_sec)

        if align:
            try:
                pred_traj.align(gt_traj, correct_scale=correct_scale)#!注意此处默认了n=-1，用全部的pose进行对齐
            except GeometryException as e:
                print("Plotting error:", e)

    plot_collection = plot.PlotCollection("PlotCol")
    fig = plt.figure(figsize=(8, 8))
    plot_mode = best_plotmode(gt_traj if (gt_traj is not None) else pred_traj)
    ax = plot.prepare_axis(fig, plot_mode)
    ax.set_title(title)
    if gt_traj is not None:
        plot.traj(ax, plot_mode, gt_traj, '--', 'gray', "Ground Truth")
    plot.traj(ax, plot_mode, pred_traj, '-', 'blue', "Predicted")
    plot_collection.add_figure("traj (error)", fig)
    plot_collection.export(filename, confirm_overwrite=False)
    plt.close(fig=fig)
    # print(f"Saved {filename}")

# TODO refactor: merge in previous function plot_trajectory() with figure=False and save=False
def fig_trajectory(pred_traj, gt_traj=None, title="", filename="", align=True, correct_scale=True, save=False, return_figure=False, max_diff_sec=0.01):
    plt.switch_backend('Agg') # TODO instead install evo from source to use qt5agg backend
    
    pred_traj = make_traj(pred_traj)

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
    plot_mode = best_plotmode(gt_traj if (gt_traj is not None) else pred_traj)
    ax = plot.prepare_axis(fig, plot_mode)
    ax.set_title(title)
    if gt_traj is not None:
        plot.traj(ax, plot_mode, gt_traj, '--', 'gray', "Ground Truth")
    plot.traj(ax, plot_mode, pred_traj, '-', 'blue', "Predicted")
    
    plot_collection.add_figure("traj (error)", fig)
    
    if save:
        plot_collection.export(filename, confirm_overwrite=False)
    if return_figure:
        return fig
    plt.close(fig=fig)
    return None

# 将结果保存为tum格式
def save_trajectory_tum_format(traj, filename):
    traj = make_traj(traj)
    tostr = lambda a: ' '.join(map(str, a))
    with Path(filename).open('w') as f:
        for i in range(traj.num_poses):
            f.write(f"{traj.timestamps[i]} {tostr(traj.positions_xyz[i])} {tostr(traj.orientations_quat_wxyz[i][[1,2,3,0]])}\n")#由于存入的为wxyz，所以需要调整顺序wxyz->xyzw
    # print(f"Saved {filename}")
