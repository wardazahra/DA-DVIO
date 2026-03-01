#!/usr/bin/env python3

import os
import sys
import glob
import cv2
import numpy as np
import torch
import quaternion
from pathlib import Path
from tqdm import tqdm

# Visualization and evaluation
import evo
from evo.tools.settings import SETTINGS
SETTINGS['plot_backend'] = 'Agg'

from evo.core import sync
from evo.core.metrics import PoseRelation
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface
import evo.main_ape as main_ape

from dpvo.config import cfg
from dpvo.utils import Timer

# ============================================================
# SKIP INITIAL STATIC FRAMES
# ============================================================
SKIP_INITIAL_FRAMES = {
    'z02': 0,
    'z03': 0,
    # Add more sequences here if needed:
    # 'z04': 50,
}


def load_zed_depth_for_frame(depth_dir, frame_id, target_height, target_width, depth_files_list):
    """
    Load depth map for specific frame from ZED dataset
    Supports both .npy (PSMNet) and .png (ZED native) formats
    
    Auto-detects format from file extension:
    - .npy: PSMNet stereo predictions (float32, meters)
    - .png: ZED native depth (uint16, millimeters)
    """
    if depth_dir is None:
        print(f"❌ No depth directory provided")
        return None
    
    if frame_id >= len(depth_files_list):
        print(f"❌ Frame ID {frame_id} exceeds available depth files ({len(depth_files_list)})")
        return None
    
    # Get the depth file by frame index
    depth_file = depth_files_list[frame_id]
    
    if frame_id % 100 == 0:  # Print progress every 100 frames
        print(f"Loading depth file {frame_id}: {depth_file.name}")
    
    if not depth_file.exists():
        print(f"❌ Depth file not found: {depth_file}")
        return None
        
    try:
        # Auto-detect format from file extension
        file_ext = depth_file.suffix.lower()
        
        if file_ext == '.npy':
            # PSMNet format: numpy array in meters
            depth = np.load(depth_file).astype(np.float32)
            

            
        elif file_ext == '.png':
            # ZED native format: 16-bit PNG in millimeters
            depth_raw = cv2.imread(str(depth_file), cv2.IMREAD_ANYDEPTH)
            
            if depth_raw is None:
                print(f"❌ Failed to load PNG depth image: {depth_file}")
                return None
            
            # Convert from millimeters to meters
            depth = depth_raw.astype(np.float32) / 1000.0
            

            depth = np.clip(depth, 0.5, 30.0)  # ZED typical range
            


            
            
        else:
            print(f"❌ Unsupported depth format: {file_ext}")
            return None
        
        # Resize to match target resolution if needed
        if depth.shape != (target_height, target_width):
            depth = cv2.resize(depth, (target_width, target_height), 
                               interpolation=cv2.INTER_NEAREST)
        
        # Basic depth filtering - clip to reasonable range
        
        
        # Replace invalid depths (0 or NaN) with a placeholder
        depth[depth <= 0.1] = 0.0
        depth[np.isnan(depth)] = 0.0
        
        return depth
        
    except Exception as e:
        print(f"❌ Error loading depth {depth_file}: {e}")
        import traceback
        traceback.print_exc()
        return None

@torch.no_grad()
def run_DEIO_ZED_Depth(dataset_path, cfg, network, depth_dir=None, viz=False, 
                       stride=1, timing=True, H=376, W=1024, **kwargs):
    """Run DEIO visual-inertial SLAM on ZED dataset with depth evaluation"""
    
    from devo.dba import DBA as DEIO2
    import gtsam
    
    print(f"Running DEIO depth evaluation on ZED dataset: {dataset_path}")
    
    # Load preprocessed data
    from evaluate_zed import load_zed_preprocessed_data, create_zed_image_iterator
    all_gt, all_gt_keys, traj_hf, tss_traj_us, all_imu = load_zed_preprocessed_data(dataset_path)
    
    # Create image iterator
    iterator = create_zed_image_iterator(dataset_path, stride=stride, target_H=H, target_W=W)
    
    # ============================================================
    # SKIP INITIAL STATIC FRAMES
    # ============================================================
    seq_name = Path(dataset_path).name
    skip_initial = SKIP_INITIAL_FRAMES.get(seq_name, 0)
    
    if skip_initial > 0:
        print(f"⚠️  Skipping first {skip_initial} static frames for sequence: {seq_name}")
        # Convert iterator to list, skip frames, convert back to iterator
        iterator_list = list(iterator)
        iterator = iter(iterator_list[skip_initial:])
        print(f"✅ Starting from frame {skip_initial}, total frames: {len(iterator_list) - skip_initial}")
    
    # Initialize DEIO
    slam = DEIO2(cfg, network, evs=False, ht=H, wd=W, viz=viz, **kwargs)
    
    # IMU setup (similar to ZED evaluation script)
    imu_enabled = getattr(cfg, "ENALBE_IMU", False) and (all_imu is not None)
    
    if imu_enabled:
        print("IMU-enabled mode: Setting up IMU parameters")
        
        Ti1c = np.array(cfg.Ti1c).reshape(4, 4)
        if getattr(cfg, "ENALBE_INV", False):
            Ti1c = np.linalg.inv(Ti1c)
        
        IMU_noise = np.array([
            cfg.accel_noise_sigma, cfg.gyro_noise_sigma,
            cfg.accel_bias_sigma, cfg.gyro_bias_sigma
        ])
        
        slam.Ti1c = Ti1c
        slam.Tbc = gtsam.Pose3(slam.Ti1c)
        slam.state.set_imu_params((IMU_noise * 1.0).tolist())
        
        # Ensure strictly non-decreasing timestamps
        time_diffs = np.diff(all_imu[:, 0])
        if (time_diffs < 0).any():
            mask = np.concatenate([[True], time_diffs >= 0])
            all_imu = all_imu[mask]
        
        slam.all_imu = all_imu
        slam.all_gt = all_gt
        slam.all_gt_keys = all_gt_keys
        
        # Configure for VI mode
        slam.visual_only = False
        slam.ignore_imu = False
        slam.imu_enabled = False  # Will be enabled after initialization
        
    else:
        print("Running in visual-only mode")
        slam.visual_only = True
        slam.imu_enabled = False
        slam.ignore_imu = True
        slam.all_imu = None
        slam.all_gt = all_gt
        slam.all_gt_keys = all_gt_keys
        slam.Ti1c = np.eye(4)

    # Depth configuration
    depth_success_count = 0
    frame_count = 0
    depth_stats = {
        'valid_frames': 0,
        'depths': [],
        'errors': []
    }

    # Prepare sorted depth files list if depth directory provided
    depth_files_list = None
    depth_format = 'auto'
    
    if depth_dir:
        # Check for /data subdirectory first (ZED structure)
        if (depth_dir / "data").exists():
            depth_search_dir = depth_dir / "data"
            print(f"Found /data subdirectory, searching there...")
        else:
            depth_search_dir = depth_dir
        
        # Try to find depth files in order of preference:
        # 1. PSMNet .npy files (generated stereo predictions)
        npy_files = sorted(list(depth_search_dir.glob("*.npy")))
        
        # 2. ZED native .png files
        png_files = sorted(list(depth_search_dir.glob("*.png")))
        
        if len(npy_files) > 0:
            depth_files_list = npy_files
            depth_format = 'npy'
            print(f"✅ Found {len(npy_files)} PSMNet depth files (.npy) in {depth_search_dir}")
            
        elif len(png_files) > 0:
            depth_files_list = png_files
            depth_format = 'png'
            print(f"✅ Found {len(png_files)} ZED native depth files (.png) in {depth_search_dir}")
            
        else:
            print(f"⚠️ No depth files (.npy or .png) found in {depth_search_dir}")
            depth_dir = None
            depth_files_list = None
    
    # Processing frames
    for i, (image, intrinsics, t) in enumerate(tqdm(iterator, desc="Processing frames")):
        # Load depth map for current frame
        psmnet_depth = None
        if depth_dir and depth_files_list:
            psmnet_depth = load_zed_depth_for_frame(depth_search_dir, frame_count, H, W, depth_files_list)
            
            if psmnet_depth is not None:
                depth_success_count += 1
                
                # Compute depth statistics
                valid_depth = psmnet_depth[psmnet_depth > 0.5]
                if valid_depth.size > 0:
                    depth_stats['valid_frames'] += 1
                    depth_stats['depths'].append({
                        'min': valid_depth.min(),
                        'max': valid_depth.max(),
                        'median': np.median(valid_depth),
                        'mean': np.mean(valid_depth),
                        'std': np.std(valid_depth)
                    })

        # Call DEIO with depth
        with Timer("DEIO_ZED", enabled=timing):
            slam(t, image, intrinsics, scale=1.0, psmnet_depth=psmnet_depth)

        frame_count += 1

    # Final optimization
    for _ in range(12):
        slam.update()

    poses, tstamps = slam.terminate()

    # Depth evaluation report
    if depth_dir:
        success_rate = 100.0 * depth_success_count / max(frame_count, 1)
        print("\n=== Depth Integration Summary ===")
        print(f"Depth format: {depth_format}")
        print(f"Processed frames: {frame_count}")
        print(f"Successful depth loads: {depth_success_count}")
        print(f"Depth success rate: {success_rate:.1f}%")
        
        # Aggregate depth statistics
        if depth_stats['depths']:
            depth_agg = {
                'min': min(d['min'] for d in depth_stats['depths']),
                'max': max(d['max'] for d in depth_stats['depths']),
                'median_of_medians': np.median([d['median'] for d in depth_stats['depths']]),
                'mean_of_means': np.mean([d['mean'] for d in depth_stats['depths']]),
                'mean_of_stds': np.mean([d['std'] for d in depth_stats['depths']])
            }
            print("\nDepth Statistics:")
            for k, v in depth_agg.items():
                print(f"  {k}: {v:.2f}m")

    return poses, tstamps


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate DEIO Depth on ZED dataset')
    parser.add_argument('--processed_dir', type=Path, required=True,
                       help='Path to processed ZED dataset directory')
    parser.add_argument('--network', type=str, default='dpvo.pth',
                       help='Path to DPVO network weights')
    parser.add_argument('--config', default="config/zed_deio_vi.yaml",
                       help='Path to ZED DEIO config file')
    parser.add_argument('--depth_dir', type=Path, default=None,
                       help='Directory containing depth maps (.npy or .png)')
    parser.add_argument('--stride', type=int, default=1,
                       help='Frame stride (use every Nth frame)')
    parser.add_argument('--viz', action="store_true",
                       help='Enable visualization')
    parser.add_argument('--trials', type=int, default=1,
                       help='Number of trials to run')
    parser.add_argument('--backend_thresh', type=float, default=32.0)
    parser.add_argument('--plot', action="store_true",
                       help='Generate trajectory plots')
    parser.add_argument('--save_trajectory', action="store_true",
                       help='Save estimated trajectories')
    parser.add_argument('--opts', nargs='+', default=[],
                       help='Override config options (e.g., ENALBE_IMU False)')
    
    args = parser.parse_args()
    
    # Load config
    cfg.merge_from_file(args.config)
    cfg.BACKEND_THRESH = args.backend_thresh
    cfg.merge_from_list(args.opts)
    cfg.resnet = False
    
    print("=" * 60)
    print("DEIO Depth Evaluation on ZED Dataset")
    print("=" * 60)
    print(f"Dataset: {args.processed_dir}")
    print(f"Config: {args.config}")
    print(f"Depth Directory: {args.depth_dir}")
    print(f"Stride: {args.stride}")
    print(f"IMU Enabled: {getattr(cfg, 'ENALBE_IMU', False)}")
    print(f"Depth Priors: {getattr(cfg, 'ENABLE_DEPTH_PRIORS', False)}")
    print(f"Depth Init: {getattr(cfg, 'ENABLE_DEPTH_INIT', False)}")
    print("=" * 60)
    print(cfg, "\n")
    
    # Set random seed
    torch.manual_seed(1234)
    
    results = []
    
    for trial_num in range(args.trials):
        print(f"\n{'=' * 60}")
        print(f"TRIAL {trial_num + 1}/{args.trials}")
        print(f"{'=' * 60}\n")
        
        try:
            # Run DEIO Depth Evaluation
            est_poses, est_timestamps = run_DEIO_ZED_Depth(
                args.processed_dir, cfg, args.network,
                depth_dir=args.depth_dir, 
                viz=args.viz, 
                stride=args.stride, 
                timing=True
            )
            
            print(f"\n✅ DEIO estimated {len(est_poses)} poses")
            
            # Save trajectory
            if args.save_trajectory:
                Path("saved_trajectories").mkdir(exist_ok=True)
                
                # Create descriptive filename
                seq_name = args.processed_dir.name
                imu_suffix = "imu" if getattr(cfg, 'ENALBE_IMU', False) else "visual"
                depth_suffix = "depth" if args.depth_dir else "nodepth"
                output_file = f"saved_trajectories/{seq_name}_{imu_suffix}_{depth_suffix}_trial{trial_num+1:02d}.txt"
                
                traj_est = PoseTrajectory3D(
                    positions_xyz=est_poses[:,:3],
                    orientations_quat_wxyz=est_poses[:, [6, 3, 4, 5]],
                    timestamps=est_timestamps  # FIXED: Don't multiply by stride! Timestamps are already correct from DPVO
                )
                
                file_interface.write_tum_trajectory_file(output_file, traj_est)
                print(f"💾 Saved trajectory: {output_file}")
            
            results.append({
                'trial': trial_num + 1,
                'num_poses': len(est_poses),
                'success': True
            })
            
        except Exception as e:
            print(f"\n❌ Error in trial {trial_num + 1}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'trial': trial_num + 1,
                'success': False,
                'error': str(e)
            })
            continue
    
    # Summary
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    successful_trials = sum(1 for r in results if r.get('success', False))
    print(f"Successful trials: {successful_trials}/{args.trials}")
    
    if args.save_trajectory and successful_trials > 0:
        print(f"\n📁 Trajectories saved in: saved_trajectories/")
        print("\nTo evaluate against ground truth, run:")
        print(f"  evo_ape tum groundtruth.txt saved_trajectories/{args.processed_dir.name}_*.txt -va --plot")

if __name__ == '__main__':
    main()
