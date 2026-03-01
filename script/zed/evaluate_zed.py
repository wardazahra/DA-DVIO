from itertools import count
from multiprocessing import Process, Queue
from pathlib import Path
import os
import time

import cv2
import evo.main_ape as main_ape
import numpy as np
import torch
from evo.core import sync
from evo.core.metrics import PoseRelation
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface, plot

from dpvo.config import cfg
from dpvo.dpvo import DPVO
from dpvo.plot_utils import plot_trajectory
from dpvo.utils import Timer

def show_image(image, t=0):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(t)

def read_calib_file(filepath):
    """Read in a calibration file and parse into a dictionary."""
    data = {}
    with open(filepath, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass
    return data

def zed_image_stream(queue, dataset_dir, stride, skip=0):
    """
    Image generator for ZED dataset
    Expects structure:
    dataset_dir/
    ├── sequences/00/
    │   ├── image_2/  (left camera images)
    │   └── calib.txt
    ├── poses/
    │   └── 00.txt
    └── times.txt or image_times.txt
    """
    
    sequence_dir = dataset_dir / "sequences" / "00"
    images_dir = sequence_dir / "image_2"
    
    # Get list of images
    image_files = sorted(list(images_dir.glob("*.png")))[skip::stride]
    
    print(f"Looking for images in: {images_dir}")
    print(f"Found {len(image_files)} images with stride={stride}, skip={skip}")
    
    if len(image_files) == 0:
        print(f"ERROR: No images found in {images_dir}")
        print("Expected directory structure:")
        print("  dataset_dir/sequences/00/image_2/*.png")
        # Send dummy data to prevent hanging
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        dummy_intrinsics = np.array([500.0, 500.0, 320.0, 240.0])
        queue.put((-1, dummy_image, dummy_intrinsics))
        return
    
    print(f"First few images: {[img.name for img in image_files[:3]]}")
    
    # Load calibration
    calib_file = sequence_dir / "calib.txt"
    if calib_file.exists():
        calib = read_calib_file(calib_file)
        # Extract intrinsics from P0 matrix [fx, fy, cx, cy]
        intrinsics = calib['P0'][[0, 5, 2, 6]]
        print(f"Loaded calibration from: {calib_file}")
        print(f"Intrinsics: fx={intrinsics[0]:.1f}, fy={intrinsics[1]:.1f}, cx={intrinsics[2]:.1f}, cy={intrinsics[3]:.1f}")
    else:
        print(f"WARNING: No calibration file found at {calib_file}")
        print("Using default intrinsics - results may be poor!")
        intrinsics = np.array([500.0, 500.0, 320.0, 240.0])
    
    # Process images
    for t, img_file in enumerate(image_files):
        # Load image
        image = cv2.imread(str(img_file))
        if image is None:
            print(f"ERROR: Could not load image {img_file}")
            continue
            
        # Ensure dimensions are divisible by 4 (DPVO requirement)
        H, W = image.shape[:2]
        H, W = (H - H%4, W - W%4)
        image = image[:H, :W]
        
        # Send to DPVO
        queue.put((t, image, intrinsics))
    
    # Send termination signal
    queue.put((-1, image if len(image_files) > 0 else dummy_image, intrinsics))

@torch.no_grad()
def run_dpvo_on_zed(cfg, network, dataset_dir, stride=1, viz=False, show_img=False):
    """Run DPVO on ZED dataset"""
    
    slam = None
    
    queue = Queue(maxsize=8)
    reader = Process(target=zed_image_stream, args=(queue, dataset_dir, stride, 0))
    reader.start()
    
    print(f"Starting DPVO on ZED dataset: {dataset_dir}")
    
    with Timer("DPVO_TOTAL"):
        for step in count(start=1):
            (t, image, intrinsics) = queue.get()
            if t < 0: 
                break
            
            # Convert to torch tensors
            image = torch.as_tensor(image, device='cuda').permute(2,0,1)
            intrinsics = torch.as_tensor(intrinsics, dtype=torch.float, device='cuda')
            
            if show_img:
                show_image(image, 1)
            
            # Initialize DPVO on first frame
            if slam is None:
                slam = DPVO(cfg, network, ht=image.shape[-2], wd=image.shape[-1], viz=viz)
                print(f"DPVO initialized with image size: {image.shape[-1]}x{image.shape[-2]}")
            
            # Process frame
            with Timer("SLAM", enabled=False):
                slam(t, image, intrinsics)
    
    reader.join()
    
    if slam is None:
        print("ERROR: SLAM initialization failed - no images processed")
        return None, None
    
    return slam.terminate()

def load_zed_ground_truth(dataset_dir):
    """Load ground truth poses from ZED dataset"""
    
    poses_file = dataset_dir / "poses" / "00.txt"
    
    if not poses_file.exists():
        print(f"WARNING: No ground truth poses found at {poses_file}")
        return None
    
    # Read poses in KITTI format (12 values per line)
    poses_data = []
    with open(poses_file, 'r') as f:
        for line in f:
            pose_vals = [float(x) for x in line.strip().split()]
            if len(pose_vals) == 12:
                poses_data.append(pose_vals)
    
    if len(poses_data) == 0:
        print("ERROR: No valid poses found in ground truth file")
        return None
    
    # Convert to trajectory format
    poses_array = np.array(poses_data)
    
    # Extract positions (columns 3, 7, 11 from 3x4 matrices)
    positions = poses_array[:, [3, 7, 11]]  
    
    # Extract rotations and convert to quaternions
    orientations = []
    for pose_vals in poses_data:
        # Reshape to 3x4 matrix
        T = np.array(pose_vals).reshape(3, 4)
        R = T[:, :3]  # Extract rotation matrix
        
        # Convert rotation matrix to quaternion [w, x, y, z]
        quat = rotation_matrix_to_quaternion(R)
        orientations.append(quat)
    
    orientations = np.array(orientations)
    
    # Create trajectory
    timestamps = np.arange(len(poses_data), dtype=np.float64)
    
    traj_gt = PoseTrajectory3D(
        positions_xyz=positions,
        orientations_quat_wxyz=orientations,
        timestamps=timestamps
    )
    
    print(f"Loaded ground truth: {len(poses_data)} poses")
    
    # Calculate trajectory length manually
    if len(positions) > 1:
        diffs = np.diff(positions, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        total_length = np.sum(distances)
        print(f"Trajectory length: {total_length:.2f} m")
    
    return traj_gt

def rotation_matrix_to_quaternion(R):
    """Convert 3x3 rotation matrix to quaternion [w, x, y, z]"""
    trace = np.trace(R)
    
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2  # s = 4 * qw
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # s = 4 * qx
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # s = 4 * qy
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # s = 4 * qz
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s
    
    return np.array([qw, qx, qy, qz])

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate DPVO on ZED dataset')
    parser.add_argument('--network', type=str, default='dpvo.pth', help='Path to DPVO network weights')
    parser.add_argument('--config', default="config/default.yaml", help='Path to config file')
    parser.add_argument('--stride', type=int, default=1, help='Frame stride (use every Nth frame)')
    parser.add_argument('--viz', action="store_true", help='Enable visualization')
    parser.add_argument('--show_img', action="store_true", help='Show input images')
    parser.add_argument('--trials', type=int, default=1, help='Number of trials to run')
    parser.add_argument('--dataset_dir', type=Path, required=True, 
                       help='Path to ZED dataset directory')
    parser.add_argument('--backend_thresh', type=float, default=32.0)
    parser.add_argument('--plot', action="store_true", help='Generate trajectory plots')
    parser.add_argument('--save_trajectory', action="store_true", help='Save estimated trajectory')
    parser.add_argument('--opts', nargs='+', default=[], help='Override config options')
    
    args = parser.parse_args()
    
    # Load config
    cfg.merge_from_file(args.config)
    cfg.BACKEND_THRESH = args.backend_thresh
    cfg.merge_from_list(args.opts)
    
    print("Running DPVO on ZED dataset...")
    print(f"Dataset: {args.dataset_dir}")
    print(f"Stride: {args.stride}")
    print(f"Config: {cfg}\n")
    
    # Set random seed
    torch.manual_seed(1234)
    
    # Load ground truth
    traj_gt = load_zed_ground_truth(args.dataset_dir)
    
    if traj_gt is None:
        print("WARNING: No ground truth available - will only run DPVO without evaluation")
    
    # Run trials
    results = []
    
    for trial_num in range(args.trials):
        print(f"\n=== TRIAL {trial_num + 1}/{args.trials} ===")
        
        # Run DPVO
        traj_est, timestamps = run_dpvo_on_zed(
            cfg, args.network, args.dataset_dir, 
            args.stride, args.viz, args.show_img
        )
        
        if traj_est is None:
            print(f"Trial {trial_num + 1} failed - skipping")
            continue
        
        print(f"DPVO estimated {len(traj_est)} poses")
        
        # Create estimated trajectory
        traj_est_obj = PoseTrajectory3D(
            positions_xyz=traj_est[:,:3],
            orientations_quat_wxyz=traj_est[:, [6, 3, 4, 5]],  # [w,x,y,z] order
            timestamps=timestamps * args.stride
        )
        
        # Save trajectory
        if args.save_trajectory:
            Path("saved_trajectories").mkdir(exist_ok=True)
            output_file = f"saved_trajectories/ZED_trial_{trial_num+1:02d}.txt"
            file_interface.write_tum_trajectory_file(output_file, traj_est_obj)
            print(f"Saved estimated trajectory: {output_file}")
        
        # Evaluate against ground truth
        if traj_gt is not None:
            try:
                # Align trajectories temporally
                traj_gt_aligned, traj_est_aligned = sync.associate_trajectories(traj_gt, traj_est_obj)
                
                print(f"Aligned trajectories: GT={traj_gt_aligned.num_poses}, EST={traj_est_aligned.num_poses}")
                
                # Compute ATE (Absolute Trajectory Error)
                result = main_ape.ape(
                    traj_gt_aligned, traj_est_aligned, 
                    est_name='DPVO', 
                    pose_relation=PoseRelation.translation_part, 
                    align=True, 
                    correct_scale=False
                )
                
                ate_score = result.stats["rmse"]
                results.append(ate_score)
                
                print(f"ATE RMSE: {ate_score:.4f} m")
                
                # Generate plots
                if args.plot:
                    Path("trajectory_plots").mkdir(exist_ok=True)
                    plot_file = f"trajectory_plots/ZED_trial_{trial_num+1:02d}.pdf"
                    plot_trajectory(
                        traj_est_aligned, traj_gt_aligned, 
                        f"ZED Dataset Trial #{trial_num+1}", 
                        plot_file, 
                        align=True, correct_scale=False
                    )
                    print(f"Saved plot: {plot_file}")
                
            except Exception as e:
                print(f"Evaluation failed: {e}")
                print("Continuing without evaluation...")
        
        else:
            print("Skipping evaluation (no ground truth)")
    
    # Summary
    if results:
        print(f"\n=== EVALUATION SUMMARY ===")
        print(f"Trials: {len(results)}")
        print(f"ATE RMSE: {results}")
        print(f"Mean ATE: {np.mean(results):.4f} m")
        print(f"Std ATE:  {np.std(results):.4f} m")
    else:
        print("\n=== DPVO COMPLETED ===")
        print("No evaluation metrics available")
    
    print("Done!")