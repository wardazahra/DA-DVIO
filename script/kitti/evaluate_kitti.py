from itertools import count
from multiprocessing import Process, Queue
from pathlib import Path
import os
import time  # Add at the top with other imports

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

SKIP = 0

def show_image(image, t=0):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(t)

# From https://github.com/utiasSTARS/pykitti/blob/d3e1bb81676e831886726cc5ed79ce1f049aef2c/pykitti/utils.py#L68
def read_calib_file(filepath):
    """Read in a calibration file and parse into a dictionary."""
    data = {}

    with open(filepath, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data

def load_kitti_official_depth(kitti_dir, sequence, frame_id, target_height, target_width):
    """
    Load official KITTI depth maps (LiDAR-based or depth completion dataset)
    
    Priority order:
    1. Dense depth completion maps: train/SEQUENCE/proj_depth/groundtruth/image_02/FRAME.png
    2. Sparse LiDAR depths: train/SEQUENCE/proj_depth/velodyne_raw/image_02/FRAME.png  
    3. Raw LiDAR data: sequences/SEQUENCE/velodyne/FRAME.bin
    """
    
    # Option 1: Dense depth completion groundtruth
    dense_depth_file = kitti_dir / "train" / sequence / "proj_depth" / "groundtruth" / "image_02" / f"{frame_id:010d}.png"
    if dense_depth_file.exists():
        depth_png = cv2.imread(str(dense_depth_file), cv2.IMREAD_UNCHANGED)
        if depth_png is not None:
            # KITTI depth completion format: depth = pixel_value / 256.0
            depth = depth_png.astype(np.float32) / 256.0
            if depth.shape != (target_height, target_width):
                depth = cv2.resize(depth, (target_width, target_height))
            return depth, "dense_gt"
    
    # Option 2: Sparse LiDAR projections
    sparse_depth_file = kitti_dir / "train" / sequence / "proj_depth" / "velodyne_raw" / "image_02" / f"{frame_id:010d}.png"
    if sparse_depth_file.exists():
        depth_png = cv2.imread(str(sparse_depth_file), cv2.IMREAD_UNCHANGED)
        if depth_png is not None:
            depth = depth_png.astype(np.float32) / 256.0
            if depth.shape != (target_height, target_width):
                depth = cv2.resize(depth, (target_width, target_height))
            return depth, "sparse_lidar"
    
    # Option 3: Process raw LiDAR data (more complex)
    velodyne_file = kitti_dir / "sequences" / sequence / "velodyne" / f"{frame_id:06d}.bin"
    if velodyne_file.exists():
        # This requires LiDAR-to-image projection (complex)
        print(f"⚠️  Found raw LiDAR for frame {frame_id}, but projection not implemented")
        return None, "raw_lidar_found"
    
    return None, "not_found"


def load_psmnet_depth_for_frame(psmnet_dir, frame_id, target_height, target_width, calib_data=None):
    """
    Load PSMNet depth map for specific frame
    Handles naming convention: XXXXXX_depth.npy
    Converts from disparity to metric depth using proper calibration
    """
    if psmnet_dir is None:
        return None
        
    # PSMNet naming convention: 000000_depth.npy, 000001_depth.npy, etc.
    npy_file = psmnet_dir / f"{frame_id:06d}_depth.npy"
    
    if npy_file.exists():
        depth = np.load(npy_file).astype(np.float32)
        
        # Check if this looks like disparity (very large values)
        if np.median(depth[depth > 0]) > 1000:
            # Use calibration data if available
            if calib_data is not None and 'P0' in calib_data:
                # Extract focal length from calibration
                fx = calib_data['P0'][0, 0]  # Focal length in x
                # Use baseline from P1 (right camera)
                if 'P1' in calib_data:
                    # Baseline = |T_x| / fx where T_x is P1[0,3]
                    baseline = abs(calib_data['P1'][0, 3]) / fx
                    baseline_focal = baseline * fx
                    print(f"🎯 Using calibration: fx={fx:.1f}, baseline={baseline:.3f}m, bf={baseline_focal:.1f}")
                else:
                    # Use sequence 05 specific parameters  
                    baseline_focal = 379.8145  # Exact from sequence 05 calib
                    print(f"🎯 Using sequence 05 exact calibration: fx={fx:.1f}, bf={baseline_focal:.1f}")
            else:
                # For sequence 05 specifically - use exact calibration
                baseline_focal = 379.8145  # Exact from sequence 05 calib.txt
                print(f"🎯 Using sequence 05 exact calibration: bf={baseline_focal:.1f}")
            
            original_median = np.median(depth[depth > 0])
            
            # Avoid division by zero
            depth = np.where(depth > 1.0, baseline_focal / depth, 0.0)
            
            # Clamp to reasonable KITTI depth range
            depth = np.clip(depth, 1.0, 80.0)


            
            converted_median = np.median(depth[depth > 1.0])
            
            if frame_id % 100 == 0:  # Print every 100th frame
                print(f"🔧 Frame {frame_id}: Disparity→Depth conversion")
                print(f"   Original disparity median: {original_median:.1f} pixels")
                print(f"   Converted depth median: {converted_median:.2f} meters")
        
        # Resize to match DPVO input resolution if needed
        if depth.shape != (target_height, target_width):
            depth = cv2.resize(depth, (target_width, target_height))
        
        return depth
    else:
        return None

def kitti_image_stream(queue, kittidir, sequence, stride, skip=0, psmnet_base_dir=None):
    """ image generator - MODIFIED to include PSMNet depth loading """
    images_dir = kittidir / "sequences" / sequence
    image_list = sorted((images_dir / "image_2").glob("*.png"))[skip::stride]
    
    print(f"🔍 Debug: Looking for images in: {images_dir / 'image_2'}")
    print(f"🔍 Debug: Found {len(image_list)} images with stride={stride}, skip={skip}")
    if len(image_list) > 0:
        print(f"🔍 Debug: First few images: {[img.name for img in image_list[:3]]}")
    else:
        print(f"❌ No images found in {images_dir / 'image_2'}")
        # Send termination signal immediately
        dummy_image = np.zeros((376, 1241, 3), dtype=np.uint8)
        dummy_intrinsics = np.array([718.856, 718.856, 607.1928, 185.2157])
        queue.put((-1, dummy_image, dummy_intrinsics, None))
        return

    # Try sequence-specific calib first, then fall back to central calib
    calib_file = images_dir / "calib.txt"
    if not calib_file.exists():
        calib_file = kittidir / "calib" / "kitti.txt"
    
    if calib_file.exists():
        calib = read_calib_file(calib_file)
        intrinsics = calib['P0'][[0, 5, 2, 6]]
        print(f"✅ Loaded calibration from: {calib_file}")
    else:
        print(f"⚠️  No calibration file found. Checked:")
        print(f"    - {images_dir / 'calib.txt'}")
        print(f"    - {kittidir / 'calib' / 'kitti.txt'}")
        print("Using default KITTI calibration parameters")
        # Default KITTI calibration for sequences 00-10
        intrinsics = np.array([718.856, 718.856, 607.1928, 185.2157])  # fx, fy, cx, cy
    
    # Determine PSMNet directory for this sequence
    psmnet_dir = None
    if psmnet_base_dir is not None:
        # Try multiple possible structures:
        # 1. sequence/depth_maps/ subdirectory
        psmnet_dir = psmnet_base_dir / sequence / "depth_maps"
        if not psmnet_dir.exists():
            # 2. sequence/ subdirectory  
            psmnet_dir = psmnet_base_dir / sequence
        if not psmnet_dir.exists():
            # 3. All depths in base directory (your case)
            psmnet_dir = psmnet_base_dir
            
        if psmnet_dir.exists():
            print(f"✅ Using PSMNet depths from: {psmnet_dir}")
        else:
            print(f"⚠️  PSMNet directory not found. Tried:")
            print(f"    - {psmnet_base_dir / sequence / 'depth_maps'}")
            print(f"    - {psmnet_base_dir / sequence}")
            print(f"    - {psmnet_base_dir}")
            psmnet_dir = None

    for t, imfile in enumerate(image_list):
        image_left = cv2.imread(str(imfile))
        H, W, _ = image_left.shape
        H, W = (H - H%4, W - W%4)
        image_left = image_left[:H, :W, :]

        # Calculate frame_id for PSMNet alignment
        frame_id = t * stride + skip
        
        # Load corresponding PSMNet depth
        psmnet_depth = load_psmnet_depth_for_frame(psmnet_dir, frame_id, H, W) if psmnet_dir else None

        queue.put((t, image_left, intrinsics, psmnet_depth))

    # Send termination signal - use dummy image if no images were processed
    if len(image_list) > 0:
        queue.put((-1, image_left, intrinsics, None))
    else:
        dummy_image = np.zeros((376, 1241, 3), dtype=np.uint8)  # KITTI image size
        queue.put((-1, dummy_image, intrinsics, None))


@torch.no_grad()
def run(cfg, network, kittidir, sequence, stride=1, viz=False, show_img=False, psmnet_base_dir=None):
    slam = None

    queue = Queue(maxsize=8)
    reader = Process(target=kitti_image_stream, args=(queue, kittidir, sequence, stride, 0, psmnet_base_dir))
    reader.start()

    with Timer("DPVO_TOTAL"):
        for step in count(start=1):
            (t, image, intrinsics, psmnet_depth) = queue.get()
            if t < 0: break

            image = torch.as_tensor(image, device='cuda').permute(2,0,1)
            intrinsics = torch.as_tensor(intrinsics, dtype=torch.float, device='cuda')

            if show_img:
                show_image(image, 1)

            if slam is None:
                slam = DPVO(cfg, network, ht=image.shape[-2], wd=image.shape[-1], viz=viz)

            intrinsics = intrinsics.cuda()

            with Timer("SLAM", enabled=False):
                slam(t, image, intrinsics, psmnet_depth=psmnet_depth)

    reader.join()

    if slam is None:
        print("❌ SLAM initialization failed - no images processed")
        return None, None

    return slam.terminate()



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, default='dpvo.pth')
    parser.add_argument('--config', default="config/default.yaml")
    parser.add_argument('--stride', type=int, default=2)
    parser.add_argument('--viz', action="store_true")
    parser.add_argument('--show_img', action="store_true")
    parser.add_argument('--trials', type=int, default=1)
    parser.add_argument('--kittidir', type=Path, default="datasets/KITTI")
    parser.add_argument('--psmnet_dir', type=Path, default=None,
                       help='Base directory containing PSMNet depths (e.g., /path/to/PSMNet/output)')
    parser.add_argument('--backend_thresh', type=float, default=32.0)
    parser.add_argument('--plot', action="store_true")
    parser.add_argument('--opts', nargs='+', default=[])
    parser.add_argument('--save_trajectory', action="store_true")
    parser.add_argument('--sequences', nargs='+', default=None,
                       help='Specific sequences to evaluate (e.g., --sequences 00 05 10)')
    args = parser.parse_args()

    cfg.merge_from_file(args.config)
    cfg.BACKEND_THRESH = args.backend_thresh
    cfg.merge_from_list(args.opts)

    print("\nRunning with config...")
    print(cfg, "\n")
    
    if args.psmnet_dir:
        print(f"🎯 Using PSMNet depths from: {args.psmnet_dir}")
    else:
        print("⚠️  No PSMNet depths provided - using original DPVO initialization")

    torch.manual_seed(1234)

    # Determine sequences to evaluate
    if args.sequences:
        kitti_scenes = [f"{int(s):02d}" for s in args.sequences]
    else:
        kitti_scenes = [f"{i:02d}" for i in range(11)]
    
    print(f"Evaluating sequences: {kitti_scenes}")

    results = {}
    for scene in kitti_scenes:
        groundtruth = args.kittidir / "poses" / f"{scene}.txt"
        poses_ref = file_interface.read_kitti_poses_file(groundtruth)
        print(f"Evaluating KITTI {scene} with {poses_ref.num_poses // args.stride} poses")

        scene_results = []
        for trial_num in range(args.trials):
            traj_est, timestamps = run(cfg, args.network, args.kittidir, scene, args.stride, args.viz, args.show_img, args.psmnet_dir)

            if traj_est is None:
                print(f"❌ Trial {trial_num + 1} failed - skipping")
                continue

            traj_est = PoseTrajectory3D(
                positions_xyz=traj_est[:,:3],
                orientations_quat_wxyz=traj_est[:, [6, 3, 4, 5]],
                timestamps=timestamps * args.stride)

            import numpy as np
            traj_raw = traj_est.poses_se3
            np.save('trajectory_raw_depth_supervised.txt', traj_raw)
            
            traj_ref = PoseTrajectory3D(
                positions_xyz=poses_ref.positions_xyz,
                orientations_quat_wxyz=poses_ref.orientations_quat_wxyz,
                timestamps=np.arange(poses_ref.num_poses, dtype=np.float64))

            traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)

            result = main_ape.ape(traj_ref, traj_est, est_name='traj', 
                pose_relation=PoseRelation.translation_part, align=True, correct_scale=False)
            ate_score = result.stats["rmse"]

            if args.plot:
                plot_trajectory(traj_est, traj_ref, f"kitti sequence {scene} Trial #{trial_num+1}", f"trajectory_plots/kitti_seq{scene}_trial{trial_num+1:02d}.pdf", align=True, correct_scale=False)

            if args.save_trajectory:
                Path("saved_trajectories").mkdir(exist_ok=True)
                file_interface.write_tum_trajectory_file(f"saved_trajectories/KITTI_{scene}.txt", traj_est)
                # file_interface.write_kitti_poses_file(f"saved_trajectories/{scene}.txt", traj_est) # standard kitti format

            scene_results.append(ate_score)

        results[scene] = np.median(scene_results)
        print(scene, sorted(scene_results))

    xs = []
    for scene in results:
        print(scene, results[scene])
        xs.append(results[scene])

    print("AVG: ", np.mean(xs))
