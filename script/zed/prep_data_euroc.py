#!/usr/bin/env python3
"""
Convert EuRoC format ZED data to DEIO format
Handles the mav0 structure from ZED ROS recordings
"""

import os
import numpy as np
from pathlib import Path
import shutil
import argparse
import cv2

def load_euroc_imu(imu_csv_path):
    """Load IMU data from EuRoC format CSV"""
    print(f"Loading IMU data from: {imu_csv_path}")
    
    # Skip header line, load data
    imu_data = np.loadtxt(imu_csv_path, delimiter=',', skiprows=1)
    
    print(f"Loaded {len(imu_data)} IMU measurements")
    print(f"Time range: {imu_data[0, 0]/1e9:.3f} to {imu_data[-1, 0]/1e9:.3f} seconds")
    print(f"Sample: ts={imu_data[0,0]}, gyro=[{imu_data[0,1]:.3f}, {imu_data[0,2]:.3f}, {imu_data[0,3]:.3f}], accel=[{imu_data[0,4]:.3f}, {imu_data[0,5]:.3f}, {imu_data[0,6]:.3f}]")
    
    return imu_data

def load_euroc_images(cam_csv_path):
    """Load image timestamps and filenames from EuRoC format"""
    print(f"Loading image data from: {cam_csv_path}")
    
    # Read CSV with timestamp and filename
    data = []
    with open(cam_csv_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split(',')
            if len(parts) == 2:
                timestamp_ns = int(parts[0])
                filename = parts[1]
                data.append((timestamp_ns, filename))
    
    timestamps_ns = np.array([d[0] for d in data])
    filenames = [d[1] for d in data]
    
    print(f"Loaded {len(timestamps_ns)} image timestamps")
    print(f"Time range: {timestamps_ns[0]/1e9:.3f} to {timestamps_ns[-1]/1e9:.3f} seconds")
    
    return timestamps_ns, filenames

def write_deio_imu_csv(output_path, imu_data):
    """Write IMU data in DEIO format (already in correct format)"""
    print(f"Writing DEIO IMU CSV: {output_path}")
    
    with open(output_path, 'w') as f:
        # DEIO expects: timestamp[ns], gyro_x, gyro_y, gyro_z, accel_x, accel_y, accel_z
        f.write("#timestamp [ns],w_RS_S_x,w_RS_S_y,w_RS_S_z,a_RS_S_x,a_RS_S_y,a_RS_S_z\n")
        
        for row in imu_data:
            # EuRoC format is already: timestamp[ns], gyro_x, gyro_y, gyro_z, accel_x, accel_y, accel_z
            f.write(f"{int(row[0])},{row[1]:.10f},{row[2]:.10f},{row[3]:.10f},{row[4]:.10f},{row[5]:.10f},{row[6]:.10f}\n")
    
    print(f"Wrote {len(imu_data)} IMU measurements")

def write_deio_timestamps(output_path, timestamps_ns):
    """Write image timestamps in seconds for DEIO"""
    print(f"Writing image timestamps: {output_path}")
    
    timestamps_sec = timestamps_ns / 1e9
    with open(output_path, 'w') as f:
        for ts in timestamps_sec:
            f.write(f"{ts:.9f}\n")

def setup_images(src_dir, dst_dir, filenames, start_idx=0):
    """Setup images with sequential naming for DEIO"""
    print(f"Setting up images: {src_dir} -> {dst_dir}")
    
    dst_dir.mkdir(parents=True, exist_ok=True)
    
    if start_idx > 0:
        filenames = filenames[start_idx:]
    
    for i, fname in enumerate(filenames):
        src_path = src_dir / fname
        if not src_path.exists():
            print(f"Warning: Image not found: {src_path}")
            continue
        
        # Sequential naming: 000000.png, 000001.png, etc.
        dst_name = f"{i:012d}.png"
        dst_path = dst_dir / dst_name
        
        if dst_path.exists():
            dst_path.unlink()
        
        try:
            os.symlink(src_path.resolve(), dst_path)
        except OSError:
            shutil.copy2(src_path, dst_path)
    
    print(f"Processed {len(filenames)} images")
    return len(filenames)

def estimate_intrinsics(image_dir, filenames):
    """Estimate camera intrinsics from image dimensions"""
    # Load first image to get dimensions
    sample_img = cv2.imread(str(image_dir / filenames[0]))
    if sample_img is None:
        raise RuntimeError(f"Cannot read sample image: {image_dir / filenames[0]}")
    
    h, w = sample_img.shape[:2]
    
    # Common ZED intrinsics (approximate)
    # For ZED2: 1920x1080 has roughly fx=fy=1050, cx=960, cy=540
    # Scale based on actual resolution
    fx = w * 0.95  # Rough estimate
    fy = fx
    cx = w / 2.0
    cy = h / 2.0
    
    print(f"Estimated intrinsics for {w}x{h}: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
    print("WARNING: These are estimates. Use actual calibration if available!")
    
    return fx, fy, cx, cy

def write_calib_file(output_path, fx, fy, cx, cy):
    """Write calibration file"""
    print(f"Writing calibration: {output_path}")
    np.savetxt(output_path, np.array([fx, fy, cx, cy]), fmt="%.9f")

def main():
    parser = argparse.ArgumentParser(description="Convert EuRoC format ZED data to DEIO format")
    parser.add_argument("--euroc_dir", type=Path, required=True,
                       help="Path to EuRoC format directory (containing mav0)")
    parser.add_argument("--output_dir", type=Path, required=True,
                       help="Output directory for DEIO format")
    parser.add_argument("--camera", choices=['cam0', 'cam1'], default='cam0',
                       help="Which camera to use (cam0=left, cam1=right)")
    parser.add_argument("--start_frame", type=int, default=0,
                       help="Starting frame index")
    parser.add_argument("--intrinsics", nargs=4, type=float, default=None,
                       help="Camera intrinsics: fx fy cx cy (if not provided, will estimate)")
    parser.add_argument("--visual_only", action="store_true",
                       help="Generate config for visual-only mode (no IMU)")
    
    args = parser.parse_args()
    
    euroc_dir = args.euroc_dir
    output_dir = args.output_dir
    
    print(f"\n{'='*60}")
    print("EuRoC to DEIO Converter")
    print(f"{'='*60}")
    print(f"Input: {euroc_dir}")
    print(f"Output: {output_dir}")
    print(f"Camera: {args.camera}")
    
    # Paths
    mav0_dir = euroc_dir / "mav0"
    cam_dir = mav0_dir / args.camera
    imu_dir = mav0_dir / "imu0"
    
    cam_csv = cam_dir / "data.csv"
    cam_data_dir = cam_dir / "data"
    imu_csv = imu_dir / "data.csv"
    
    # Verify structure
    required_paths = [mav0_dir, cam_dir, imu_dir, cam_csv, cam_data_dir, imu_csv]
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Required path not found: {path}")
    
    print("All required files found")
    
    # Load data
    imu_data = load_euroc_imu(imu_csv)
    timestamps_ns, filenames = load_euroc_images(cam_csv)
    
    # Apply start frame offset
    if args.start_frame > 0:
        timestamps_ns = timestamps_ns[args.start_frame:]
        filenames = filenames[args.start_frame:]
        print(f"Applied start_frame offset: {args.start_frame}")
    
    # Verify time alignment
    img_start_sec = timestamps_ns[0] / 1e9
    img_end_sec = timestamps_ns[-1] / 1e9
    imu_start_sec = imu_data[0, 0] / 1e9
    imu_end_sec = imu_data[-1, 0] / 1e9
    
    print(f"\nTime alignment check:")
    print(f"  Images: {img_start_sec:.3f}s to {img_end_sec:.3f}s ({img_end_sec - img_start_sec:.1f}s)")
    print(f"  IMU:    {imu_start_sec:.3f}s to {imu_end_sec:.3f}s ({imu_end_sec - imu_start_sec:.1f}s)")
    
    if imu_start_sec > img_start_sec + 0.1:
        print(f"  WARNING: IMU starts {imu_start_sec - img_start_sec:.3f}s after first image")
    if imu_end_sec < img_end_sec - 0.1:
        print(f"  WARNING: IMU ends {img_end_sec - imu_end_sec:.3f}s before last image")
    
    # Create output structure
    output_dir.mkdir(parents=True, exist_ok=True)
    images_output_dir = output_dir / "images_undistorted_left"
    
    # Setup images
    num_images = setup_images(cam_data_dir, images_output_dir, filenames, 0)
    
    # Write timestamps
    write_deio_timestamps(output_dir / "tss_imgs_sec_left.txt", timestamps_ns)
    
    # Write IMU data
    write_deio_imu_csv(output_dir / "imu_data.csv", imu_data)
    
    # Handle calibration
    if args.intrinsics:
        fx, fy, cx, cy = args.intrinsics
        print(f"Using provided intrinsics: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
    else:
        fx, fy, cx, cy = estimate_intrinsics(cam_data_dir, filenames)
    
    write_calib_file(output_dir / "calib_undist_left.txt", fx, fy, cx, cy)
    
    # Create config suggestion
    config_suggestion = output_dir / "zed_config_suggestion.yaml"
    with open(config_suggestion, 'w') as f:
        f.write("# ZED2 EuRoC Dataset Configuration for DEIO\n")
        f.write("# Generated from EuRoC format conversion\n")
        f.write(f"# Images: {num_images}, Duration: {(timestamps_ns[-1] - timestamps_ns[0]) / 1e9:.1f}s\n\n")
        
        if args.visual_only:
            f.write("# ========== VISUAL-ONLY MODE ==========\n")
            f.write("# IMU disabled - will run pure visual SLAM\n")
            f.write("ENALBE_IMU: False\n\n")
        else:
            f.write("# ========== VISUAL-INERTIAL MODE ==========\n")
            f.write("ENALBE_IMU: True\n")
            f.write("ENALBE_INV: False\n\n")
            
            f.write("# Camera-IMU transformation from your ORB-SLAM3 calibration (cam0 to imu0)\n")
            f.write("# Tbc from zed2_stereo_inertial.yaml\n")
            f.write("Ti1c: [0.01474591, -0.00169408, 0.99988984, 0.02575494, ")
            f.write("-0.99988765, 0.00266864, 0.01475039, 0.01928873, ")
            f.write("-0.00269334, -0.999995, -0.00165454, 0.00278677, ")
            f.write("0.0, 0.0, 0.0, 1.0]\n\n")
            
            f.write("# IMU noise parameters from your ZED calibration\n")
            f.write("accel_noise_sigma: 0.01249652  # accelerometer_noise_density\n")
            f.write("gyro_noise_sigma: 0.001313112  # gyroscope_noise_density\n")
            f.write("accel_bias_sigma: 0.0003807570  # accelerometer_random_walk\n")
            f.write("gyro_bias_sigma: 0.000008600470  # gyroscope_random_walk\n\n")
        
        f.write("# DEIO parameters - tuned for low frame rate (~10fps)\n")
        f.write("PATCHES_PER_FRAME: 96\n")
        f.write("REMOVAL_WINDOW: 22\n")
        f.write("OPTIMIZATION_WINDOW: 10\n")
        f.write("PATCH_LIFETIME: 18  # Increased for low frame rate\n")
        f.write("KEYFRAME_THRESH: 12.0  # Lowered for low frame rate\n\n")
        f.write("MOTION_MODEL: 'DAMPED_LINEAR'\n")
        f.write("MOTION_DAMPING: 0.5\n")
        f.write("MIXED_PRECISION: True\n")
        f.write("PATCH_SELECTOR: 'scorer'\n")
        f.write("NORM: 'none'\n\n")
        f.write("SCORER_EVAL_MODE: 'multi'\n")
        f.write("SCORER_EVAL_USE_GRID: True\n")
        f.write("LOOP_CLOSURE: False\n")
        f.write("BUFFER_SIZE: 8192\n")
        f.write("KEYFRAME_INDEX: 4\n")  # Lowered for low fps
        f.write("CLASSIC_LOOP_CLOSURE: False\n")
    
    print(f"\n{'='*60}")
    print("CONVERSION COMPLETE")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"Images: {num_images}")
    print(f"IMU samples: {len(imu_data)}")
    print(f"Duration: {(timestamps_ns[-1] - timestamps_ns[0]) / 1e9:.1f} seconds")
    
    if args.visual_only:
        print(f"\n*** VISUAL-ONLY MODE CONFIG GENERATED ***")
        print(f"IMU data converted but disabled in config")
    else:
        print(f"\n*** VISUAL-INERTIAL MODE CONFIG GENERATED ***")
        print(f"Using real ZED2 calibration parameters")
        print(f"WARNING: Your data shows active motion from start")
        print(f"         VI initialization may fail without stationary period")
    
    print(f"\nNext steps:")
    print(f"1. Review config: {config_suggestion}")
    print(f"2. Run DEIO:")
    print(f"   python evaluate_zed.py --processed_dir {output_dir} \\")
    print(f"       --config {config_suggestion} --network dpvo.pth --stride 1")
    
    if not args.visual_only:
        print(f"\n3. If VI initialization fails, try visual-only:")
        print(f"   python {Path(__file__).name} --euroc_dir {euroc_dir} \\")
        print(f"       --output_dir {output_dir} --visual_only")

if __name__ == "__main__":
    main()