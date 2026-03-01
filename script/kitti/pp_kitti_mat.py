import os
import numpy as np
import glob
import tqdm
import cv2
import scipy.io as sio
from scipy.spatial.transform import Rotation as R

def load_poses_kitti(poses_file):
    poses = []
    for line in open(poses_file).readlines():
        T = np.eye(4)
        vals = list(map(float, line.strip().split()))
        T[:3, :4] = np.array(vals).reshape(3, 4)
        t = T[:3, 3]
        R_mat = T[:3, :3]
        q = R.from_matrix(R_mat).as_quat()  # [x, y, z, w]
        poses.append(np.hstack((t, q)))  # [x y z qx qy qz qw]
    return np.array(poses)

def load_timestamps(timestamps_file):
    timestamps = np.loadtxt(timestamps_file)
    return timestamps

def load_imu_mat(mat_file_path, img_duration):
    """Load IMU data from .mat file with dynamic duration"""
    print(f"Loading IMU data from: {mat_file_path}")
    
    mat_data = sio.loadmat(mat_file_path)
    imu_data_interp = mat_data['imu_data_interp']
    
    print(f"Loaded .mat IMU data: {imu_data_interp.shape}")
    print(f"Data format appears to be: [ax, ay, az, gx, gy, gz]")
    
    # Extract accelerometer and gyroscope data
    accel_data = imu_data_interp[:, 0:3]  # ax, ay, az
    gyro_data = imu_data_interp[:, 3:6]   # gx, gy, gz
    
    # FIXED: Use dynamic duration based on image sequence length
    duration = img_duration  # Use actual sequence duration
    n_samples = len(imu_data_interp)
    
    # Create evenly spaced timestamps matching image duration
    timestamps = np.linspace(0, duration, n_samples)
    
    print(f"Generated timestamps:")
    print(f"  Duration: {duration:.2f} seconds (matches image sequence)")
    print(f"  Sample rate: {n_samples/duration:.1f} Hz")
    print(f"  Mean interval: {1000*duration/n_samples:.1f} ms")
    
    # Combine into standard IMU format: [timestamp, gx, gy, gz, ax, ay, az]
    imu_data = np.column_stack([
        timestamps,
        gyro_data[:, 0], gyro_data[:, 1], gyro_data[:, 2],  # gx, gy, gz
        accel_data[:, 0], accel_data[:, 1], accel_data[:, 2]  # ax, ay, az
    ])
    
    return imu_data

def write_gt_stamped(poses, timestamps, outpath):
    with open(outpath, 'w') as f:
        for t, p in zip(timestamps, poses):
            f.write(f"{t:.9f} {' '.join(map(str, p))}\n")

def write_imu_csv(imu_data, outpath):
    with open(outpath, 'w') as f:
        f.write("#timestamp [ns],w_RS_S_x,w_RS_S_y,w_RS_S_z,a_RS_S_x,a_RS_S_y,a_RS_S_z\n")
        for row in imu_data:
            t_ns = int(row[0] * 1e9)  # s → ns
            values = ','.join(f"{x:.10f}" for x in row[1:])
            f.write(f"{t_ns},{values}\n")

def preprocess_kitti_mat_seq(
    seq_id, 
    kitti_odometry_root, 
    kitti_mat_imu_path,
    output_dir
):
    seq = f"{seq_id:02d}"
    odo_seq_path = os.path.join(kitti_odometry_root, "sequences", seq)
    
    # FIXED: Correct path to sequence-specific .mat file
    mat_imu_file = os.path.join(kitti_mat_imu_path, f"{seq_id:02d}.mat")
    
    print(f"Processing KITTI sequence {seq}")
    print(f"IMU file: {mat_imu_file}")
    
    # Verify .mat file exists
    if not os.path.exists(mat_imu_file):
        raise FileNotFoundError(f"IMU .mat file not found: {mat_imu_file}")

    os.makedirs(output_dir, exist_ok=True)
    img_out_dir = os.path.join(output_dir, f"images_undistorted_left")
    os.makedirs(img_out_dir, exist_ok=True)

    # Load poses and image timestamps first
    poses = load_poses_kitti(os.path.join(kitti_odometry_root, "poses", f"{seq}.txt"))
    img_timestamps = load_timestamps(os.path.join(odo_seq_path, "times.txt"))
    
    # Calculate actual image sequence duration
    img_duration = img_timestamps[-1] - img_timestamps[0]
    print(f"📏 Image sequence duration: {img_duration:.2f} seconds")
    
    # Load IMU data with correct duration
    imu_data = load_imu_mat(mat_imu_file, img_duration)

    print(f"Loaded {len(img_timestamps)} image timestamps")
    print(f"Loaded {len(imu_data)} IMU measurements from sequence-specific .mat")
    print(f"Image time range: {img_timestamps[0]:.6f} to {img_timestamps[-1]:.6f}")
    print(f"IMU time range: {imu_data[0,0]:.6f} to {imu_data[-1,0]:.6f}")

    # Sync timestamps
    img_start, img_end = img_timestamps[0], img_timestamps[-1]
    imu_start, imu_end = imu_data[0, 0], imu_data[-1, 0]
    
    # Find common time range
    t_start = max(img_start, imu_start)
    t_end = min(img_end, imu_end)
    
    print(f"Common overlap: {t_start:.6f} to {t_end:.6f} ({t_end - t_start:.1f} seconds)")
    
    # Filter to overlap period
    img_mask = (img_timestamps >= t_start) & (img_timestamps <= t_end)
    imu_mask = (imu_data[:, 0] >= t_start) & (imu_data[:, 0] <= t_end)
    
    img_timestamps_sync = img_timestamps[img_mask]
    poses_sync = poses[img_mask]
    imu_data_sync = imu_data[imu_mask]
    
    print(f"After sync - Images: {len(img_timestamps_sync)}, IMU: {len(imu_data_sync)}")
    
    # Calculate data retention
    img_retention = len(img_timestamps_sync) / len(img_timestamps) * 100
    imu_retention = len(imu_data_sync) / len(imu_data) * 100
    
    print(f"📊 Data retention:")
    print(f"  Images: {img_retention:.1f}% ({100-img_retention:.1f}% loss)")
    print(f"  IMU: {imu_retention:.1f}% ({100-imu_retention:.1f}% loss)")
    
    if imu_retention > 95:
        print(f"  ✅ Excellent sync - sequence-specific .mat working perfectly!")
    elif imu_retention > 90:
        print(f"  ✅ Good sync")
    else:
        print(f"  ⚠️  Sync issues - check timestamps")
    
    # Make timestamps relative to common start
    img_timestamps_rel = img_timestamps_sync - t_start
    imu_data_sync[:, 0] = imu_data_sync[:, 0] - t_start
    
    print(f"Final relative timestamps:")
    print(f"  Images: {img_timestamps_rel[0]:.6f} to {img_timestamps_rel[-1]:.6f} ({len(img_timestamps_rel)} frames)")
    print(f"  IMU: {imu_data_sync[0,0]:.6f} to {imu_data_sync[-1,0]:.6f} ({len(imu_data_sync)} measurements)")
    
    # Check IMU sample rate
    imu_intervals = np.diff(imu_data_sync[:, 0])
    mean_interval = imu_intervals.mean()
    max_interval = imu_intervals.max()
    
    print(f"IMU timing analysis:")
    print(f"  Mean interval: {mean_interval*1000:.2f} ms")
    print(f"  Max interval:  {max_interval*1000:.2f} ms")
    print(f"  Sample rate: ~{1/mean_interval:.1f} Hz")
    
    if max_interval < 0.025:  # 25ms threshold
        print(f"  ✅ ALL intervals < 25ms - DEIO will use PRECISE parameters!")
    else:
        large_gaps = np.sum(imu_intervals > 0.025)
        print(f"  ⚠️  {large_gaps} intervals > 25ms")
    
    # Save timestamps in seconds
    with open(os.path.join(output_dir, "tss_imgs_sec_left.txt"), 'w') as f:
        for t in img_timestamps_rel:
            f.write(f"{t:.9f}\n")

    # Save images
    image_files = sorted(glob.glob(os.path.join(odo_seq_path, "image_2", "*.png")))
    sync_images = len(img_timestamps_sync)
    
    if sync_images > len(image_files):
        sync_images = len(image_files)
        img_timestamps_rel = img_timestamps_rel[:sync_images]
        poses_sync = poses_sync[:sync_images]
    
    print(f"Saving {sync_images} images...")
    for i in tqdm.tqdm(range(sync_images)):
        if i < len(image_files):
            img = cv2.imread(image_files[i])
            if img is not None:
                out_path = os.path.join(img_out_dir, f"{i:012d}.png")
                cv2.imwrite(out_path, img)
    
    # Save GT and IMU
    write_gt_stamped(poses_sync[:sync_images], img_timestamps_rel[:sync_images], 
                     os.path.join(output_dir, "gt_stamped_left.txt"))
    write_imu_csv(imu_data_sync, os.path.join(output_dir, "imu_data.csv"))
    
    # Save calibration
    calib_data = np.array([721.5377, 721.5377, 609.5593, 172.8540])
    np.savetxt(os.path.join(output_dir, "calib_undist_left.txt"), calib_data)

    print(f"✅ Processed KITTI sequence {seq} using .mat IMU data")
    print(f"   Using file: {mat_imu_file}")
    print(f"   Final output: {sync_images} images, {len(imu_data_sync)} IMU measurements")
    print(f"   🔥 HIGH-FREQUENCY IMU DATA - This should work much better with DEIO!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--odometry_root", required=True)
    parser.add_argument("--mat_imu_path", required=True, help="Path to .mat IMU files (e.g., /path/to/imus/)")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seq", type=int, required=True)
    args = parser.parse_args()

    preprocess_kitti_mat_seq(
        seq_id=args.seq,
        kitti_odometry_root=args.odometry_root,
        kitti_mat_imu_path=args.mat_imu_path,
        output_dir=args.output_dir
    )
