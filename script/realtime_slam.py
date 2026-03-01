#!/usr/bin/env python3
"""
STEP 3: ROS → Queue → DEIO (with real-time ROS depth and CORRECT latency measurement)
"""

import sys
import cv2
import numpy as np
import torch
from pathlib import Path
import time
from queue import Queue
import rospy
import os
from cv_bridge import CvBridge
from threading import Lock

os.environ['ROS_MASTER_URI'] = 'http://localhost:11311'
sys.path.insert(0, '/media/t2508/C477-48FA/DEIO')

from dpvo.config import cfg
from devo.dba import DBA
import gtsam
from sensor_msgs.msg import Image, Imu, CameraInfo


class ROSFeeder:
    """ROS callbacks fill queue"""
    def __init__(self, queue, target_H=376, target_W=1024, stride=1):
        self.queue = queue
        self.target_H = target_H
        self.target_W = target_W
        self.stride = stride
        self.bridge = CvBridge()
        
        self.intrinsics = None
        
        # IMU accumulator - NEVER CLEARED (like file version)
        self.accumulated_imu = []
        self.imu_lock = Lock()
        
        # Depth cache with timestamps
        self.depth_cache = {}
        self.depth_lock = Lock()
        
        self.frame_count = 0
        self.frames_queued = 0
        self.dropped_frames = 0
        self.depth_received = 0
        self.last_frame_time = time.time()
        self.lock = Lock()
        
        print(f"ROSFeeder: Target resolution {target_W}x{target_H}, stride={stride}")
    
    def camera_info_cb(self, msg):
        if self.intrinsics is not None:
            return
        
        K = np.array(msg.K).reshape(3, 3)
        fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
        orig_W, orig_H = msg.width, msg.height
        
        sx = self.target_W / float(orig_W)
        sy = self.target_H / float(orig_H)
        
        self.intrinsics = {
            'fx': fx * sx,
            'fy': fy * sy,
            'cx': cx * sx,
            'cy': cy * sy,
            'orig_W': orig_W,
            'orig_H': orig_H
        }
        print(f"✅ Intrinsics: {orig_W}x{orig_H} → {self.target_W}x{self.target_H}")
    
    def imu_cb(self, msg):
        """Accumulate IMU continuously - NEVER CLEAR (like file version)"""
        t = msg.header.stamp.to_sec()
        
        with self.imu_lock:
            self.accumulated_imu.append([
                t,
                msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z,
                msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z
            ])
        
        # Update last frame time
        with self.lock:
            self.last_frame_time = time.time()
    
    def depth_cb(self, msg):
        """Store depth images by timestamp"""
        if self.intrinsics is None:
            return
        
        try:
            timestamp = msg.header.stamp.to_sec()
            
            # Convert depth image
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
            depth_map = depth_image.astype(np.float32)
            
            # Clean and clip
            depth_map = np.clip(depth_map, 0.5, 30.0)
            depth_map[depth_map <= 0.1] = 0.0
            depth_map[np.isnan(depth_map)] = 0.0
            
            # Resize if needed
            if depth_map.shape != (self.target_H, self.target_W):
                depth_map = cv2.resize(depth_map, (self.target_W, self.target_H),
                                      interpolation=cv2.INTER_NEAREST)
            
            with self.depth_lock:
                self.depth_cache[timestamp] = depth_map
                self.depth_received += 1
                
                # Keep cache size manageable (last 200 depths)
                if len(self.depth_cache) > 200:
                    oldest = min(self.depth_cache.keys())
                    del self.depth_cache[oldest]
                    
        except Exception as e:
            print(f"⚠️  Depth error: {e}")
    
    def get_depth_for_timestamp(self, target_timestamp, tolerance=0.05):
        """Get depth image closest to target timestamp within tolerance"""
        with self.depth_lock:
            if len(self.depth_cache) == 0:
                return None
            
            # Find closest timestamp
            closest_ts = min(self.depth_cache.keys(), 
                           key=lambda t: abs(t - target_timestamp))
            
            # Check if within tolerance
            if abs(closest_ts - target_timestamp) < tolerance:
                return self.depth_cache[closest_ts].copy()
            
            return None
    
    def get_accumulated_imu(self):
        """Get snapshot of accumulated IMU (like file version)"""
        with self.imu_lock:
            return list(self.accumulated_imu)
    
    def image_cb(self, msg):
        if self.intrinsics is None:
            return
        
        self.frame_count += 1
        
        # Update last frame time
        with self.lock:
            self.last_frame_time = time.time()
        
        # Apply stride in callback
        if (self.frame_count - 1) % self.stride != 0:
            return
        
        timestamp = msg.header.stamp.to_sec()
        
        cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        
        if cv_image.shape[:2] != (self.target_H, self.target_W):
            cv_image = cv2.resize(cv_image, (self.target_W, self.target_H),
                                 interpolation=cv2.INTER_LINEAR)
        
        frame_data = {
            'timestamp': timestamp,
            'image': cv_image,
            'queue_entry_time': time.time()  # ADDED: Track when frame enters queue
        }
        
        if self.queue.full():
            self.dropped_frames += 1
            if self.dropped_frames % 10 == 0:
                print(f"⚠️  Dropped {self.dropped_frames} frames (queue full)")
        else:
            self.queue.put(frame_data)
            self.frames_queued += 1
        
        if self.frames_queued % 100 == 0:
            print(f"ROSFeeder: Queued {self.frames_queued} frames, Depth received {self.depth_received}, queue size={self.queue.qsize()}")
    
    def is_still_receiving(self):
        """Check if we're still receiving data from bag"""
        with self.lock:
            return (time.time() - self.last_frame_time) < 15.0
    
    def get_intrinsics(self):
        if self.intrinsics is None:
            return None
        return (self.intrinsics['fx'], self.intrinsics['fy'],
                self.intrinsics['cx'], self.intrinsics['cy'])
    
    def get_depth_stats(self):
        with self.depth_lock:
            return {
                'received': self.depth_received,
                'cached': len(self.depth_cache)
            }


@torch.no_grad()
def run_from_queue(queue, cfg, network, feeder, imu_lookahead=0.2):
    
    print("\n" + "="*80)
    print("DEIO Processing from Queue")
    print("="*80)
    
    # Wait for intrinsics
    print("Waiting for camera intrinsics...")
    while feeder.get_intrinsics() is None and not rospy.is_shutdown():
        time.sleep(0.1)
    
    if rospy.is_shutdown():
        return None, None
    
    target_H = feeder.target_H
    target_W = feeder.target_W
    
    fx_s, fy_s, cx_s, cy_s = feeder.get_intrinsics()
    intrinsics = torch.tensor([fx_s, fy_s, cx_s, cy_s], dtype=torch.float32).cuda()
    
    # Initialize SLAM
    slam = DBA(cfg, network, evs=False, ht=target_H, wd=target_W, viz=False)
    
    # IMU setup
    if getattr(cfg, "ENALBE_IMU", False):
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
        slam.visual_only = False
        slam.ignore_imu = False
        slam.imu_enabled = False
        slam.all_imu = np.array([]).reshape(0, 7)
        print("IMU: Enabled")
    else:
        slam.visual_only = True
        slam.imu_enabled = False
        slam.ignore_imu = True
        slam.all_imu = None
        slam.Ti1c = np.eye(4)
        print("IMU: Disabled")
    
    slam.all_gt = None
    slam.all_gt_keys = None
    
    print("\nProcessing frames from queue with real-time depth...")
    print("="*80)
    
    processed = 0
    skipped = 0
    depth_count = 0
    depth_matched = 0
    depth_miss = 0
    processing_times = []
    e2e_latencies = []
    queue_wait_times = []
    
    while not rospy.is_shutdown():
        # Get frame from queue (BLOCKS - DEIO controls pace!)
        try:
            frame_data = queue.get(timeout=0.5)
        except:
            # Queue empty - check if bag is still running
            if not feeder.is_still_receiving() and queue.qsize() == 0:
                print(f"\nBag finished and queue empty, stopping...")
                break
            continue
        
        timestamp = frame_data['timestamp']
        img = frame_data['image']
        queue_entry_time = frame_data.get('queue_entry_time', time.time())  # ADDED
        
        # Track when we start processing this frame
        processing_start_wall = time.time()
        queue_wait_time = processing_start_wall - queue_entry_time
        queue_wait_times.append(queue_wait_time)
        
        # Wait for sufficient IMU before processing
        imu_ready = False
        if getattr(cfg, "ENALBE_IMU", False):
            max_wait_time = 2.0
            wait_start = time.time()
            
            while (time.time() - wait_start) < max_wait_time:
                accumulated_imu = feeder.get_accumulated_imu()
                
                if accumulated_imu:
                    latest_imu_time = accumulated_imu[-1][0]
                    required_time = timestamp + imu_lookahead + 0.05
                    
                    if latest_imu_time >= required_time:
                        slam.all_imu = np.array(accumulated_imu)
                        imu_ready = True
                        break
                
                if not feeder.is_still_receiving():
                    break
                
                time.sleep(0.05)
            
            if not imu_ready:
                skipped += 1
                if skipped % 10 == 0:
                    print(f"⚠️  Skipped {skipped} frames due to insufficient IMU")
                continue
        
        # Get depth from ROS (timestamp matching with tolerance)
        depth_map = feeder.get_depth_for_timestamp(timestamp, tolerance=0.15)
        
        if depth_map is not None:
            depth_matched += 1
        else:
            depth_miss += 1
        
        depth_count = depth_matched
        
        # Convert to tensor
        image_tensor = torch.from_numpy(img).permute(2, 0, 1).float().cuda()
        
        # Process (DEIO blocks here, controls pace!)
        process_start = time.time()
        try:
            slam(timestamp, image_tensor, intrinsics, scale=1.0, psmnet_depth=depth_map)
        except Exception as e:
            print(f"❌ Error at frame {processed}: {e}")
            import traceback
            traceback.print_exc()
            del image_tensor
            skipped += 1
            continue
        
        process_end = time.time()
        
        # Calculate latencies
        processing_time = process_end - process_start
        end_to_end_latency = process_end - queue_entry_time  # FIXED: From queue entry to output
        
        processing_times.append(processing_time)
        e2e_latencies.append(end_to_end_latency)
        
        # Cleanup
        del image_tensor
        
        processed += 1
        
        if processed % 50 == 0:
            avg_proc = np.mean(processing_times[-50:]) * 1000
            avg_e2e = np.mean(e2e_latencies[-50:]) * 1000
            avg_queue = np.mean(queue_wait_times[-50:]) * 1000
            qsize = queue.qsize()
            imu_count = len(feeder.accumulated_imu)
            depth_stats = feeder.get_depth_stats()
            print(f"Frame {processed}: Process={avg_proc:.1f}ms | E2E={avg_e2e:.1f}ms | Queue wait={avg_queue:.1f}ms | "
                  f"Qsize={qsize} | IMU={imu_count} | "
                  f"Depth={depth_matched}/{processed} (recv={depth_stats['received']}, miss={depth_miss})")
    
    if processed == 0:
        print("⚠️  No frames processed!")
        return None, None
    
    # Final optimization
    print(f"\n🏁 Processed {processed} frames, running final optimization...")
    for i in range(12):
        try:
            slam.update()
        except Exception as e:
            print(f"Warning: optimization iteration {i} failed: {e}")
    
    try:
        poses, tstamps = slam.terminate()
    except Exception as e:
        print(f"❌ Error during terminate: {e}")
        import traceback
        traceback.print_exc()
        return None, None
    
    depth_stats = feeder.get_depth_stats()
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Frames received: {feeder.frame_count}")
    print(f"Frames queued: {feeder.frames_queued}")
    print(f"Frames processed: {processed}")
    print(f"Frames skipped: {skipped}")
    print(f"Frames dropped: {feeder.dropped_frames}")
    print(f"Poses generated: {len(poses) if poses is not None else 0}")
    print(f"Depth received total: {depth_stats['received']}")
    print(f"Depth matched: {depth_matched}/{processed} ({100.0*depth_matched/max(processed,1):.1f}%)")
    print(f"Depth missed: {depth_miss}")
    
    # Latency statistics
    if e2e_latencies:
        print(f"\n--- LATENCY STATISTICS ---")
        print(f"Queue wait time: avg={np.mean(queue_wait_times)*1000:.1f}ms, "
              f"min={np.min(queue_wait_times)*1000:.1f}ms, "
              f"max={np.max(queue_wait_times)*1000:.1f}ms")
        print(f"Processing time: avg={np.mean(processing_times)*1000:.1f}ms, "
              f"min={np.min(processing_times)*1000:.1f}ms, "
              f"max={np.max(processing_times)*1000:.1f}ms")
        print(f"End-to-end latency: avg={np.mean(e2e_latencies)*1000:.1f}ms, "
              f"min={np.min(e2e_latencies)*1000:.1f}ms, "
              f"max={np.max(e2e_latencies)*1000:.1f}ms")
        print(f"  (E2E = Queue wait + Processing time)")
    
    print("="*80)
    
    return poses, tstamps


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--queue_size', type=int, default=10)
    parser.add_argument('--imu_lookahead', type=float, default=0.2)
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--output', default='saved_trajectories/rosbag_ros_depth.txt')
    parser.add_argument('--opts', nargs='+', default=[])
    
    args = parser.parse_args()
    
    cfg.merge_from_file(args.config)
    cfg.merge_from_list(args.opts)
    cfg.resnet = False
    
    torch.manual_seed(1234)
    
    rospy.init_node('deio_ros_queue', anonymous=True)
    
    # Create queue
    frame_queue = Queue(maxsize=args.queue_size)
    
    # Create ROS feeder
    feeder = ROSFeeder(frame_queue, target_H=376, target_W=1024, stride=args.stride)
    
    # Setup ROS subscribers (including depth!)
    rospy.Subscriber('/zed2/zed_node/left/camera_info', CameraInfo, feeder.camera_info_cb)
    rospy.Subscriber('/zed2/zed_node/left/image_rect_color', Image, feeder.image_cb, queue_size=1)
    rospy.Subscriber('/zed2/zed_node/imu/data', Imu, feeder.imu_cb, queue_size=100)
    rospy.Subscriber('/zed2/zed_node/depth/depth_registered', Image, feeder.depth_cb, queue_size=10)
    
    print("="*80)
    print("STEP 3: ROS → Queue → DEIO (with real-time ROS depth)")
    print(f"Queue size: {args.queue_size}")
    print(f"Stride: {args.stride}")
    print(f"IMU lookahead: {args.imu_lookahead}s")
    print("="*80)
    print("\n🎬 Ready! Start rosbag play now\n")
    
    time.sleep(1.0)
    
    # Process from queue
    poses, tstamps = run_from_queue(
        frame_queue, cfg, args.network, feeder,
        imu_lookahead=args.imu_lookahead
    )
    
    if args.save and poses is not None and tstamps is not None and len(poses) > 0:
        from evo.core.trajectory import PoseTrajectory3D
        from evo.tools import file_interface
        
        Path(args.output).parent.mkdir(exist_ok=True, parents=True)
        traj = PoseTrajectory3D(
            positions_xyz=poses[:,:3],
            orientations_quat_wxyz=poses[:, [6, 3, 4, 5]],
            timestamps=tstamps
        )
        
        file_interface.write_tum_trajectory_file(args.output, traj)
        print(f"\n💾 Trajectory saved: {args.output}")
    else:
        print("\n⚠️  No valid trajectory to save")

if __name__ == '__main__':
    main()
