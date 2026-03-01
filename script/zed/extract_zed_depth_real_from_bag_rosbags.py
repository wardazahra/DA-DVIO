#!/usr/bin/env python3
"""
Reference script for extracting depth from ZED ROS bags using rosbags library
"""
import cv2
import numpy as np
from pathlib import Path
from rosbags.rosbag1 import Reader
from rosbags.typesys import get_typestore, Stores

def ros_image_to_numpy(msg):
    """Convert ROS Image message to numpy array"""
    height = msg.height
    width = msg.width
    encoding = msg.encoding
    
    dtype_map = {
        '32FC1': np.float32,
        '16UC1': np.uint16,
        'mono16': np.uint16,
    }
    
    dtype = dtype_map.get(encoding, np.uint8)
    img_array = np.frombuffer(bytes(msg.data), dtype=dtype)
    
    if encoding == '32FC1':
        img_array = img_array.reshape((height, width))
    elif encoding in ['16UC1', 'mono16']:
        img_array = img_array.reshape((height, width))
    else:
        img_array = img_array.reshape((height, width, -1))
    
    return img_array

def extract_depth_from_bag(bag_file, output_dir, depth_topic='/zed2/zed_node/depth/depth_registered'):
    """Extract depth images from ZED ROS bag"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    bag_path = Path(bag_file)
    print(f"📂 Opening bag: {bag_file}")
    
    typestore = get_typestore(Stores.ROS1_NOETIC)
    
    with Reader(bag_path) as reader:
        connections = [c for c in reader.connections if c.topic == depth_topic]
        
        if not connections:
            print(f"❌ Topic '{depth_topic}' not found!")
            print(f"Available topics:")
            for c in reader.connections:
                print(f"   - {c.topic}")
            return
        
        print(f"🚀 Extracting depth images...")
        depth_count = 0
        
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            try:
                msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
                cv_image = ros_image_to_numpy(msg)
                
                if msg.encoding == '32FC1':
                    depth_mm = np.clip(cv_image * 1000.0, 0, 65535).astype(np.uint16)
                else:
                    depth_mm = cv_image.astype(np.uint16)
                
                timestamp_ns = msg.header.stamp.sec * 1000000000 + msg.header.stamp.nanosec
                output_file = output_dir / f"{timestamp_ns}.png"
                
                cv2.imwrite(str(output_file), depth_mm)
                depth_count += 1
                
                # Print progress every 100 frames
                if depth_count % 100 == 0:
                    print(f"   Extracted {depth_count} frames...")
                    
            except Exception as e:
                print(f"\n⚠️  Error at frame {depth_count}: {e}")
        
        print(f"✅ Extracted {depth_count} depth images to {output_dir}")
        
        # Verify first image
        png_files = sorted(output_dir.glob("*.png"))
        if png_files:
            first_file = png_files[0]
            test_depth = cv2.imread(str(first_file), cv2.IMREAD_ANYDEPTH)
            print(f"\n📊 Verification:")
            print(f"   Shape: {test_depth.shape}")
            print(f"   Range: {test_depth.min()} - {test_depth.max()} mm")

if __name__ == '__main__':
    sequences = {
        'z09': '/media/t2508/Tra/z09.bag',
    }
    
    for seq_name, bag_path in sequences.items():
        if not Path(bag_path).exists():
            print(f"⚠️  Skipping {seq_name}: bag not found")
            continue
        
        output_dir = f"/media/t2508/Tra/{seq_name}/mav0/depth_real/"
        
        if Path(output_dir).exists() and len(list(Path(output_dir).glob("*.png"))) > 0:
            print(f"✅ {seq_name} already extracted ({len(list(Path(output_dir).glob('*.png')))} files)")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing {seq_name}")
        print(f"{'='*60}")
        extract_depth_from_bag(bag_path, output_dir)
