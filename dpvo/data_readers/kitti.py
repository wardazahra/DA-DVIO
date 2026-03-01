import cv2
import numpy as np
import torch
import glob
import os
import os.path as osp
from pathlib import Path
from scipy.spatial.transform import Rotation

from .base import RGBDDataset


def read_calib_file(filepath):
    """Read KITTI calib file into dict of numpy arrays."""
    data = {}
    with open(filepath, 'r') as f:
        for line in f.readlines():
            if ':' not in line:
                continue
            key, value = line.split(':', 1)
            try:
                data[key.strip()] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass
    return data


class KITTI(RGBDDataset):
    """
    KITTI Odometry dataset with stereo depths and train/val split
    Supports sequences 00, 05, 07 for training, 10 for validation
    """
    
    # ✅ Class-level sequence configuration
    TRAIN_SEQUENCES = ['00', '05', '07']  # Training sequences
    VAL_SEQUENCES = ['10']                 # Validation sequence
    
    DEPTH_SCALE = 1.0  # KITTI already in meters
    
    def __init__(self, mode='training', psmnet_dir=None, sequences=None, 
                 load_stereo=True, return_depth_gt=True, **kwargs):
        self.mode = mode
        self.psmnet_dir = psmnet_dir
        self.load_stereo = load_stereo
        self.return_depth_gt = return_depth_gt
        
        # ✅ Auto-select sequences based on mode
        if sequences is None:
            if mode == 'training':
                self.sequences = self.TRAIN_SEQUENCES
                print(f"📚 Training mode - using sequences: {self.sequences}")
            elif mode == 'validation':
                self.sequences = self.VAL_SEQUENCES
                print(f"🔍 Validation mode - using sequences: {self.sequences}")
            else:
                raise ValueError(f"Unknown mode: {mode}. Use 'training' or 'validation'")
        else:
            self.sequences = sequences
            print(f"🎯 Custom sequences: {self.sequences}")
        
        print(f"✅ [DEBUG] psmnet_dir = {self.psmnet_dir}")
        print(f"✅ [DEBUG] datapath = {kwargs.get('datapath', 'NOT SET')}")
        print(f"✅ [DEBUG] load_stereo = {self.load_stereo}")
        print(f"✅ [DEBUG] return_depth_gt = {self.return_depth_gt}")
        
        super(KITTI, self).__init__(name=f'KITTI_{mode}', psmnet_dir=psmnet_dir, **kwargs)
    
    @staticmethod
    def is_test_scene(scene):
        """No test scenes - we split by sequences instead"""
        return False
    
    def _build_dataset(self):
        """
        Build KITTI dataset structure matching TartanAir format
        This is called once and cached to pickle file
        """
        from tqdm import tqdm
        print(f"Building KITTI dataset with frame graph for sequences {self.sequences}...")
        
        scene_info = {}
        
        for seq in tqdm(self.sequences, desc="Processing KITTI sequences"):
            # Handle both path formats
            seq_path = Path(self.root) / seq
            if not seq_path.exists():
                seq_path = Path(self.root) / "sequences" / seq
            
            if not seq_path.exists():
                print(f"  ⚠️  Warning: Sequence path not found")
                print(f"      Tried: {Path(self.root) / seq}")
                print(f"      Tried: {Path(self.root) / 'sequences' / seq}")
                continue
            
            print(f"  ✅ Found sequence path: {seq_path}")
            
            # ✅ Load LEFT images (image_2)
            images_left = sorted(glob.glob(str(seq_path / "image_2" / "*.png")))
            
            if len(images_left) == 0:
                print(f"  ⚠️  No left images found in {seq_path / 'image_2'}")
                continue
            
            # ✅ Load RIGHT images (image_3) for stereo
            if self.load_stereo:
                images_right = sorted(glob.glob(str(seq_path / "image_3" / "*.png")))
                
                if len(images_right) != len(images_left):
                    print(f"  ⚠️  Warning: Mismatched stereo pair count")
                    print(f"     Left (image_2): {len(images_left)}")
                    print(f"     Right (image_3): {len(images_right)}")
                    # Truncate to minimum
                    min_len = min(len(images_left), len(images_right))
                    images_left = images_left[:min_len]
                    images_right = images_right[:min_len]
                
                print(f"  📸 Sequence {seq}: Found {len(images_left)} stereo pairs")
            else:
                images_right = [None] * len(images_left)
                print(f"  📸 Sequence {seq}: Found {len(images_left)} left images only")
            
            # Load stereo depths
            depths = []
            depth_count = 0
            missing_count = 0
            
            for img_path in images_left:
                frame_id = int(Path(img_path).stem)
                depth_file = None
                
                if self.psmnet_dir:
                    # Try multiple naming conventions
                    possible_paths = [
                        Path(self.psmnet_dir) / seq / f"{frame_id:06d}.npy",
                        Path(self.psmnet_dir) / seq / f"{frame_id:06d}_depth.npy",
                        Path(self.psmnet_dir) / seq / "depth_maps" / f"{frame_id:06d}.npy",
                        Path(self.psmnet_dir) / seq / "depth_maps" / f"{frame_id:06d}_depth.npy",
                    ]
                    
                    for path in possible_paths:
                        if path.exists():
                            depth_file = path
                            break
                
                if depth_file and depth_file.exists():
                    depths.append(str(depth_file))
                    depth_count += 1
                else:
                    depths.append(None)
                    missing_count += 1
                    
                    if missing_count <= 3:
                        print(f"      ⚠️  Missing depth for frame {frame_id:06d}")
            
            print(f"  🌊 Sequence {seq}: Found {depth_count}/{len(images_left)} depth maps")
            
            if depth_count == 0:
                print(f"  ⚠️  ERROR: No depth maps found for sequence {seq}!")
                print(f"       psmnet_dir: {self.psmnet_dir}")
                continue
            
            # Filter out frames without depth
            valid_indices = [i for i, d in enumerate(depths) if d is not None]
            
            if len(valid_indices) < 15:
                print(f"  ⚠️  Only {len(valid_indices)} valid frames, need at least 15. Skipping.")
                continue
            
            images_left = [images_left[i] for i in valid_indices]
            images_right = [images_right[i] for i in valid_indices] if self.load_stereo else [None] * len(valid_indices)
            depths = [depths[i] for i in valid_indices]
            
            print(f"  ✅ Sequence {seq}: Using {len(valid_indices)} frames with valid depths")
            
            # Load GT poses
            poses_file = Path(self.root) / "poses" / f"{seq}.txt"
            if not poses_file.exists():
                poses_file = Path(self.root).parent / "poses" / f"{seq}.txt"
            if not poses_file.exists():
                poses_file = Path(self.root) / ".." / "poses" / f"{seq}.txt"
            
            if not poses_file.exists():
                print(f"  ⚠️  ERROR: No poses file found for sequence {seq}")
                continue
            
            print(f"  ✅ Found poses: {poses_file}")
            
            poses_matrices = np.loadtxt(poses_file)
            poses = self._poses_matrix_to_7d(poses_matrices)
            poses = poses[valid_indices]
            
            # Load calibration
            calib_file = seq_path / "calib.txt"
            if calib_file.exists():
                calib_data = read_calib_file(calib_file)
                P2 = calib_data.get('P2', np.array([718.856, 0, 607.1928, 0,
                                                     0, 718.856, 185.2157, 0,
                                                     0, 0, 1, 0]))
                fx, fy, cx, cy = P2[0], P2[5], P2[2], P2[6]
            else:
                fx, fy, cx, cy = 718.856, 718.856, 607.1928, 185.2157
            
            intrinsics = [np.array([fx, fy, cx, cy])] * len(images_left)
            
            # Build simplified frame graph
            print(f"  🔗 Sequence {seq}: Building frame graph (temporal proximity)...")
            graph = self.build_simple_frame_graph(len(poses), max_temporal_dist=50)
            print(f"  ✅ Sequence {seq}: Frame graph built with {len(graph)} nodes")
            
            scene_name = f'sequences/{seq}'
            scene_info[scene_name] = {
                'images': images_left,
                'images_right': images_right,  # ✅ Store right images
                'depths': depths,
                'poses': poses,
                'intrinsics': intrinsics,
                'graph': graph
            }
            
            print(f"  ✅ Sequence {seq} loaded successfully\n")
        
        if len(scene_info) == 0:
            raise RuntimeError(
                "❌ No KITTI sequences could be loaded!\n"
                "   Check paths and depth files."
            )
        
        print(f"✅ Successfully loaded {len(scene_info)} sequence(s)")
        return scene_info
    
    def build_simple_frame_graph(self, num_frames, max_temporal_dist=50):
        """Build simplified frame graph based on temporal proximity"""
        graph = {}
        
        for i in range(num_frames):
            neighbors = []
            distances = []
            
            for j in range(max(0, i - max_temporal_dist), 
                          min(num_frames, i + max_temporal_dist + 1)):
                if i != j:
                    neighbors.append(j)
                    temporal_gap = abs(i - j)
                    estimated_flow = 10.0 + (temporal_gap * 1.3)
                    distances.append(estimated_flow)
            
            graph[i] = (np.array(neighbors), np.array(distances))
        
        return graph
    
    @staticmethod
    def _poses_matrix_to_7d(pose_matrices):
        """Convert KITTI 3x4 pose matrices to 7D format [tx, ty, tz, qx, qy, qz, qw]"""
        poses_7d = []
        
        for i in range(len(pose_matrices)):
            T = np.eye(4)
            T[:3, :] = pose_matrices[i].reshape(3, 4)
            
            t = T[:3, 3]
            R = T[:3, :3]
            quat = Rotation.from_matrix(R).as_quat()
            
            poses_7d.append(np.concatenate([t, quat]))
        
        return np.array(poses_7d, dtype=np.float32)
    
    @staticmethod
    def image_read(image_file):
        """Read image as BGR uint8"""
        if image_file is None:
            return None
        img = cv2.imread(image_file)
        if img is None:
            raise ValueError(f"Failed to read image: {image_file}")
        return img
    
    @staticmethod
    def depth_read(depth_file):
        """Read depth map from .npy file"""
        if depth_file is None or not os.path.exists(depth_file):
            raise ValueError(f"Depth file not found: {depth_file}")
        
        depth = np.load(depth_file).astype(np.float32)
        
        # Clean up invalid values
        depth[np.isnan(depth)] = 1.0
        depth[np.isinf(depth)] = 1.0
        depth[depth < 0.01] = 1.0
        depth[depth > 80.0] = 80.0
        
        return depth
    
    def __getitem__(self, index):
        """
        Returns training sample with optional stereo images and depth GT
        
        Returns:
            If return_depth_gt=True and load_stereo=True:
                images_left, images_right, poses, disps, intrinsics, depths_gt
            If return_depth_gt=True and load_stereo=False:
                images_left, poses, disps, intrinsics, depths_gt
            Else:
                images_left, poses, disps, intrinsics
        """
        index = index % len(self.dataset_index)
        scene_id, ix = self.dataset_index[index]
        
        frame_graph = self.scene_info[scene_id]['graph']
        images_left_list = self.scene_info[scene_id]['images']
        images_right_list = self.scene_info[scene_id].get('images_right', [None]*len(images_left_list))
        depths_list = self.scene_info[scene_id]['depths']
        poses_list = self.scene_info[scene_id]['poses']
        intrinsics_list = self.scene_info[scene_id]['intrinsics']
        
        # Sample frames using graph
        inds = [ix]
        while len(inds) < self.n_frames:
            if self.sample:
                k = (frame_graph[ix][1] > self.fmin) & (frame_graph[ix][1] < self.fmax)
                frames = frame_graph[ix][0][k]
                
                if np.count_nonzero(frames[frames > ix]):
                    ix = np.random.choice(frames[frames > ix])
                elif ix + 1 < len(images_left_list):
                    ix = ix + 1
                elif np.count_nonzero(frames):
                    ix = np.random.choice(frames)
                else:
                    if ix + 1 < len(images_left_list):
                        ix = ix + 1
                    else:
                        break
            else:
                if ix + 1 < len(images_left_list):
                    ix = ix + 1
                else:
                    break
            
            inds.append(ix)
        
        # Load data
        images_left, images_right, depths, poses, intrinsics = [], [], [], [], []
        
        for i in inds:
            # Left image
            left = self.__class__.image_read(images_left_list[i])
            images_left.append(left)
            
            # Right image (if available)
            if self.load_stereo and images_right_list[i] is not None:
                right = self.__class__.image_read(images_right_list[i])
                images_right.append(right)
            
            # Depth
            depth = self.__class__.depth_read(depths_list[i])
            depths.append(depth)
            
            # Pose and intrinsics
            poses.append(poses_list[i])
            intrinsics.append(intrinsics_list[i])
        
        # Convert to tensors
        images_left = np.stack(images_left).astype(np.float32)
        images_left = torch.from_numpy(images_left).permute(0, 3, 1, 2).float()  # [T, 3, H, W]
        
        if self.load_stereo and len(images_right) > 0:
            images_right = np.stack(images_right).astype(np.float32)
            images_right = torch.from_numpy(images_right).permute(0, 3, 1, 2).float()
        else:
            images_right = None
        
        depths = np.stack(depths).astype(np.float32)
        poses = np.stack(poses).astype(np.float32)
        intrinsics = np.stack(intrinsics).astype(np.float32)
        
        # Convert depth to disparity
        disps = torch.from_numpy(1.0 / depths)
        poses = torch.from_numpy(poses)
        intrinsics = torch.from_numpy(intrinsics)
        
        # ✅ Keep depths as GT (before normalization!)
        depths_gt = torch.from_numpy(depths).clone()
        
        # Apply augmentation if enabled
        if self.aug:
            images_left, poses, disps, intrinsics = \
                self.aug(images_left, poses, disps, intrinsics)
        
        # Normalize disps and poses
        s = .7 * torch.quantile(disps, .98)
        disps = disps / s
        poses[..., :3] *= s
        
        # Return based on configuration
        if images_right is not None and self.load_stereo:
            # Format: images_left, images_right, poses, disps, intrinsics, depths_gt
            return images_left, images_right, poses, disps, intrinsics, depths_gt
        else:
            # Format: images_left, poses, disps, intrinsics, depths_gt  
            return images_left, poses, disps, intrinsics, depths_gt
