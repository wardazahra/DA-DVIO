import math
import numpy as np
import torch
import torchvision

# from devo.lietorch import SE3

class SE3:
    """SE3 class extracted from DPVO lietorch - minimal implementation"""
    
    def __init__(self, data):
        """Initialize SE3 from data in format [x, y, z, qx, qy, qz, qw]"""
        if isinstance(data, torch.Tensor):
            self.data = data
        else:
            self.data = torch.tensor(data, dtype=torch.float32)
    
    def scale(self, s):
        """Scale translation components - exact implementation from DPVO"""
        t, q = self.data.split([3, 4], -1)  # Split into translation [3] and quaternion [4]
        t = t * s.unsqueeze(-1)              # Scale translation
        return SE3(torch.cat([t, q], dim=-1)) # Recombine

def transform_rescale(scale, voxels, disps=None, poses=None, intrinsics=None):
    """Transform voxels/images, depth maps, poses and intrinsics (n_frames,*)"""
    H, W = voxels.shape[-2:]
    nH, nW = math.floor(scale * H), math.floor(scale * W)
    resize = torchvision.transforms.Resize((nH, nW))

    voxels = resize(voxels)
    if disps is not None:
        disps = resize(disps)
    if poses is not None:
        poses = transform_rescale_poses(scale, poses)
    if intrinsics is not None:
        intrinsics = scale * intrinsics

    return voxels, disps, poses, intrinsics

def transform_rescale_poses(scale, poses):
    s = torch.tensor(scale)
    poses = SE3(poses).scale(s).data
    return poses
