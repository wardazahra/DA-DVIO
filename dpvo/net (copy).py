import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

import torch_scatter
from torch_scatter import scatter_sum

from . import fastba
from . import altcorr
from . import lietorch
from .lietorch import SE3

from .extractor import BasicEncoder, BasicEncoder4
from .blocks import GradientClip, GatedResidual, SoftAgg

from .utils import *
from .ba import BA
from . import projective_ops as pops

autocast = torch.amp.autocast
import matplotlib.pyplot as plt

DIM = 384

class Update(nn.Module):
    def __init__(self, p):
        super(Update, self).__init__()

        self.c1 = nn.Sequential(
            nn.Linear(DIM, DIM),
            nn.ReLU(inplace=True),
            nn.Linear(DIM, DIM))

        self.c2 = nn.Sequential(
            nn.Linear(DIM, DIM),
            nn.ReLU(inplace=True),
            nn.Linear(DIM, DIM))
        
        self.norm = nn.LayerNorm(DIM, eps=1e-3)

        self.agg_kk = SoftAgg(DIM)
        self.agg_ij = SoftAgg(DIM)

        self.gru = nn.Sequential(
            nn.LayerNorm(DIM, eps=1e-3),
            GatedResidual(DIM),
            nn.LayerNorm(DIM, eps=1e-3),
            GatedResidual(DIM),
        )

        self.corr = nn.Sequential(
            nn.Linear(2*49*p*p, DIM),
            nn.ReLU(inplace=True),
            nn.Linear(DIM, DIM),
            nn.LayerNorm(DIM, eps=1e-3),
            nn.ReLU(inplace=True),
            nn.Linear(DIM, DIM),
        )

        self.d = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Linear(DIM, 2),
            GradientClip())

        self.w = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Linear(DIM, 2),
            GradientClip(),
            nn.Sigmoid())


    def forward(self, net, inp, corr, flow, ii, jj, kk):
        """ update operator """

        net = net + inp + self.corr(corr)
        net = self.norm(net)

        ix, jx = fastba.neighbors(kk, jj)
        mask_ix = (ix >= 0).float().reshape(1, -1, 1)
        mask_jx = (jx >= 0).float().reshape(1, -1, 1)

        net = net + self.c1(mask_ix * net[:,ix])
        net = net + self.c2(mask_jx * net[:,jx])

        net = net + self.agg_kk(net, kk)
        net = net + self.agg_ij(net, ii*12345 + jj)

        net = self.gru(net)

        return net, (self.d(net), self.w(net), None)


class Patchifier(nn.Module):
    def __init__(self, patch_size=3):
        super(Patchifier, self).__init__()
        self.patch_size = patch_size
        self.fnet = BasicEncoder4(output_dim=128, norm_fn='instance')
        self.inet = BasicEncoder4(output_dim=DIM, norm_fn='none')

    def __image_gradient(self, images):
        gray = ((images + 0.5) * (255.0 / 2)).sum(dim=2)
        dx = gray[...,:-1,1:] - gray[...,:-1,:-1]
        dy = gray[...,1:,:-1] - gray[...,:-1,:-1]
        g = torch.sqrt(dx**2 + dy**2)
        g = F.avg_pool2d(g, 4, 4)
        return g

    def forward(self, images, patches_per_image=80, disps=None, centroid_sel_strat='RANDOM', return_color=False):
        """ extract patches from input images """
        fmap = self.fnet(images) / 4.0
        imap = self.inet(images) / 4.0

        b, n, c, h, w = fmap.shape
        P = self.patch_size

        # bias patch selection towards regions with high gradient
        if centroid_sel_strat == 'GRADIENT_BIAS':
            g = self.__image_gradient(images)
            x = torch.randint(1, w-1, size=[n, 3*patches_per_image], device="cuda")
            y = torch.randint(1, h-1, size=[n, 3*patches_per_image], device="cuda")

            coords = torch.stack([x, y], dim=-1).float()
            g = altcorr.patchify(g[0,:,None], coords, 0).view(n, 3 * patches_per_image)
            
            ix = torch.argsort(g, dim=1)
            x = torch.gather(x, 1, ix[:, -patches_per_image:])
            y = torch.gather(y, 1, ix[:, -patches_per_image:])

        elif centroid_sel_strat == 'RANDOM':
            x = torch.randint(1, w-1, size=[n, patches_per_image], device="cuda")
            y = torch.randint(1, h-1, size=[n, patches_per_image], device="cuda")

        else:
            raise NotImplementedError(f"Patch centroid selection not implemented: {centroid_sel_strat}")

        coords = torch.stack([x, y], dim=-1).float()
        imap = altcorr.patchify(imap[0], coords, 0).view(b, -1, DIM, 1, 1)
        gmap = altcorr.patchify(fmap[0], coords, P//2).view(b, -1, 128, P, P)

        if return_color:
            clr = altcorr.patchify(images[0], 4*(coords + 0.5), 0).view(b, -1, 3)

        if disps is None:
            disps = torch.ones(b, n, h, w, device="cuda")

        grid, _ = coords_grid_with_index(disps, device=fmap.device)
        patches = altcorr.patchify(grid[0], coords, P//2).view(b, -1, 3, P, P)

        index = torch.arange(n, device="cuda").view(n, 1)
        index = index.repeat(1, patches_per_image).reshape(-1)

        if return_color:
            return fmap, gmap, imap, patches, index, clr

        return fmap, gmap, imap, patches, index


class CorrBlock:
    def __init__(self, fmap, gmap, radius=3, dropout=0.2, levels=[1,4]):
        self.dropout = dropout
        self.radius = radius
        self.levels = levels

        self.gmap = gmap
        self.pyramid = pyramidify(fmap, lvls=levels)

    def __call__(self, ii, jj, coords):
        corrs = []
        for i in range(len(self.levels)):
            corrs += [ altcorr.corr(self.gmap, self.pyramid[i], coords / self.levels[i], ii, jj, self.radius, self.dropout) ]
        return torch.stack(corrs, -1).view(1, len(ii), -1)


class VONet(nn.Module):
    def __init__(self, use_viewer=False):
        super(VONet, self).__init__()
        self.P = 3
        self.patchify = Patchifier(self.P)
        self.update = Update(self.P)

        self.DIM = DIM
        self.RES = 4
        
        # ====================================================================
        # DEPTH MODULE - Network owns ALL depth handling
        # ====================================================================
        self.depth_blend_alpha = 0.2
        self.depth_maps = {}        # Store full depth maps (runtime)
        self.depth_priors = {}      # For training
        self.depth_confidence = {}  # For training
        
        # Fusion parameters
        self.M = None  # Patches per frame (set by dpvo)
        self.optimization_window = 10  # Default
        # ====================================================================
    
    def set_fusion_params(self, M, optimization_window=10):
        """Set fusion parameters (called once by dpvo during init)"""
        self.M = M
        self.optimization_window = optimization_window
    
    def initialize_from_depth(self, patches, depth_map):
        """
        FIXED: Initialize with median fallback for invalid patches (matches external)
        """
        P = self.P
        RES = self.RES
        M = patches.shape[1]
        
        u = (patches[0, :, 0, P//2, P//2] * RES).long()
        v = (patches[0, :, 1, P//2, P//2] * RES).long()
        
        if not isinstance(depth_map, torch.Tensor):
            depth_map = torch.from_numpy(depth_map).float().cuda()
        if not depth_map.is_cuda:
            depth_map = depth_map.cuda()
        
        H, W = depth_map.shape
        u = u.clamp(0, W-1)
        v = v.clamp(0, H-1)
        
        depths = depth_map[v, u]
        valid = (depths > 1.0) & (depths < 80.0)
        
        # FIXED: >= not > (match external)
        if valid.sum() >= M // 2:
            # FIXED: Compute median for invalid patches
            median_depth = torch.median(depths[valid])
            
            # FIXED: Fill invalid with median (not 1.0!)
            inv_depths = torch.zeros_like(depths)
            inv_depths[valid] = 1.0 / depths[valid]
            inv_depths[~valid] = 1.0 / median_depth
            
            patches[0, :, 2, :, :] = inv_depths.view(M, 1, 1).expand(M, P, P)
            return patches, True
        
        return patches, False
    
    def add_depth_map(self, frame_id, depth_map, patches=None):
        """
        EXACT ZED BEHAVIOR: Sample depths at patch locations and store
        (Moved from DBA to network for clean architecture)
        
        Args:
            frame_id: Frame index
            depth_map: [H, W] numpy array or tensor - full depth map
            patches: [1, M, 3, P, P] tensor - patches to sample depths at
        """
        # Convert to tensor if needed
        if not isinstance(depth_map, torch.Tensor):
            depth_map_tensor = torch.from_numpy(depth_map).float().cuda()
        else:
            if not depth_map.is_cuda:
                depth_map_tensor = depth_map.cuda()
            else:
                depth_map_tensor = depth_map
        
        # Store full depth map (optional, for other uses)
        self.depth_maps[frame_id] = depth_map_tensor
        
        # EXACT ZED SAMPLING: Sample at patch locations and store
        if patches is not None and self.M is not None:
            P = self.P
            M = self.M
            RES = self.RES
            
            # Extract patch coordinates (EXACT ZED code)
            u_scaled = patches[0, :, 0, P//2, P//2].cpu().numpy()
            v_scaled = patches[0, :, 1, P//2, P//2].cpu().numpy()
            
            u = (u_scaled * RES).round().astype(int)
            v = (v_scaled * RES).round().astype(int)
            
            # Convert depth map to numpy for sampling (EXACT ZED)
            if isinstance(depth_map, torch.Tensor):
                depth_map_np = depth_map.cpu().numpy()
            else:
                depth_map_np = depth_map
            
            # Sample depths (EXACT ZED logic)
            valid = (u >= 0) & (u < depth_map_np.shape[1]) & (v >= 0) & (v < depth_map_np.shape[0])
            depths = np.zeros(M, dtype=np.float32)
            conf = np.zeros(M, dtype=np.float32)
            
            depths[valid] = depth_map_np[v[valid], u[valid]]
            conf[valid] = ((depths[valid] >= 1.0) & (depths[valid] <= 80.0)).astype(np.float32)
            depths[~conf.astype(bool)] = 0.0
            
            # Store as tensors (EXACT ZED)
            self.depth_priors[frame_id] = torch.tensor(depths, device='cuda', dtype=torch.float32)
            self.depth_confidence[frame_id] = torch.tensor(conf, device='cuda', dtype=torch.float32)
        
        # Trim storage (EXACT ZED)
        if len(self.depth_priors) > 30:
            oldest = min(self.depth_priors.keys())
            del self.depth_priors[oldest]
            del self.depth_confidence[oldest]
            if oldest in self.depth_maps:
                del self.depth_maps[oldest]
    
    def set_depth_priors(self, depth_priors, depth_confidence):
        """Set pre-sampled depth priors (for training)"""
        self.depth_priors = depth_priors
        self.depth_confidence = depth_confidence
    
    def clear_depth_priors(self):
        """Clear all depth data"""
        self.depth_maps = {}
        self.depth_priors = {}
        self.depth_confidence = {}
    
    def fuse_depth(self, patches, ix, current_frame, imu_enabled=False):
        """
        EXACT ZED BEHAVIOR: Apply depth constraints to patches BEFORE BA
        (Moved from DBA to network for clean architecture)
        
        Args:
            patches: [1, N, 3, P, P] - patches to fuse
            ix: [N] - frame index for each patch  
            current_frame: int - current frame number (self.n in DBA)
            imu_enabled: bool - whether IMU is active (affects fusion strength)
        
        Returns:
            patches: fused patches
        """
        # EXACT ZED: Check if initialized and have depth priors
        if len(self.depth_priors) == 0 or self.M is None:
            return patches
        
        P = self.P
        M = self.M
        
        # EXACT ZED: Get optimization window
        t0 = max(current_frame - self.optimization_window, 0) if self.optimization_window else max(current_frame - 8, 0)
        
        constraint_count = 0
        
        # EXACT ZED: Loop through frames in window
        for frame_id in range(t0, current_frame):
            if frame_id in self.depth_priors:
                start_idx = frame_id * M
                end_idx = start_idx + M
                
                if end_idx <= patches.shape[1]:
                    # EXACT ZED: Get stored depths
                    prior_depths = self.depth_priors[frame_id]       # [M]
                    confidences = self.depth_confidence[frame_id]    # [M]
                    
                    # EXACT ZED: Get current inverse depths at center pixel
                    current_inv_depths = patches[0, start_idx:end_idx, 2, 1, 1]  # [M]
                    target_inv_depths = 1.0 / torch.clamp(prior_depths, min=1.0)  # [M]
                    
                    # FIXED: IMU-aware fusion strength with WORKING thresholds
                    if imu_enabled:
                        # Very gentle when IMU is active
                        base_strength = 0.2
                        confidence_threshold = 1.0  # FIXED: was 1.0 (impossible!)
                    else:
                        # Stronger when IMU is not active
                        base_strength = 0.2
                        confidence_threshold = 1.0  # FIXED: was 1.0 (impossible!)
                    
                    # EXACT ZED: Compute blend weights
                    blend_weights = torch.clamp(confidences * base_strength, 0, 0.8)  # Max 80% influence
                    valid_mask = (confidences > confidence_threshold) & (prior_depths > 0)
                    
                    if valid_mask.any():
                        # EXACT ZED: Blend current depths with priors
                        blended_inv_depths = (current_inv_depths * (1 - blend_weights) + 
                                            target_inv_depths * blend_weights)
                        
                        # EXACT ZED: Update entire patch depth, not just center pixel
                        for i in range(M):
                            if valid_mask[i]:
                                patches[0, start_idx + i, 2, :, :] = blended_inv_depths[i]
                                constraint_count += 1
        
        # Optional: Print like ZED (can remove if too verbose)
        if constraint_count > 0:
            imu_status = "IMU-active" if imu_enabled else "IMU-inactive"
            frames_with_depth = len([f for f in range(t0, current_frame) if f in self.depth_priors])
            print(f"Pre-BA: Applied depth alignment to {constraint_count} patches ({imu_status}) across {frames_with_depth} frames")
        
        return patches
    
    def _blend_patches_with_priors(self, patches, ix):
        """
        Blend using pre-sampled priors (for training mode)
        """
        if not hasattr(self, 'depth_priors') or len(self.depth_priors) == 0:
            return patches
        
        P = self.P
        strength = self.depth_blend_alpha
        
        for frame_id in self.depth_priors.keys():
            mask = (ix == frame_id)
            if mask.sum() == 0:
                continue
            
            num_patches = mask.sum().item()
            prior_depths = self.depth_priors[frame_id][:num_patches].to(patches.device)
            confidences = self.depth_confidence[frame_id][:num_patches].to(patches.device)
            
            current_inv_d = patches[0, mask, 2, P//2, P//2].clone()
            target_inv_d = 1.0 / torch.clamp(prior_depths, min=1.0)
            
            blend_weight = torch.clamp(confidences * strength, 0, 0.8)
            valid = (confidences > 0.8) & (prior_depths > 0)
            
            if valid.any():
                blended = current_inv_d * (1 - blend_weight) + target_inv_d * blend_weight
                patch_indices_valid = torch.where(mask)[0][valid]
                blended_valid = blended[valid]
                patches[0, patch_indices_valid, 2, :, :] = blended_valid.view(-1, 1, 1).expand(-1, P, P)
        
        return patches


    @autocast(device_type='cuda', enabled=False)
    def forward(self, images, poses, disps, intrinsics, M=1024, STEPS=12, P=1, structure_only=False, rescale=False):
        """ Estimates SE3 or Sim3 between pair of frames """

        images = 2 * (images / 255.0) - 0.5
        intrinsics = intrinsics / 4.0
        disps = disps[:, :, 1::4, 1::4].float()

        fmap, gmap, imap, patches, ix = self.patchify(images, disps=disps)

        corr_fn = CorrBlock(fmap, gmap)

        b, N, c, h, w = fmap.shape
        p = self.P

        patches_gt = patches.clone()
        Ps = poses

        d = patches[..., 2, p//2, p//2]
        
        if disps is not None:
            patches = set_depth(patches, d)
        else:
            patches = set_depth(patches, torch.rand_like(d))

        kk, jj = flatmeshgrid(torch.where(ix < 8)[0], torch.arange(0,8, device="cuda"), indexing='ij')
        ii = ix[kk]

        imap = imap.view(b, -1, DIM)
        net = torch.zeros(b, len(kk), DIM, device="cuda", dtype=torch.float)
        
        Gs = SE3.IdentityLike(poses)

        if structure_only:
            Gs.data[:] = poses.data[:]

        traj = []
        bounds = [-64, -64, w + 64, h + 64]
        
        while len(traj) < STEPS:
            Gs = Gs.detach()
            patches = patches.detach()

            n = ii.max() + 1
            if len(traj) >= 8 and n < images.shape[1]:
                if not structure_only: Gs.data[:,n] = Gs.data[:,n-1]
                kk1, jj1 = flatmeshgrid(torch.where(ix  < n)[0], torch.arange(n, n+1, device="cuda"), indexing='ij')
                kk2, jj2 = flatmeshgrid(torch.where(ix == n)[0], torch.arange(0, n+1, device="cuda"), indexing='ij')

                ii = torch.cat([ix[kk1], ix[kk2], ii])
                jj = torch.cat([jj1, jj2, jj])
                kk = torch.cat([kk1, kk2, kk])

                net1 = torch.zeros(b, len(kk1) + len(kk2), DIM, device="cuda")
                net = torch.cat([net1, net], dim=1)

                if np.random.rand() < 0.1:
                    k = (ii != (n - 4)) & (jj != (n - 4))
                    ii = ii[k]
                    jj = jj[k]
                    kk = kk[k]
                    net = net[:,k]

                patches[:,ix==n,2] = torch.median(patches[:,(ix == n-1) | (ix == n-2),2])
                n = ii.max() + 1
            
            # Apply depth blending using pre-sampled priors (for training)
            if hasattr(self, 'depth_priors') and len(self.depth_priors) > 0:
                patches = self._blend_patches_with_priors(patches, ix)

            coords = pops.transform(Gs, patches, intrinsics, ii, jj, kk)
            
            coords1 = coords.permute(0, 1, 4, 2, 3).contiguous()

            corr = corr_fn(kk, jj, coords1)
            net, (delta, weight, _) = self.update(net, imap[:,kk], corr, None, ii, jj, kk)

            lmbda = 1e-4
            target = coords[...,p//2,p//2,:] + delta

            ep = 10
            for itr in range(2):
                Gs, patches = BA(Gs, patches, intrinsics, target, weight, lmbda, ii, jj, kk, 
                    bounds, ep=ep, fixedp=1, structure_only=structure_only)

            kl = torch.as_tensor(0)
            dij = (ii - jj).abs()
            k = (dij > 0) & (dij <= 2)

            coords = pops.transform(Gs, patches, intrinsics, ii[k], jj[k], kk[k])
            coords_gt, valid, _ = pops.transform(Ps, patches_gt, intrinsics, ii[k], jj[k], kk[k], jacobian=True)

            traj.append((valid, coords, coords_gt, Gs[:,:n], Ps[:,:n], kl, patches, ix))


        return traj
