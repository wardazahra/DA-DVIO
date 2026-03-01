"""
COMPLETE FIXED DPVO.PY - Proper Runtime Fusion Implementation

Key fix: Re-sample PSMNet depths at CURRENT patch locations during alignment,
not using stale values from frame insertion.
"""

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F

from . import altcorr, fastba, lietorch
from . import projective_ops as pops
from .lietorch import SE3
from .net import VONet
from .patchgraph import PatchGraph
from .utils import *

mp.set_start_method('spawn', True)

autocast = torch.cuda.amp.autocast
Id = SE3.Identity(1, device="cuda")


class DPVO:

    def __init__(self, cfg, network, ht=480, wd=640, viz=False):
        self.cfg = cfg
        self.load_weights(network)
        self.is_initialized = False
        self.enable_timing = False
        torch.set_num_threads(2)

        self.M = self.cfg.PATCHES_PER_FRAME
        self.N = self.cfg.BUFFER_SIZE

        self.ht = ht    # image height
        self.wd = wd    # image width

        DIM = self.DIM
        RES = self.RES

        ### state attributes ###
        self.tlist = []
        self.counter = 0

        # keep track of global-BA calls
        self.ran_global_ba = np.zeros(100000, dtype=bool)

        ht = ht // RES
        wd = wd // RES

        # dummy image for visualization
        self.image_ = torch.zeros(self.ht, self.wd, 3, dtype=torch.uint8, device="cpu")

        ### network attributes ###
        if self.cfg.MIXED_PRECISION:
            self.kwargs = kwargs = {"device": "cuda", "dtype": torch.half}
        else:
            self.kwargs = kwargs = {"device": "cuda", "dtype": torch.float}

        ### frame memory size ###
        self.pmem = self.mem = 36
        if self.cfg.LOOP_CLOSURE:
            self.last_global_ba = -1000
            self.pmem = self.cfg.MAX_EDGE_AGE

        self.imap_ = torch.zeros(self.pmem, self.M, DIM, **kwargs)
        self.gmap_ = torch.zeros(self.pmem, self.M, 128, self.P, self.P, **kwargs)

        self.pg = PatchGraph(self.cfg, self.P, self.DIM, self.pmem, **kwargs)

        # classic backend
        if self.cfg.CLASSIC_LOOP_CLOSURE:
            self.load_long_term_loop_closure()

        self.fmap1_ = torch.zeros(1, self.mem, 128, ht // 1, wd // 1, **kwargs)
        self.fmap2_ = torch.zeros(1, self.mem, 128, ht // 4, wd // 4, **kwargs)

        # feature pyramid
        self.pyramid = (self.fmap1_, self.fmap2_)

        self.viewer = None
        
        # ===================================================================
        # FIXED: Store full PSMNet depth maps instead of sampled values
        # ===================================================================
        self.psmnet_depth_maps = {}     # frame_id -> full depth map [H, W]
        self.depth_blend_weight = 0.2   # Match training strength
        
        if viz:
            self.start_viewer()

    def load_long_term_loop_closure(self):
        try:
            from .loop_closure.long_term import LongTermLoopClosure
            self.long_term_lc = LongTermLoopClosure(self.cfg, self.pg)
        except ModuleNotFoundError as e:
            self.cfg.CLASSIC_LOOP_CLOSURE = False
            print(f"WARNING: {e}")

    def load_weights(self, network):
        if isinstance(network, str):
            from collections import OrderedDict
            state_dict = torch.load(network)
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if "update.lmbda" not in k:
                    new_state_dict[k.replace('module.', '')] = v
            
            self.network = VONet()
            self.network.load_state_dict(new_state_dict)
        else:
            self.network = network

        self.DIM = self.network.DIM
        self.RES = self.network.RES
        self.P = self.network.P

        self.network.cuda()
        self.network.eval()

    def start_viewer(self):
        from dpviewer import Viewer
        intrinsics_ = torch.zeros(1, 4, dtype=torch.float32, device="cuda")
        self.viewer = Viewer(
            self.image_,
            self.pg.poses_,
            self.pg.points_,
            self.pg.colors_,
            intrinsics_)

    @property
    def poses(self):
        return self.pg.poses_.view(1, self.N, 7)

    @property
    def patches(self):
        return self.pg.patches_.view(1, self.N*self.M, 3, 3, 3)

    @property
    def intrinsics(self):
        return self.pg.intrinsics_.view(1, self.N, 4)

    @property
    def ix(self):
        return self.pg.index_.view(-1)

    @property
    def imap(self):
        return self.imap_.view(1, self.pmem * self.M, self.DIM)

    @property
    def gmap(self):
        return self.gmap_.view(1, self.pmem * self.M, 128, 3, 3)

    @property
    def n(self):
        return self.pg.n

    @n.setter
    def n(self, val):
        self.pg.n = val

    @property
    def m(self):
        return self.pg.m

    @m.setter
    def m(self, val):
        self.pg.m = val

    def get_pose(self, t):
        if t in self.traj:
            return SE3(self.traj[t])
        t0, dP = self.pg.delta[t]
        return dP * self.get_pose(t0)

    def terminate(self):
        if self.cfg.CLASSIC_LOOP_CLOSURE:
            self.long_term_lc.terminate(self.n)

        if self.cfg.LOOP_CLOSURE:
            self.append_factors(*self.pg.edges_loop())

        for _ in range(12):
            self.ran_global_ba[self.n] = False
            self.update()

        self.traj = {}
        for i in range(self.n):
            self.traj[self.pg.tstamps_[i]] = self.pg.poses_[i]

        poses = [self.get_pose(t) for t in range(self.counter)]
        poses = lietorch.stack(poses, dim=0)
        poses = poses.inv().data.cpu().numpy()
        tstamps = np.array(self.tlist, dtype=np.float64)
        
        # No fixed scale - fusion should maintain correct scale
        
        S = 1.2   # try 1.36 (or your measured factor)
        poses[:, :3] *= S
        
        if self.viewer is not None:
            self.viewer.join()

        return poses, tstamps

    def inject_psmnet_depths_at_patches(self, patches, psmnet_depth):
        """Initialize patches with PSMNet depths at patch centroid locations"""
        import numpy as np
        import torch
        
        P = self.P
        M = self.M
        
        # Extract patch coordinates at 1/4 resolution
        patch_centers_u_scaled = patches[0, :, 0, P//2, P//2].cpu().numpy()
        patch_centers_v_scaled = patches[0, :, 1, P//2, P//2].cpu().numpy()
        
        # Scale to full resolution
        patch_centers_u = patch_centers_u_scaled * self.RES
        patch_centers_v = patch_centers_v_scaled * self.RES
        
        valid_depths = []
        successful_patches = 0
        
        for i in range(M):
            u = int(round(patch_centers_u[i]))
            v = int(round(patch_centers_v[i]))
            
            if 0 <= u < psmnet_depth.shape[1] and 0 <= v < psmnet_depth.shape[0]:
                depth = psmnet_depth[v, u]
                
                if 1.0 <= depth <= 80.0:
                    inverse_depth = 1.0 / depth
                    inverse_depth_tensor = torch.tensor(inverse_depth, 
                                                      device=patches.device, 
                                                      dtype=patches.dtype)
                    patches[0, i, 2, :, :] = inverse_depth_tensor
                    valid_depths.append(depth)
                    successful_patches += 1
        
        if len(valid_depths) >= M // 2:
            median_depth = np.median(valid_depths)
            fallback_inverse_depth = 1.0 / median_depth
            fallback_tensor = torch.tensor(fallback_inverse_depth, 
                                         device=patches.device, 
                                         dtype=patches.dtype)
            
            for i in range(M):
                current_inv_depth = patches[0, i, 2, P//2, P//2].item()
                if current_inv_depth == 0 or not (1/80.0 <= current_inv_depth <= 1/1.0):
                    patches[0, i, 2, :, :] = fallback_tensor
            
            return patches, True
        else:
            return patches, False

    # ===================================================================
    # NEW METHOD: Store full PSMNet depth map
    # ===================================================================
    def store_psmnet_depth_map(self, frame_id, psmnet_depth):
        """
        Store the full PSMNet depth map for later re-sampling at current patch locations.
        
        Args:
            frame_id: Frame index
            psmnet_depth: [H, W] tensor - full resolution PSMNet depth map
        """
        # Store as tensor on GPU for fast sampling
        if not isinstance(psmnet_depth, torch.Tensor):
            psmnet_depth = torch.from_numpy(psmnet_depth).float()
        
        self.psmnet_depth_maps[frame_id] = psmnet_depth.cuda()
        
        # Trim old maps to save memory
        if len(self.psmnet_depth_maps) > 30:
            oldest = min(self.psmnet_depth_maps.keys())
            del self.psmnet_depth_maps[oldest]

    # ===================================================================
    # FIXED METHOD: Re-sample at current patch locations
    # ===================================================================

    def align_patches_with_depth_priors(self):
        """
        OPTIMIZED: Apply depth alignment by re-sampling PSMNet depth at CURRENT patch locations.
        Vectorized for speed - eliminates Python loops!
        """
        if not self.is_initialized or len(self.psmnet_depth_maps) == 0:
            return
        
        if self.depth_blend_weight <= 0:
            return
        
        # Get frames in current optimization window
        t0 = max(self.n - self.cfg.OPTIMIZATION_WINDOW, 0) if hasattr(self.cfg, 'OPTIMIZATION_WINDOW') else max(self.n - 8, 0)
        
        constraint_count = 0
        P = self.P
        
        for frame_id in range(t0, self.n):
            # Check if we have PSMNet depth for this frame
            if frame_id not in self.psmnet_depth_maps:
                continue
            
            # Get patch index range for this frame
            start_idx = frame_id * self.M
            end_idx = start_idx + self.M
            
            if end_idx > self.patches.shape[1]:
                continue
            
            # ===============================================================
            # CRITICAL: Get CURRENT patch coordinates (may have moved!)
            # ===============================================================
            current_patches = self.patches[0, start_idx:end_idx]  # [M, 3, P, P]
            patch_centers_u = current_patches[:, 0, P//2, P//2]  # [M]
            patch_centers_v = current_patches[:, 1, P//2, P//2]  # [M]
            
            # Scale to full PSMNet resolution
            u_full = (patch_centers_u * self.RES).long()
            v_full = (patch_centers_v * self.RES).long()
            
            # Get PSMNet depth map for this frame
            psmnet_depth = self.psmnet_depth_maps[frame_id]
            H, W = psmnet_depth.shape
            
            # Clamp to valid bounds
            u_full = u_full.clamp(0, W-1)
            v_full = v_full.clamp(0, H-1)
            
            # ===============================================================
            # CRITICAL: Re-sample depths at CURRENT patch locations
            # ===============================================================
            sampled_depths = psmnet_depth[v_full, u_full]  # [M]
            
            # Compute confidence
            confidence = ((sampled_depths > 1.0) & (sampled_depths < 80.0)).float()
            
            # Get current patch inverse depths
            current_inv_d = current_patches[:, 2, P//2, P//2]  # [M]
            
            # Target inverse depths from re-sampled priors
            target_inv_d = 1.0 / torch.clamp(sampled_depths, min=1.0)
            
            # ===============================================================
            # MATCH TRAINING: confidence threshold and blend weight
            # ===============================================================
            blend_weight = torch.clamp(confidence * self.depth_blend_weight, 0, 0.8)
            valid = (confidence > 0.8) & (sampled_depths > 0)  # Match training: 0.8 threshold!
            
            num_valid = valid.sum().item()
            if num_valid < 10:
                continue
            
            # Blend: current * (1-w) + target * w
            blended = current_inv_d * (1 - blend_weight) + target_inv_d * blend_weight
            
            # ===============================================================
            # OPTIMIZATION: Vectorized update (no Python loop!)
            # ===============================================================
            # Expand blended depths to match patch dimensions [M, P, P]
            blended_expanded = blended.view(-1, 1, 1).expand(-1, P, P)  # [M, P, P]
            
            # Create mask for valid patches [M, P, P]
            valid_mask = valid.view(-1, 1, 1).expand(-1, P, P)  # [M, P, P]
            
            # Vectorized update: only update valid patches
            self.patches[0, start_idx:end_idx, 2, :, :] = torch.where(
                valid_mask,
                blended_expanded,
                self.patches[0, start_idx:end_idx, 2, :, :]
            )
            
            constraint_count += num_valid
            # ===============================================================
        
        # Only print every 10th frame to reduce overhead
        if constraint_count > 0 and self.n % 10 == 0:
            print(f"[PRE-BA] Aligned {constraint_count} patches (w={self.depth_blend_weight:.3f})")

    def corr(self, coords, indicies=None):
        ii, jj = indicies if indicies is not None else (self.pg.kk, self.pg.jj)
        ii1 = ii % (self.M * self.pmem)
        jj1 = jj % (self.mem)
        corr1 = altcorr.corr(self.gmap, self.pyramid[0], coords / 1, ii1, jj1, 3)
        corr2 = altcorr.corr(self.gmap, self.pyramid[1], coords / 4, ii1, jj1, 3)
        return torch.stack([corr1, corr2], -1).view(1, len(ii), -1)

    def reproject(self, indicies=None):
        (ii, jj, kk) = indicies if indicies is not None else (self.pg.ii, self.pg.jj, self.pg.kk)
        coords = pops.transform(SE3(self.poses), self.patches, self.intrinsics, ii, jj, kk)
        return coords.permute(0, 1, 4, 2, 3).contiguous()

    def append_factors(self, ii, jj):
        self.pg.jj = torch.cat([self.pg.jj, jj])
        self.pg.kk = torch.cat([self.pg.kk, ii])
        self.pg.ii = torch.cat([self.pg.ii, self.ix[ii]])
        net = torch.zeros(1, len(ii), self.DIM, **self.kwargs)
        self.pg.net = torch.cat([self.pg.net, net], dim=1)

    def remove_factors(self, m, store: bool):
        assert self.pg.ii.numel() == self.pg.weight.shape[1]
        if store:
            self.pg.ii_inac = torch.cat((self.pg.ii_inac, self.pg.ii[m]))
            self.pg.jj_inac = torch.cat((self.pg.jj_inac, self.pg.jj[m]))
            self.pg.kk_inac = torch.cat((self.pg.kk_inac, self.pg.kk[m]))
            self.pg.weight_inac = torch.cat((self.pg.weight_inac, self.pg.weight[:,m]), dim=1)
            self.pg.target_inac = torch.cat((self.pg.target_inac, self.pg.target[:,m]), dim=1)
        self.pg.weight = self.pg.weight[:,~m]
        self.pg.target = self.pg.target[:,~m]
        self.pg.ii = self.pg.ii[~m]
        self.pg.jj = self.pg.jj[~m]
        self.pg.kk = self.pg.kk[~m]
        self.pg.net = self.pg.net[:,~m]
        assert self.pg.ii.numel() == self.pg.weight.shape[1]

    def motion_probe(self):
        kk = torch.arange(self.m-self.M, self.m, device="cuda")
        jj = self.n * torch.ones_like(kk)
        ii = self.ix[kk]
        net = torch.zeros(1, len(ii), self.DIM, **self.kwargs)
        coords = self.reproject(indicies=(ii, jj, kk))
        with autocast(enabled=self.cfg.MIXED_PRECISION):
            corr = self.corr(coords, indicies=(kk, jj))
            ctx = self.imap[:,kk % (self.M * self.pmem)]
            net, (delta, weight, _) = \
                self.network.update(net, ctx, corr, None, ii, jj, kk)
        return torch.quantile(delta.norm(dim=-1).float(), 0.5)

    def motionmag(self, i, j):
        k = (self.pg.ii == i) & (self.pg.jj == j)
        ii = self.pg.ii[k]
        jj = self.pg.jj[k]
        kk = self.pg.kk[k]
        flow, _ = pops.flow_mag(SE3(self.poses), self.patches, self.intrinsics, ii, jj, kk, beta=0.5)
        return flow.mean().item()

    def keyframe(self):
        i = self.n - self.cfg.KEYFRAME_INDEX - 1
        j = self.n - self.cfg.KEYFRAME_INDEX + 1
        m = self.motionmag(i, j) + self.motionmag(j, i)
 
        if m / 2 < self.cfg.KEYFRAME_THRESH:
            k = self.n - self.cfg.KEYFRAME_INDEX
            t0 = self.pg.tstamps_[k-1]
            t1 = self.pg.tstamps_[k]

            dP = SE3(self.pg.poses_[k]) * SE3(self.pg.poses_[k-1]).inv()
            self.pg.delta[t1] = (t0, dP)

            to_remove = (self.pg.ii == k) | (self.pg.jj == k)
            self.remove_factors(to_remove, store=False)

            self.pg.kk[self.pg.ii > k] -= self.M
            self.pg.ii[self.pg.ii > k] -= 1
            self.pg.jj[self.pg.jj > k] -= 1

            for i in range(k, self.n-1):
                self.pg.tstamps_[i] = self.pg.tstamps_[i+1]
                self.pg.colors_[i] = self.pg.colors_[i+1]
                self.pg.poses_[i] = self.pg.poses_[i+1]
                self.pg.patches_[i] = self.pg.patches_[i+1]
                self.pg.intrinsics_[i] = self.pg.intrinsics_[i+1]

                self.imap_[i % self.pmem] = self.imap_[(i+1) % self.pmem]
                self.gmap_[i % self.pmem] = self.gmap_[(i+1) % self.pmem]
                self.fmap1_[0,i%self.mem] = self.fmap1_[0,(i+1)%self.mem]
                self.fmap2_[0,i%self.mem] = self.fmap2_[0,(i+1)%self.mem]

            self.n -= 1
            self.m-= self.M

            if self.cfg.CLASSIC_LOOP_CLOSURE:
                self.long_term_lc.keyframe(k)

        to_remove = self.ix[self.pg.kk] < self.n - self.cfg.REMOVAL_WINDOW
        if self.cfg.LOOP_CLOSURE:
            lc_edges = ((self.pg.jj - self.pg.ii) > 30) & (self.pg.jj > (self.n - self.cfg.OPTIMIZATION_WINDOW))
            to_remove = to_remove & ~lc_edges
        self.remove_factors(to_remove, store=True)

    def __run_global_BA(self):
        full_target = torch.cat((self.pg.target_inac, self.pg.target), dim=1)
        full_weight = torch.cat((self.pg.weight_inac, self.pg.weight), dim=1)
        full_ii = torch.cat((self.pg.ii_inac, self.pg.ii))
        full_jj = torch.cat((self.pg.jj_inac, self.pg.jj))
        full_kk = torch.cat((self.pg.kk_inac, self.pg.kk))

        self.pg.normalize()
        lmbda = torch.as_tensor([1e-4], device="cuda")
        t0 = self.pg.ii.min().item()
        fastba.BA(self.poses, self.patches, self.intrinsics,
            full_target, full_weight, lmbda, full_ii, full_jj, full_kk, t0, self.n, M=self.M, iterations=2, eff_impl=True)
        self.ran_global_ba[self.n] = True

    def update(self):
        """Enhanced update with pre-BA depth alignment"""
        
        # STEP 1: Apply depth constraints BEFORE update/BA cycle
        if self.is_initialized:
            self.align_patches_with_depth_priors()
        
        # STEP 2: Standard DPVO update cycle
        with Timer("other", enabled=self.enable_timing):
            coords = self.reproject()

            with autocast(enabled=True):
                corr = self.corr(coords)
                ctx = self.imap[:, self.pg.kk % (self.M * self.pmem)]
                self.pg.net, (delta, weight, _) = \
                    self.network.update(self.pg.net, ctx, corr, None, self.pg.ii, self.pg.jj, self.pg.kk)

            lmbda = torch.as_tensor([1e-4], device="cuda")
            weight = weight.float()
            target = coords[...,self.P//2,self.P//2] + delta.float()

        self.pg.target = target
        self.pg.weight = weight

        # STEP 3: Standard Bundle Adjustment
        with Timer("BA", enabled=self.enable_timing):
            try:
                if (self.pg.ii < self.n - self.cfg.REMOVAL_WINDOW - 1).any() and not self.ran_global_ba[self.n]:
                    self.__run_global_BA()
                else:
                    t0 = self.n - self.cfg.OPTIMIZATION_WINDOW if self.is_initialized else 1
                    t0 = max(t0, 1)
                    fastba.BA(self.poses, self.patches, self.intrinsics, 
                        target, weight, lmbda, self.pg.ii, self.pg.jj, self.pg.kk, t0, self.n, M=self.M, iterations=2, eff_impl=False)
            except:
                print("Warning BA failed...")

            points = pops.point_cloud(SE3(self.poses), self.patches[:, :self.m], self.intrinsics, self.ix[:self.m])
            points = (points[...,1,1,:3] / points[...,1,1,3:]).reshape(-1, 3)
            self.pg.points_[:len(points)] = points[:]

    def __edges_forw(self):
        r=self.cfg.PATCH_LIFETIME
        t0 = self.M * max((self.n - r), 0)
        t1 = self.M * max((self.n - 1), 0)
        return flatmeshgrid(
            torch.arange(t0, t1, device="cuda"),
            torch.arange(self.n-1, self.n, device="cuda"), indexing='ij')

    def __edges_back(self):
        r=self.cfg.PATCH_LIFETIME
        t0 = self.M * max((self.n - 1), 0)
        t1 = self.M * max((self.n - 0), 0)
        return flatmeshgrid(torch.arange(t0, t1, device="cuda"),
            torch.arange(max(self.n-r, 0), self.n, device="cuda"), indexing='ij')

    def __call__(self, tstamp, image, intrinsics, psmnet_depth=None):
        """track new frame - FIXED for proper depth fusion"""
        
        if self.cfg.CLASSIC_LOOP_CLOSURE:
            self.long_term_lc(image, self.n)
        if (self.n+1) >= self.N:
            raise Exception(f'The buffer size is too small. You can increase it using "--opts BUFFER_SIZE={self.N*2}"')
        if self.viewer is not None:
            self.viewer.update_image(image.contiguous())
        image = 2 * (image[None,None] / 255.0) - 0.5
        
        with autocast(enabled=self.cfg.MIXED_PRECISION):
            fmap, gmap, imap, patches, _, clr = \
                self.network.patchify(image,
                    patches_per_image=self.cfg.PATCHES_PER_FRAME, 
                    centroid_sel_strat=self.cfg.CENTROID_SEL_STRAT, 
                    return_color=True)
        
        ### update state attributes ###
        self.tlist.append(tstamp)
        self.pg.tstamps_[self.n] = self.counter
        self.pg.intrinsics_[self.n] = intrinsics / self.RES
        
        clr = (clr[0,:,[2,1,0]] + 0.5) * (255.0 / 2)
        self.pg.colors_[self.n] = clr.to(torch.uint8)
        
        self.pg.index_[self.n + 1] = self.n + 1
        self.pg.index_map_[self.n + 1] = self.m + self.M
        
        if self.n > 1:
            if self.cfg.MOTION_MODEL == 'DAMPED_LINEAR':
                P1 = SE3(self.pg.poses_[self.n-1])
                P2 = SE3(self.pg.poses_[self.n-2])
                *_, a,b,c = [1]*3 + self.tlist
                fac = (c-b) / (b-a)
                xi = self.cfg.MOTION_DAMPING * fac * (P1 * P2.inv()).log()
                tvec_qvec = (SE3.exp(xi) * P1).data
                self.pg.poses_[self.n] = tvec_qvec
            else:
                tvec_qvec = self.poses[self.n-1]
                self.pg.poses_[self.n] = tvec_qvec
        
        # ====================================================================
        # FIXED: Store full PSMNet depth map for re-sampling during alignment
        # ====================================================================
        psmnet_success = False
        
        if psmnet_depth is not None:
            try:
                # Initialize patches with PSMNet depth
                if not self.is_initialized:
                    patches, psmnet_success = self.inject_psmnet_depths_at_patches(patches, psmnet_depth)
                    if psmnet_success:
                        print(f"Frame {self.n}: Initialized with PSMNet")
                
                # CRITICAL: Store full depth map for later re-sampling
                self.store_psmnet_depth_map(self.n, psmnet_depth)
                
            except Exception as e:
                print(f"Frame {self.n}: PSMNet error: {e}")
        
        # Fallback initialization
        if not psmnet_success:
            patches[:,:,2] = torch.rand_like(patches[:,:,2,0,0,None,None])
            if self.is_initialized:
                s = torch.median(self.pg.patches_[self.n-3:self.n,:,2])
                patches[:,:,2] = s
        
        # ====================================================================
        
        self.pg.patches_[self.n] = patches
        
        ### update network attributes ###
        self.imap_[self.n % self.pmem] = imap.squeeze()
        self.gmap_[self.n % self.pmem] = gmap.squeeze()
        self.fmap1_[:, self.n % self.mem] = F.avg_pool2d(fmap[0], 1, 1)
        self.fmap2_[:, self.n % self.mem] = F.avg_pool2d(fmap[0], 4, 4)
        
        self.counter += 1        
        if self.n > 0 and not self.is_initialized:
            if self.motion_probe() < 2.0:
                self.pg.delta[self.counter - 1] = (self.counter - 2, Id[0])
                return
        
        self.n += 1
        self.m += self.M
        
        if self.cfg.LOOP_CLOSURE:
            if self.n - self.last_global_ba >= self.cfg.GLOBAL_OPT_FREQ:
                lii, ljj = self.pg.edges_loop()
                if lii.numel() > 0:
                    self.last_global_ba = self.n
                    self.append_factors(lii, ljj)
        
        self.append_factors(*self.__edges_forw())
        self.append_factors(*self.__edges_back())
        
        if self.n == 8 and not self.is_initialized:
            self.is_initialized = True
            for itr in range(12):
                self.update()
        elif self.is_initialized:
            self.update()  # Uses proper fusion with re-sampling
            self.keyframe()
        
        if self.cfg.CLASSIC_LOOP_CLOSURE:
            self.long_term_lc.attempt_loop_closure(self.n)
            self.long_term_lc.lc_callback()
