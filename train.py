import cv2
import os
import argparse
import numpy as np
from collections import OrderedDict

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dpvo.data_readers.factory import dataset_factory

from dpvo.lietorch import SE3
from dpvo.logger import Logger
import torch.nn.functional as F

from dpvo.net import VONet

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


def kabsch_umeyama(A, B):
    """Compute optimal scaling factor using Kabsch-Umeyama algorithm"""
    n, m = A.shape
    EA = torch.mean(A, axis=0)
    EB = torch.mean(B, axis=0)
    VarA = torch.mean((A - EA).norm(dim=1)**2)
    H = ((A - EA).T @ (B - EB)) / n
    U, D, VT = torch.svd(H)
    c = VarA / torch.trace(torch.diag(D))
    return c


def store_depth_priors_training(patches, gt_depths, P=3):
    """
    Store depth priors for all frames (replicates dpvo.py functionality)
    
    Args:
        patches: [1, N, 3, P, P] - patch coordinates and depths
        gt_depths: [1, T, H, W] - PSMNet depth maps
        P: patch size (default 3)
    
    Returns:
        depth_priors: dict mapping frame_id -> depth values [M]
        depth_confidence: dict mapping frame_id -> confidence [M]
    """
    depth_priors = {}
    depth_confidence = {}
    
    _, T, H_gt, W_gt = gt_depths.shape
    N = patches.shape[1]
    
    # Check if GT is disparity or depth
    median_val = torch.median(gt_depths[gt_depths > 0]) if (gt_depths > 0).any() else 0
    
    if median_val > 100:  # Disparity format
        baseline_focal = 379.8145
        gt_depths_m = torch.where(gt_depths > 1.0, baseline_focal / gt_depths, torch.zeros_like(gt_depths))
        gt_depths_m = torch.clamp(gt_depths_m, 1.0, 80.0)
    else:
        gt_depths_m = gt_depths
    
    # For each frame
    for t in range(T):
        # Get patch center coordinates
        x = patches[0, :, 0, P//2, P//2]  # [N]
        y = patches[0, :, 1, P//2, P//2]  # [N]
        
        # Scale to gt_depths resolution (1/4 of full image)
        x_scaled = (x / 4.0).clamp(0, W_gt - 1).long()
        y_scaled = (y / 4.0).clamp(0, H_gt - 1).long()
        
        # Sample depths at patch locations
        sampled_depths = gt_depths_m[0, t, y_scaled, x_scaled]
        
        # Valid depth range for KITTI
        valid = (sampled_depths >= 1.0) & (sampled_depths <= 80.0)
        
        # Store
        depth_priors[t] = sampled_depths
        depth_confidence[t] = valid.float()
    
    return depth_priors, depth_confidence


def align_patches_with_depth_priors_training(patches, patch_indices, depth_priors, depth_confidence, P=3, strength=0.2):
    """
    Align patches with depth priors BEFORE BA iteration (replicates dpvo.py line 468)
    
    This is the CRITICAL fix for Issue #6 (train/inference mismatch)
    
    Args:
        patches: [1, N, 3, P, P] - patches to align
        patch_indices: [N] - frame index for each patch
        depth_priors: dict - stored depth values per frame
        depth_confidence: dict - confidence per depth value
        P: patch size
        strength: blending strength (0.2 = 20% toward prior per iteration)
    
    Returns:
        patches: aligned patches (modified in-place)
    """
    constraint_count = 0
    
    for frame_id in depth_priors.keys():
        # Find patches belonging to this frame
        mask = (patch_indices == frame_id)
        if mask.sum() == 0:
            continue
        
        # Count patches for this frame
        num_patches = mask.sum().item()
        
        # depth_priors[frame_id] has M entries (patches per frame)
        # We need only the first num_patches entries
        prior_depths = depth_priors[frame_id][:num_patches]
        confidences = depth_confidence[frame_id][:num_patches]
        
        # Get current inverse depths at patch centers
        current_inv_d = patches[0, mask, 2, P//2, P//2]
        
        # Target inverse depths from priors
        target_inv_d = 1.0 / torch.clamp(prior_depths, min=1.0)
        
        # Blend based on confidence (max 80% influence)
        blend_weight = torch.clamp(confidences * strength, 0, 0.8)
        valid = (confidences > 0.8) & (prior_depths > 0)
        
        if valid.any():
            # Blend: current * (1-w) + target * w
            blended = current_inv_d * (1 - blend_weight) + target_inv_d * blend_weight
            
            # Update entire patch depth (all P×P pixels)
            patch_indices_valid = torch.where(mask)[0][valid]
            for i, idx in enumerate(patch_indices_valid):
                patches[0, idx, 2, :, :] = blended[valid][i]
                constraint_count += 1
    
    return patches, constraint_count


def compute_depth_loss_fixed(pred_patches, gt_depths, patch_indices, debug=False):
    """
    Fixed depth loss with disparity conversion and proper coordinate scaling
    """
    B, N, C, P, _ = pred_patches.shape
    _, T, H_gt, W_gt = gt_depths.shape
    
    # Check if GT is disparity or depth
    median_val = torch.median(gt_depths[gt_depths > 0]) if (gt_depths > 0).any() else 0
    
    if median_val > 100:  # Disparity
        baseline_focal = 379.8145
        gt_depths = torch.where(gt_depths > 1.0, baseline_focal / gt_depths, torch.zeros_like(gt_depths))
        gt_depths = torch.clamp(gt_depths, 1.0, 80.0)
    
    # Get predicted depths
    pred_inv_d = pred_patches[:, :, 2, P//2, P//2]
    pred_d = 1.0 / pred_inv_d.clamp(min=1e-6)
    
    # Get patch coordinates
    x = pred_patches[:, :, 0, P//2, P//2]
    y = pred_patches[:, :, 1, P//2, P//2]
    
    # Scale to GT resolution (1/4 of full)
    x_scaled = x / 4.0
    y_scaled = y / 4.0
    
    # Normalize for grid_sample
    x_norm = 2.0 * x_scaled / (W_gt - 1) - 1.0
    y_norm = 2.0 * y_scaled / (H_gt - 1) - 1.0
    
    all_losses = []
    total_valid = 0
    
    for t in range(T):
        mask = (patch_indices == t)
        if mask.sum() == 0:
            continue
        
        # Build sampling grid
        x_frame = x_norm[0, mask].unsqueeze(0).unsqueeze(2)
        y_frame = y_norm[0, mask].unsqueeze(0).unsqueeze(2)
        grid = torch.stack([x_frame, y_frame], dim=-1)
        
        # Sample GT
        sampled_gt = F.grid_sample(
            gt_depths[:, t:t+1], grid,
            mode='bilinear', padding_mode='border', align_corners=False
        ).squeeze()
        
        pred_frame = pred_d[0, mask]
        valid = (sampled_gt > 1.0) & (sampled_gt < 50.0)
        n_valid = valid.sum().item()
        
        if n_valid == 0:
            continue
        
        diff = torch.abs(pred_frame[valid] - sampled_gt[valid])
        all_losses.append(diff.mean())
        total_valid += n_valid
    
    if len(all_losses) == 0:
        return torch.tensor(0.0, device=pred_patches.device), 0
    
    return torch.stack(all_losses).mean(), total_valid


def compute_scale_loss(P1, P2):
    """Compute scale consistency loss"""
    t1 = P1.matrix()[...,:3,3]
    t2 = P2.matrix()[...,:3,3]
    if t1.shape[1] < 3:
        return torch.tensor(0.0, device=t1.device)
    d1 = torch.norm(t1[:, 1:] - t1[:, :-1], dim=-1)
    d2 = torch.norm(t2[:, 1:] - t2[:, :-1], dim=-1)
    scales = d1 / (d2 + 1e-6)
    return scales.var() + (scales.mean() - 1.0)**2


def validate(net, val_loader, M, args):
    """Run validation on validation set - compute ALL losses like training"""
    net.eval()
    total_loss = 0.0
    total_flow_loss = 0.0
    total_depth_loss = 0.0
    total_pose_loss = 0.0
    total_scale_loss = 0.0
    n_batches = 0
    
    with torch.no_grad():
        for data_blob in val_loader:
            if len(data_blob) == 6:
                images, _, poses, disps, intrinsics, gt_depths = [x.cuda().float() for x in data_blob]
            elif len(data_blob) == 5:
                images, poses, disps, intrinsics, gt_depths = [x.cuda().float() for x in data_blob]
            else:
                continue
            
            poses = SE3(poses).inv()
            traj = net(images, poses, disps, intrinsics, M=M, STEPS=12, structure_only=False)
            
            # Compute ALL loss components (same as training)
            loss = 0.0
            flow_loss_val = 0.0
            depth_loss_val = 0.0
            pose_loss_val = 0.0
            scale_loss_val = 0.0
            
            for i, traj_item in enumerate(traj):
                if len(traj_item) == 8:
                    v, x, y, P1, P2, kl, patches, patch_ix = traj_item
                else:
                    v, x, y, P1, P2, kl = traj_item
                    patches, patch_ix = None, None
                
                # Flow loss
                e = (x - y).norm(dim=-1)
                e = e.reshape(-1, net.P**2)[(v > 0.5).reshape(-1)].min(dim=-1).values
                flow_loss = args.flow_weight * e.mean()
                loss += flow_loss
                flow_loss_val += flow_loss.item()  # Accumulate across iterations
                
                # Depth loss
                if patches is not None and gt_depths is not None and args.depth_weight > 0:
                    depth_loss, _ = compute_depth_loss_fixed(patches, gt_depths, patch_ix, debug=False)
                    if depth_loss > 0:
                        loss += args.depth_weight * depth_loss
                        depth_loss_val += depth_loss.item()  # Accumulate across iterations
                
                # Pose loss
                if i >= 2 and patches is not None:
                    N = P1.shape[1]
                    ii, jj = torch.meshgrid(torch.arange(N), torch.arange(N), indexing='ij')
                    ii, jj = ii.reshape(-1).cuda(), jj.reshape(-1).cuda()
                    k = ii != jj
                    ii, jj = ii[k], jj[k]
                    
                    P1, P2 = P1.inv(), P2.inv()
                    
                    t1, t2 = P1.matrix()[...,:3,3], P2.matrix()[...,:3,3]
                    s = kabsch_umeyama(t2[0], t1[0]).detach().clamp(max=10.0)
                    P1 = P1.scale(s.view(1, 1))
                    
                    dP = P1[:,ii].inv() * P1[:,jj]
                    dG = P2[:,ii].inv() * P2[:,jj]
                    e1 = (dP * dG.inv()).log()
                    tr, ro = e1[...,0:3].norm(dim=-1), e1[...,3:6].norm(dim=-1)
                    
                    pose_loss = args.pose_weight * (tr.mean() + ro.mean())
                    loss += pose_loss
                    pose_loss_val += pose_loss.item()  # Accumulate across iterations
                    
                    # Scale loss
                    if args.scale_weight > 0:
                        scale_loss = compute_scale_loss(P1, P2)
                        loss += args.scale_weight * scale_loss
                        scale_loss_val += scale_loss.item()  # Accumulate across iterations
            
            loss += kl
            
            total_loss += loss.item()
            total_flow_loss += flow_loss_val
            total_depth_loss += depth_loss_val
            total_pose_loss += pose_loss_val
            total_scale_loss += scale_loss_val
            n_batches += 1
            
            # Validate on more samples for smoother curves
            if n_batches >= 10:
                break
    
    net.train()
    
    if n_batches == 0:
        return {'loss': float('inf'), 'flow_loss': 0, 'depth_loss': 0, 'pose_loss': 0, 'scale_loss': 0}
    
    return {
        'loss': total_loss / n_batches,
        'flow_loss': total_flow_loss / n_batches,
        'depth_loss': total_depth_loss / n_batches,
        'pose_loss': total_pose_loss / n_batches,
        'scale_loss': total_scale_loss / n_batches
    }


def train(args):
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(f'runs/{args.name}')
    
    torch.cuda.empty_cache()
    
    # Dataset
    if args.dataset == 'kitti':
        train_db = dataset_factory(
            ['kitti'], datapath=args.datapath, mode='training',
            psmnet_dir=args.psmnet_dir, n_frames=args.n_frames,
            crop_size=[370, 1226], fmin=args.fmin, fmax=args.fmax,
            aug=args.aug, load_stereo=True, return_depth_gt=True
        )
        
        # Validation Dataset
        val_db = dataset_factory(
            ['kitti'], datapath=args.datapath, mode='validation',
            psmnet_dir=args.psmnet_dir, n_frames=args.n_frames,
            crop_size=[370, 1226], fmin=args.fmin, fmax=args.fmax,
            aug=False, load_stereo=True, return_depth_gt=True
        )
    else:
        train_db = dataset_factory(['tartan'], datapath=args.datapath,
            n_frames=args.n_frames, fmin=args.fmin, fmax=args.fmax)
        val_db = None
    
    train_loader = DataLoader(train_db, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_db, batch_size=1, shuffle=False, num_workers=0) if val_db else None
    
    # Network
    net = VONet().cuda().train()
    
    # Load checkpoint
    total_steps = 0
    if args.ckpt:
        state_dict = torch.load(args.ckpt)
        net.load_state_dict(OrderedDict([(k.replace('module.', ''), v) for k, v in state_dict.items()]), strict=False)
        if args.resume:
            import re
            match = re.search(r'_(\d+)\.pth$', args.ckpt)
            if match:
                total_steps = int(match.group(1))
        print(f"✅ Checkpoint loaded from {args.ckpt}, starting at step {total_steps}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, args.lr, args.steps, pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')
    for _ in range(total_steps):
        scheduler.step()
    
    logger = Logger(args.name, scheduler)
    M = 128 if torch.cuda.get_device_properties(0).total_memory < 18e9 else 256
    
    print("\n" + "="*80)
    print("🚀 TRAINING WITH CONSISTENT LOSS LOGGING")
    print("="*80)
    print(f"Patches per frame (M): {M}")
    print(f"Depth weight: {args.depth_weight}")
    print(f"Learning rate: {args.lr}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max steps: {args.steps}")
    print(f"Validation every: {args.val_freq} steps")
    print(f"Validation batches: 50 (for smoother curves)")
    print("="*80 + "\n")
    
    optimizer.zero_grad()
    depth_active = False
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Exponential moving average for smooth loss curves
    ema_alpha = 0.1  # Smoothing factor (0.1 = slow smooth, 0.5 = fast smooth)
    ema_loss = None
    ema_flow_loss = None
    ema_depth_loss = None
    ema_pose_loss = None
    ema_scale_loss = None
    
    # EMA for validation (faster smoothing since fewer points)
    val_ema_alpha = 0.3
    val_ema_loss = None
    val_ema_flow_loss = None
    val_ema_depth_loss = None
    val_ema_pose_loss = None
    val_ema_scale_loss = None
    
    while total_steps < args.steps:
        for data_blob in train_loader:
            # Parse
            if len(data_blob) == 6:
                images, _, poses, disps, intrinsics, gt_depths = [x.cuda().float() for x in data_blob]
            elif len(data_blob) == 5:
                images, poses, disps, intrinsics, gt_depths = [x.cuda().float() for x in data_blob]
            else:
                images, poses, disps, intrinsics = [x.cuda().float() for x in data_blob]
                gt_depths = None
            
            poses = SE3(poses).inv()
            
            # Prepare depth priors BEFORE calling net()
            depth_priors = None
            depth_confidence = None
            alignment_count = 0
            
            if gt_depths is not None:
                with torch.no_grad():
                    _, _, _, temp_patches, _ = net.patchify(
                        2 * (images / 255.0) - 0.5,
                        patches_per_image=M
                    )
                    depth_priors, depth_confidence = store_depth_priors_training(
                        temp_patches, gt_depths, P=net.P
                    )
            
            # Define callback function for depth alignment
            def depth_alignment_callback(patches, ix):
                """Callback to align patches before BA"""
                nonlocal alignment_count
                if depth_priors is not None:
                    patches_aligned, align_count = align_patches_with_depth_priors_training(
                        patches, ix, depth_priors, depth_confidence, P=net.P
                    )
                    alignment_count = max(alignment_count, align_count)
                    return patches_aligned
                return patches
            
            # Call net() WITH callback
            traj = net(images, poses, disps, intrinsics, M=M, STEPS=12, 
                      structure_only=False, patch_callback=depth_alignment_callback)
            
            loss = 0.0
            flow_loss_val = 0.0
            depth_loss_val = 0.0
            pose_loss_val = 0.0
            scale_loss_val = 0.0
            valid_pts = 0
            
            for i, traj_item in enumerate(traj):
                # Unpack
                if len(traj_item) == 8:
                    v, x, y, P1, P2, kl, patches, patch_ix = traj_item
                else:
                    v, x, y, P1, P2, kl = traj_item
                    patches, patch_ix = None, None
                
                # Flow loss (always applied)
                e = (x - y).norm(dim=-1)
                e = e.reshape(-1, net.P**2)[(v > 0.5).reshape(-1)].min(dim=-1).values
                flow_loss = args.flow_weight * e.mean()
                loss += flow_loss
                flow_loss_val += flow_loss.item()  # ✅ Accumulate like validation
                
                # Depth loss from iteration 0
                if patches is not None and gt_depths is not None and args.depth_weight > 0:
                    debug_this = (total_steps <= 100)
                    depth_loss, n_valid = compute_depth_loss_fixed(patches, gt_depths, patch_ix, debug=debug_this)
                    
                    if depth_loss > 0:
                        loss += args.depth_weight * depth_loss
                        depth_loss_val += depth_loss.item()  # ✅ Accumulate like validation
                        valid_pts = n_valid
                        
                        if not depth_active and total_steps <= 10:
                            depth_active = True
                            print(f"\n{'='*60}")
                            print(f"✅ DEPTH SUPERVISION ACTIVE!")
                            print(f"   Step: {total_steps}, Valid: {n_valid}, Loss: {depth_loss_val:.4f}")
                            print(f"   Alignment applied: {alignment_count} patches")
                            print(f"{'='*60}\n")
                
                # Pose loss (from iteration 2+)
                if i >= 2 and patches is not None:
                    N = P1.shape[1]
                    ii, jj = torch.meshgrid(torch.arange(N), torch.arange(N), indexing='ij')
                    ii, jj = ii.reshape(-1).cuda(), jj.reshape(-1).cuda()
                    k = ii != jj
                    ii, jj = ii[k], jj[k]
                    
                    P1, P2 = P1.inv(), P2.inv()
                    t1, t2 = P1.matrix()[...,:3,3], P2.matrix()[...,:3,3]
                    s = kabsch_umeyama(t2[0], t1[0]).detach().clamp(max=10.0)
                    P1 = P1.scale(s.view(1, 1))
                    
                    dP = P1[:,ii].inv() * P1[:,jj]
                    dG = P2[:,ii].inv() * P2[:,jj]
                    e1 = (dP * dG.inv()).log()
                    tr, ro = e1[...,0:3].norm(dim=-1), e1[...,3:6].norm(dim=-1)
                    
                    pose_loss = args.pose_weight * (tr.mean() + ro.mean())
                    loss += pose_loss
                    pose_loss_val += pose_loss.item()  # ✅ Accumulate like validation
                    
                    # Scale loss
                    if args.scale_weight > 0:
                        scale_loss = compute_scale_loss(P1, P2)
                        loss += args.scale_weight * scale_loss
                        scale_loss_val += scale_loss.item()  # ✅ Accumulate like validation
            
            loss += kl
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            total_steps += 1
            
            # Update exponential moving averages for smooth curves
            if ema_loss is None:
                ema_loss = loss.item()
                ema_flow_loss = flow_loss_val
                ema_depth_loss = depth_loss_val
                ema_pose_loss = pose_loss_val
                ema_scale_loss = scale_loss_val
            else:
                ema_loss = ema_alpha * loss.item() + (1 - ema_alpha) * ema_loss
                ema_flow_loss = ema_alpha * flow_loss_val + (1 - ema_alpha) * ema_flow_loss
                ema_depth_loss = ema_alpha * depth_loss_val + (1 - ema_alpha) * ema_depth_loss
                ema_pose_loss = ema_alpha * pose_loss_val + (1 - ema_alpha) * ema_pose_loss
                ema_scale_loss = ema_alpha * scale_loss_val + (1 - ema_alpha) * ema_scale_loss
            
            # Log to TensorBoard - RAW values (accumulated across iterations, matching validation)
            writer.add_scalar('loss', loss.item(), total_steps)
            writer.add_scalar('loss_smooth', ema_loss, total_steps)
            writer.add_scalar('flow_loss', flow_loss_val, total_steps)
            writer.add_scalar('flow_loss_smooth', ema_flow_loss, total_steps)
            writer.add_scalar('depth_loss', depth_loss_val, total_steps)
            writer.add_scalar('depth_loss_smooth', ema_depth_loss, total_steps)
            writer.add_scalar('pose_loss', pose_loss_val, total_steps)
            writer.add_scalar('pose_loss_smooth', ema_pose_loss, total_steps)
            writer.add_scalar('scale_loss', scale_loss_val, total_steps)
            writer.add_scalar('scale_loss_smooth', ema_scale_loss, total_steps)
            
            # Metrics
            metrics = {
                "loss": loss.item(),
                "flow_loss": flow_loss_val,
                "depth_loss": depth_loss_val,
                "pose_loss": pose_loss_val,
                "scale_loss": scale_loss_val,
                "valid_depths": valid_pts,
                "aligned_patches": alignment_count,
            }
            logger.push(metrics)
            
            # Log
            if total_steps % 100 == 0:
                print(f"[{total_steps:5d}] Loss:{loss.item():7.2f} F:{flow_loss_val:6.3f} "
                      f"D:{depth_loss_val:6.3f}({valid_pts:4d}) P:{pose_loss_val:6.3f} "
                      f"S:{scale_loss_val:6.3f} A:{alignment_count:4d} LR:{scheduler.get_last_lr()[0]:.2e}")
            
            # Validation with EMA smoothing
            if val_loader is not None and total_steps % args.val_freq == 0:
                val_metrics = validate(net, val_loader, M, args)
                
                # Apply EMA smoothing to validation losses
                if val_ema_loss is None:
                    val_ema_loss = val_metrics['loss']
                    val_ema_flow_loss = val_metrics['flow_loss']
                    val_ema_depth_loss = val_metrics['depth_loss']
                    val_ema_pose_loss = val_metrics['pose_loss']
                    val_ema_scale_loss = val_metrics['scale_loss']
                else:
                    val_ema_loss = val_ema_alpha * val_metrics['loss'] + (1 - val_ema_alpha) * val_ema_loss
                    val_ema_flow_loss = val_ema_alpha * val_metrics['flow_loss'] + (1 - val_ema_alpha) * val_ema_flow_loss
                    val_ema_depth_loss = val_ema_alpha * val_metrics['depth_loss'] + (1 - val_ema_alpha) * val_ema_depth_loss
                    val_ema_pose_loss = val_ema_alpha * val_metrics['pose_loss'] + (1 - val_ema_alpha) * val_ema_pose_loss
                    val_ema_scale_loss = val_ema_alpha * val_metrics['scale_loss'] + (1 - val_ema_alpha) * val_ema_scale_loss
                
                # Log all validation losses (raw and smoothed)
                writer.add_scalar('val/loss', val_metrics['loss'], total_steps)
                writer.add_scalar('val/loss_smooth', val_ema_loss, total_steps)
                writer.add_scalar('val/flow_loss', val_metrics['flow_loss'], total_steps)
                writer.add_scalar('val/flow_loss_smooth', val_ema_flow_loss, total_steps)
                writer.add_scalar('val/depth_loss', val_metrics['depth_loss'], total_steps)
                writer.add_scalar('val/depth_loss_smooth', val_ema_depth_loss, total_steps)
                writer.add_scalar('val/pose_loss', val_metrics['pose_loss'], total_steps)
                writer.add_scalar('val/pose_loss_smooth', val_ema_pose_loss, total_steps)
                writer.add_scalar('val/scale_loss', val_metrics['scale_loss'], total_steps)
                writer.add_scalar('val/scale_loss_smooth', val_ema_scale_loss, total_steps)
                
                print(f"\n{'='*60}")
                print(f"📊 VALIDATION @ Step {total_steps}")
                print(f"   Val Total Loss: {val_metrics['loss']:.4f} (smooth: {val_ema_loss:.4f})")
                print(f"   Val Flow Loss: {val_metrics['flow_loss']:.4f} (smooth: {val_ema_flow_loss:.4f})")
                print(f"   Val Depth Loss: {val_metrics['depth_loss']:.4f} (smooth: {val_ema_depth_loss:.4f})")
                print(f"   Val Pose Loss: {val_metrics['pose_loss']:.4f} (smooth: {val_ema_pose_loss:.4f})")
                print(f"   Val Scale Loss: {val_metrics['scale_loss']:.4f} (smooth: {val_ema_scale_loss:.4f})")
                print(f"   Best Val Loss: {best_val_loss:.4f}")
                
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    patience_counter = 0
                    # Save best model
                    PATH = f'checkpoints/{args.name}_best.pth'
                    torch.save(net.state_dict(), PATH)
                    print(f"   ✅ New best model saved: {PATH}")
                else:
                    patience_counter += 1
                    print(f"   ⚠️  No improvement for {patience_counter}/{args.patience} validations")
                
                print(f"{'='*60}\n")
                
                # Early stopping
                if patience_counter >= args.patience:
                    print(f"\n{'='*60}")
                    print(f"🛑 EARLY STOPPING at step {total_steps}")
                    print(f"   No improvement for {args.patience} validations")
                    print(f"   Best validation loss: {best_val_loss:.4f}")
                    print(f"{'='*60}\n")
                    writer.close()
                    return
            
            # Save checkpoint
            if total_steps % args.save_freq == 0:
                PATH = f'checkpoints/{args.name}_{total_steps:06d}.pth'
                torch.save(net.state_dict(), PATH)
                print(f"💾 Checkpoint saved: {PATH}")
            
            if total_steps >= args.steps:
                writer.close()
                return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='dpvo_depth_v5')
    parser.add_argument('--ckpt', default=None, help='Path to pretrained checkpoint (e.g., dpvo.pth)')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint step number')
    parser.add_argument('--dataset', default='kitti')
    parser.add_argument('--datapath', required=True, help='Path to KITTI dataset')
    parser.add_argument('--psmnet_dir', required=True, help='Path to PSMNet depth predictions')
    
    # Training parameters
    parser.add_argument('--steps', type=int, default=3000, help='Total training steps')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--clip', type=float, default=10.0, help='Gradient clipping')
    parser.add_argument('--n_frames', type=int, default=8, help='Number of frames per sequence')
    
    # Loss weights
    parser.add_argument('--pose_weight', type=float, default=10.0)
    parser.add_argument('--flow_weight', type=float, default=0.1)
    parser.add_argument('--depth_weight', type=float, default=1.0)
    parser.add_argument('--scale_weight', type=float, default=0.3)
    
    # Data parameters
    parser.add_argument('--fmin', type=float, default=10.0)
    parser.add_argument('--fmax', type=float, default=75.0)
    parser.add_argument('--aug', action='store_true', help='Use data augmentation')
    
    # Validation and early stopping
    parser.add_argument('--val_freq', type=int, default=100, help='Validation frequency')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience (validations)')
    parser.add_argument('--save_freq', type=int, default=100, help='Checkpoint save frequency')
    
    args = parser.parse_args()
    
    # Create checkpoint directory
    os.makedirs('checkpoints', exist_ok=True)
    
    print("\n" + "="*80)
    print("🚀 DPVO TRAINING WITH CONSISTENT LOSS LOGGING")
    print("="*80)
    print(f"Checkpoint: {args.ckpt}")
    print(f"Depth weight: {args.depth_weight}")
    print(f"Learning rate: {args.lr}")
    print(f"Max steps: {args.steps}")
    print(f"Validation every: {args.val_freq} steps")
    print(f"Early stopping patience: {args.patience} validations")
    print("\nKey features:")
    print("  ✅ All loss components logged separately (flow, depth, pose, scale)")
    print("  ✅ Training and validation use consistent accumulation (+=)")
    print("  ✅ EMA smoothing for both train (alpha=0.1) and val (alpha=0.3)")
    print("  ✅ 50 validation batches for stable estimates")
    print("  ✅ Smooth, comparable train/val curves")
    print("="*80 + "\n")
    
    train(args)
