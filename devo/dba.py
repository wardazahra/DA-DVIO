import torch
import numpy as np
import torch.nn.functional as F

from dpvo import fastba
from dpvo import altcorr
from dpvo import lietorch
from dpvo.lietorch import SE3
from dpvo.patchgraph import PatchGraph

from dpvo.net import VONet # TODO add net.py
from .enet import eVONet
from .utils import *
from . import projective_ops as pops

autocast = torch.cuda.amp.autocast
Id = SE3.Identity(1, device="cuda")

from utils.viz_utils import visualize_voxel

from .multi_sensor import MultiSensorState
import gtsam
from gtsam.symbol_shorthand import B, V, X
import math
from scipy.spatial.transform import Rotation
import devo.geoFunc.trans as trans
import copy
import bisect


def CustomHessianFactor(values: gtsam.Values, H: np.ndarray, v: np.ndarray):
    info_expand = np.zeros([H.shape[0]+1,H.shape[1]+1])
    info_expand[0:-1,0:-1] = H
    info_expand[0:-1,-1] = v
    info_expand[-1,-1] = 0.0 # This is meaningless.
    h_f = gtsam.HessianFactor(values.keys(),[6]*len(values.keys()),info_expand)
    l_c = gtsam.LinearContainerFactor(h_f,values)
    return l_c

class DBA:
    def __init__(self, cfg, network, evs=False, ht=480, wd=640, viz=False, viz_flow=False, dim_inet=384, dim_fnet=128, dim=32):
        self.cfg = cfg
        self.evs = evs # 是否使用事件

        self.dim_inet = dim_inet
        self.dim_fnet = dim_fnet
        self.dim = dim
        # # TODO add patch_selector

        self.args = cfg#参数文件传入
        
        self.load_weights(network)#读取网络的权重
        self.is_initialized = False #是否视觉初始化
        self.enable_timing = False # TODO timing in param

        torch.set_num_threads(2)

        self.viz_flow = viz_flow
        
        self.M = self.cfg.PATCHES_PER_FRAME     # (default: 96) patch的数目
        self.N = self.cfg.BUFFER_SIZE           # max number of keyframes (default: 2048)，buffer的数目

        self.ht = ht    # image height
        self.wd = wd    # image width

        RES = self.RES

        ### state attributes ###
        self.tlist = []#时间戳列表(应该是cpu上的)
        self.counter = 0 # how often this network is called __call__()

        # keep track of global-BA calls
        self.ran_global_ba = np.zeros(100000, dtype=bool)

        self.flow_data = {}

        # dummy image for visualization
        self.image_ = torch.zeros(self.ht, self.wd, 3, dtype=torch.uint8, device="cpu")#原图的大小

        self.patches_gt_ = torch.zeros(self.N, self.M, 3, self.P, self.P, dtype=torch.float, device="cuda")

        ### network attributes ###
        if self.cfg.MIXED_PRECISION:
            self.kwargs = kwargs = {"device": "cuda", "dtype": torch.half}
        else:
            self.kwargs = kwargs = {"device": "cuda", "dtype": torch.float}

        ### frame memory size ###
        self.pmem = self.mem = 36 #32
        if self.cfg.LOOP_CLOSURE:
            self.last_global_ba = -1000 # keep track of time since last global opt
            self.pmem = self.cfg.MAX_EDGE_AGE # patch memory
        
        self.imap_ = torch.zeros(self.pmem, self.M, self.dim_inet, **kwargs)
        self.gmap_ = torch.zeros(self.pmem, self.M, self.dim_fnet, self.P, self.P, **kwargs)

        ht = int(ht // RES)
        wd = int(wd // RES)

        # 定义一个PatchGraph类的实例
        self.pg = PatchGraph(self.cfg, self.P, self.dim_inet, self.pmem, **kwargs)

        # classic backend
        if self.cfg.CLASSIC_LOOP_CLOSURE:
            self.load_long_term_loop_closure()

        self.fmap1_ = torch.zeros(1, self.mem, self.dim_fnet, int(ht // 1), int(wd // 1), **kwargs)
        self.fmap2_ = torch.zeros(1, self.mem, self.dim_fnet, int(ht // 4), int(wd // 4), **kwargs)

        # feature pyramid
        self.pyramid = (self.fmap1_, self.fmap2_)

        self.viewer = None
        
        # Add these lines after existing depth parameters
        self.base_strength_imu_active = self.cfg.get('base_strength', 0.08)
        self.base_strength_imu_inactive = self.cfg.get('base_strength', 0.2) 
        self.confidence_threshold = self.cfg.get('confidence_threshold', 0.8)
                
        # ============================================================================
        # DEPTH HANDLING: Network owns all depth logic (no local storage in DBA)
        # ============================================================================
        # Network will handle:
        # - depth_maps storage (via network.add_depth_map())
        # - depth initialization (via network.initialize_from_depth())
        # - depth fusion (via network.fuse_depth())
        # DBA only needs to call network methods
        # ============================================================================
        
        # Configure network fusion parameters
        self.network.set_fusion_params(
            M=self.M,
            optimization_window=self.cfg.OPTIMIZATION_WINDOW if hasattr(self.cfg, 'OPTIMIZATION_WINDOW') else 10
        ) 
        
        if viz:
            self.start_viewer()

        ### event-based DBA (所有时间应该都已经减去了时间偏移)
        self.state = MultiSensorState()#位姿估计的状态
        self.last_t0 = 0 #上一帧的t0
        self.last_t1 = 0 #上一帧的t1
        self.cur_graph = None #当前gtsam维护的图
        self.cur_result = None #gtsam优化的结果

        ### 记录一下边缘化的矩阵
        self.marg_factor = None #这是marginalization的因子
        self.prior_factor_map = {} #用于存储先验地图？请见函数set_prior()
        self.cur_ii = None
        self.cur_jj = None
        self.cur_kk = None
        self.cur_target = None #存有当前window下的所有的光流
        self.cur_weight = None #存有当前window下的所有的光流的权重

        self.imu_enabled = False #初始化为false，初始化的时候不用imu直到满足一定条件的时候才开启（进行了视觉惯性对齐）
        self.ignore_imu = False

        # IMU-Camera Extrinsics外参。extrinsics, need to be set in the main .py
        self.Ti1c = None  # shape = (4,4)（#相机到IMU的变换矩阵）
        self.Tbc = None   # gtsam.Pose3 （gtsam.Pose3的形式表示IMU-Camera Extrinsics的位姿）
        self.tbg = None   # shape = (3) 这个估计是重力g到IMU的变换

        self.reinit = False #是否重新初始化
        self.vi_init_time = 0.0 #视觉惯性初始化的时间
        self.vi_init_t1 = -1 #视觉惯性初始化的t1
        self.vi_warmup = 12 #视觉等待有12帧就开始初始化
        self.init_pose_sigma =np.array([0.1, 0.1, 0.0001, 0.0001,0.0001,0.0001])
        self.init_bias_sigma =np.array([1.0,1.0,1.0, 0.1, 0.1, 0.1])

        # local optimization window
        self.t0 = 0 #起始帧的索引
        self.t1 = 0 #结束帧的索引（应该也就是当前的输入数据的索引）

        self.all_imu = None #所有的IMU数据(通过前面的文件读入了)
        self.cur_imu_ii = 0 #当前处理到的IMU数据的索引
        self.is_init = False #IMU是否初始化
        self.is_init_VI = False #是否视觉惯性初始化
        self.all_gt = None #所有的GT 时间戳+pose数据
        self.all_gt_keys = None #所有的GT 时间戳

        #是否只进行视觉估计，cfg.ENALBE_IMU为False时，只进行视觉估计visual_only为true，cfg.ENALBE_IMU为True时，进行visual_only为False
        self.visual_only= (cfg.ENALBE_IMU==False)
        self.visual_only_init = False

        self.high_freq_output = False #True #是否进行高频输出

         # visualization/output
        self.plt_pos     = [[],[]]    # X, Y
        self.plt_pos_ref = [[],[]]    # X, Y
        self.refTw       = np.eye(4,4)
        self.poses_save   = [] # 记录位姿

    # ============================================================================
    # STEREO DEPTH INTEGRATION: Add depth injection methods from DPVO
    # ============================================================================
    
    # ============================================================================
    # REMOVED: store_depth_priors_for_alignment() 
    # Network now handles depth storage via network.add_depth_map()
    # ============================================================================
    
    # ============================================================================
    # REMOVED: align_patches_with_depth_priors()
    # Network now handles fusion via network.fuse_depth() during VIO_update()
    # ============================================================================
    
    # ============================================================================
    # REMOVED: inject_psmnet_depths_at_patches()
    # Network now handles initialization via network.initialize_from_depth()
    # ============================================================================
    # REMOVED: store_depth_priors_for_romeo() - was never actually called
    # Network handles depth storage via network.add_depth_map()
    # ============================================================================
    
    def should_use_depth_constraints(self):
        """Determine if depth constraints should be applied - now handled by network"""
        # Network fusion always runs when initialized and depth maps available
        return self.is_initialized and len(self.network.depth_maps) > 0

    def apply_depth_regularization_to_patches(self):
        """Apply depth fusion via network (replaces manual depth constraints)"""
        if not self.should_use_depth_constraints():
            return
        
        # Network handles all fusion logic
        self.network.fuse_depth(self.patches, self.ix, self.n)

    # ============================================================================
    # END DEPTH HANDLING (Network-based)
    # ============================================================================

    # 用于设置prior_factor_map
    def set_prior(self, t0, t1):
        for i in range(t0,t0+2):
            self.prior_factor_map[i] = []
            init_pose_sigma = self.init_pose_sigma
            if len(self.init_pose_sigma.shape) > 1:
                init_pose_sigma = self.init_pose_sigma[i-t0]
            self.prior_factor_map[i].append(gtsam.PriorFactorPose3(X(i),\
                                         self.state.wTbs[i], \
                                         gtsam.noiseModel.Diagonal.Sigmas(init_pose_sigma)))
            if not self.ignore_imu:
                self.prior_factor_map[i].append(gtsam.PriorFactorConstantBias(B(i),\
                                             self.state.bs[i], \
                                             gtsam.noiseModel.Diagonal.Sigmas(self.init_bias_sigma)))
            self.last_t0 = t0
            self.last_t1 = t1
    
    def load_long_term_loop_closure(self):
        try:
            from dpvo.loop_closure.long_term import LongTermLoopClosure
            self.long_term_lc = LongTermLoopClosure(self.cfg, self.pg)#初始化LongTermLoopClosure类
        except ModuleNotFoundError as e:
            self.cfg.CLASSIC_LOOP_CLOSURE = False
            print(f"WARNING: {e}")

    def load_weights(self, network):
        # load network from checkpoint file
        if isinstance(network, str):
            print(f"Loading from {network}")
            checkpoint = torch.load(network)
            
            # FIX: Use correct constructor parameters for each network
            if not self.evs:  # Image mode
                from dpvo.net import VONet
                print("Loading VONet for image processing")
                self.network = VONet(use_viewer=False)  # VONet's simple constructor
            else:  # Event mode
                print("Loading eVONet for event processing") 
                self.network = eVONet(self.args, dim_inet=self.dim_inet, dim_fnet=self.dim_fnet, dim=self.dim, patch_selector=self.cfg.PATCH_SELECTOR)
            
            if 'model_state_dict' in checkpoint:
                self.network.load_state_dict(checkpoint['model_state_dict'])
            else:
                # legacy
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in checkpoint.items():
                    if "update.lmbda" not in k:
                        new_state_dict[k.replace('module.', '')] = v
                self.network.load_state_dict(new_state_dict)
        else:
            self.network = network
            
        # steal network attributes - BUT VONet has different attributes!
        if hasattr(self.network, 'dim_inet'):  # eVONet
            self.dim_inet = self.network.dim_inet
            self.dim_fnet = self.network.dim_fnet
            self.dim = self.network.dim
        else:  # VONet - use hardcoded values from VONet
            self.dim_inet = 384  # DIM from VONet
            self.dim_fnet = 128  # From VONet code
            self.dim = 32       # Default
            
        self.RES = self.network.RES
        self.P = self.network.P
        self.network.cuda()
        self.network.eval()

    def start_viewer(self):
        from dpviewer import Viewer

        intrinsics_ = torch.zeros(1, 4, dtype=torch.float32, device="cuda")

        self.viewer = Viewer(
            self.image_,
            self.poses_,
            self.points_,
            self.colors_,
            intrinsics_)

    @property
    def poses(self):#调用用poses，更新写入用self.pg.poses_
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
        return self.imap_.view(1, self.pmem * self.M, self.dim_inet)

    @property
    def gmap(self):
        return self.gmap_.view(1, self.pmem * self.M, self.dim_fnet, 3, 3)

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
    
    @property
    def patches_gt(self):
        return self.patches_gt_.view(1, self.N*self.M, 3, 3, 3)

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
            
        # ===== SCALE COMPUTATION USING NETWORK DEPTH MAPS =====
        stereo_depths = []
        dba_depths = []
        
        # Use network's depth_maps instead of local depth_priors
        for frame_id in self.network.depth_maps.keys():
            if frame_id >= self.n:
                continue
            
            # Sample PSMNet depths at patch locations for this frame
            P = self.P
            M = self.M
            
            # Get patch coordinates for this frame
            start_idx = frame_id * M
            end_idx = start_idx + M
            
            if end_idx > self.patches.shape[1]:
                continue
            
            current_patches = self.patches[0, start_idx:end_idx]
            patch_centers_u = current_patches[:, 0, P//2, P//2]
            patch_centers_v = current_patches[:, 1, P//2, P//2]
            
            # Scale to full resolution
            u_full = (patch_centers_u * self.RES).long().clamp(0, self.network.depth_maps[frame_id].shape[1]-1)
            v_full = (patch_centers_v * self.RES).long().clamp(0, self.network.depth_maps[frame_id].shape[0]-1)
            
            # Sample PSMNet depths
            sampled_depths = self.network.depth_maps[frame_id][v_full, u_full]
            
            # Get DBA inverse depths
            dba_inv_depths = current_patches[:, 2, P//2, P//2]
            
            # Filter valid depths
            valid = (sampled_depths > 1.0) & (sampled_depths < 80.0) & (dba_inv_depths > 0)
            
            if valid.sum() > 10:
                stereo_depths.extend(sampled_depths[valid].cpu().numpy())
                dba_depths.extend((1.0 / dba_inv_depths[valid]).cpu().numpy())
        
        if len(stereo_depths) > 0:
            median_stereo = np.median(stereo_depths)
            median_dba = np.median(dba_depths)
            S = median_stereo / median_dba
            S = S * 1.45 #warda might disable or enable this
            print(f"Scale: {S:.4f} ({len(stereo_depths)} samples)")
        else:
            S = 1.0
            print("⚠️ No valid depth samples for scale computation, using S=1.0")
        # ===== END SCALE COMPUTATION =====

        if self.cfg.LOOP_CLOSURE: #0: #1:
            """ interpolate missing poses """
            print("keyframes", self.n)
            self.traj = {}
            for i in range(self.n):
                self.traj[self.pg.tstamps_[i]] = self.pg.poses_[i]

            poses = [self.get_pose(t) for t in range(self.counter)]
            poses = lietorch.stack(poses, dim=0)
            poses = poses.inv().data.cpu().numpy()#注意都是用w2c来存的，传出需要改为c2w
            tstamps = np.array(self.tlist, dtype=np.float64)
            if self.viewer is not None:
                self.viewer.join()
        else:
            # self.poses_save 中获取timestamps和poses，self.poses_save每一行的第一个是时间，其余七个是pose
            

            print(f"\n{'='*60}")
            print(f"DEBUG poses_save format:")
            print(f"poses_save[0] = {self.poses_save[0]}")  # First entry
            print(f"poses_save shape: {len(self.poses_save)} x {len(self.poses_save[0])}")
            
            poses = np.array(self.poses_save)[:, 1:]
            print(f"After slicing [:, 1:]:")
            print(f"poses.shape = {poses.shape}")
            print(f"poses[0] = {poses[0]}")
            print(f"{'='*60}\n")
            poses = np.array(self.poses_save)[:, 1:]
            #poses[:, :3] *= S
            
            tstamps = np.array(self.poses_save, dtype=np.float64)[:, 0]#获取时间戳

        return poses, tstamps
    
    def corr(self, coords, indicies=None):
        """ local correlation volume """
        ii, jj = indicies if indicies is not None else (self.pg.kk, self.pg.jj)
        ii1 = ii % (self.M * self.pmem)
        jj1 = jj % (self.mem)
        corr1 = altcorr.corr(self.gmap, self.pyramid[0], coords / 1, ii1, jj1, 3)
        corr2 = altcorr.corr(self.gmap, self.pyramid[1], coords / 4, ii1, jj1, 3)
        return torch.stack([corr1, corr2], -1).view(1, len(ii), -1)

    def reproject(self, indicies=None):
        """ reproject patch k from i -> j """
        (ii, jj, kk) = indicies if indicies is not None else (self.pg.ii, self.pg.jj, self.pg.kk)
        coords = pops.transform(SE3(self.poses), self.patches, self.intrinsics, ii, jj, kk)
        return coords.permute(0, 1, 4, 2, 3).contiguous()

    def append_factors(self, ii, jj):
        self.pg.jj = torch.cat([self.pg.jj, jj])
        self.pg.kk = torch.cat([self.pg.kk, ii])#插入的ii其实就是patch的索引，kk
        self.pg.ii = torch.cat([self.pg.ii, self.ix[ii]]) #self.ix[ii]也就是self.ix[kk]才是ii的索引

        net = torch.zeros(1, len(ii), self.dim_inet, **self.kwargs)
        self.pg.net = torch.cat([self.pg.net, net], dim=1)

    def remove_factors(self, m, store: bool):
        assert self.pg.ii.numel() == self.pg.weight.shape[1]
        if store:#如果store为True，则将要删除的边存储到inactive edge中
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
        """ kinda hacky way to ensure enough motion for initialization """
        kk = torch.arange(self.m-self.M, self.m, device="cuda")
        jj = self.n * torch.ones_like(kk)
        ii = self.ix[kk]

        net = torch.zeros(1, len(ii), self.dim_inet, **self.kwargs)
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

        # flow, _ = pops.flow_mag(SE3(self.poses), self.patches, self.intrinsics, ii, jj, kk, beta=0.5)
        flow = pops.flow_mag(SE3(self.poses), self.patches, self.intrinsics, ii, jj, kk, beta=0.5)
        return flow.mean().item()

    def keyframe(self):
        
        i = self.n - self.cfg.KEYFRAME_INDEX - 1 #倒数第5帧
        j = self.n - self.cfg.KEYFRAME_INDEX + 1 #倒数第3帧
        m = self.motionmag(i, j) + self.motionmag(j, i)

        # print(f'the mition between {i} and {j} is {m/2}')

        if m / 2 < self.cfg.KEYFRAME_THRESH:#如果运动小于阈值，就不是关键帧
            k = self.n - self.cfg.KEYFRAME_INDEX#倒数第4帧
            t0 = self.pg.tstamps_[k-1]
            t1 = self.pg.tstamps_[k]

            dP = SE3(self.pg.poses_[k]) * SE3(self.pg.poses_[k-1]).inv()
            self.pg.delta[t1] = (t0, dP)

            to_remove = (self.pg.ii == k) | (self.pg.jj == k)
            self.remove_factors(to_remove, store=False)#此处是不会存的，因为运动不够，不是关键帧

            # 将k之后的索引都减掉
            self.pg.kk[self.pg.ii > k] -= self.M
            self.pg.ii[self.pg.ii > k] -= 1
            self.pg.jj[self.pg.jj > k] -= 1

            # 执行数据移动的操作（从k到当前帧）
            for i in range(k, self.n-1):
                self.pg.tstamps_[i] = self.pg.tstamps_[i+1]
                self.pg.colors_[i] = self.pg.colors_[i+1]
                self.pg.poses_[i] = self.pg.poses_[i+1]
                self.pg.patches_[i] = self.pg.patches_[i+1]
                self.patches_gt_[i] = self.patches_gt_[i+1]
                self.pg.intrinsics_[i] = self.pg.intrinsics_[i+1]

                self.imap_[i % self.pmem] = self.imap_[(i+1) % self.pmem]
                self.gmap_[i % self.pmem] = self.gmap_[(i+1) % self.pmem]
                self.fmap1_[0,i%self.mem] = self.fmap1_[0,(i+1)%self.mem]
                self.fmap2_[0,i%self.mem] = self.fmap2_[0,(i+1)%self.mem]

                #IMU数据的移动
                if i == k:
                    for iii in range(len(self.state.preintegrations_meas[i])):
                        dd = self.state.preintegrations_meas[i][iii]#获取第k帧的IMU信息(Acc, Omega, Delta_t, t)
                        if dd[2] > 0:
                            self.state.preintegrations[i-1].integrateMeasurement(dd[0],\
                                                                                        dd[1],\
                                                                                        dd[2])
                        
                        self.state.preintegrations_meas[i-1].append(dd)
                    self.state.preintegrations.pop(i)
                    self.state.preintegrations_meas.pop(i)

                    self.state.wTbs.pop(i)
                    self.state.bs.pop(i)
                    self.state.vs .pop(i)

            self.n -= 1 #由于删掉了一帧，所以往前挪一帧
            self.m-= self.M #减去这些patch获得总的patch数量

            if self.cfg.CLASSIC_LOOP_CLOSURE:
                self.long_term_lc.keyframe(k)

        # 当ii是当前帧之前的22帧时，去掉
        to_remove = self.ix[self.pg.kk] < self.n - self.cfg.REMOVAL_WINDOW # Remove edges falling outside the optimization window
        if self.cfg.LOOP_CLOSURE:
            # ...unless they are being used for loop closure
            lc_edges = ((self.pg.jj - self.pg.ii) > 30) & (self.pg.jj > (self.n - self.cfg.OPTIMIZATION_WINDOW))
            to_remove = to_remove & ~lc_edges
        self.remove_factors(to_remove, store=True) #此处则是要存的，因为是关键帧，只是滑动出了窗口

    # 全局的BA优化
    def __run_global_BA(self):
        """ Global bundle adjustment
         Includes both active and inactive edges """
        full_target = torch.cat((self.pg.target_inac, self.pg.target), dim=1)
        full_weight = torch.cat((self.pg.weight_inac, self.pg.weight), dim=1)
        full_ii = torch.cat((self.pg.ii_inac, self.pg.ii))
        full_jj = torch.cat((self.pg.jj_inac, self.pg.jj))
        full_kk = torch.cat((self.pg.kk_inac, self.pg.kk))

        # self.pg.normalize()#! 归一化,目的是？
        lmbda = torch.as_tensor([1e-4], device="cuda") #给定值，不像droid那样需要计算
        t0 = self.pg.ii.min().item()
        # 似乎只是加入了全局的边，和targer weight等信息，然后进行全局的BA优化，并无太大区别？
        # 主要区别应该就是前面用的eff_impl=False，这里用的是True
        fastba.BA(self.poses, self.patches, self.intrinsics,
            full_target, full_weight, lmbda, full_ii, full_jj, full_kk, t0, self.n, M=self.M, iterations=2, eff_impl=True)
        self.ran_global_ba[self.n] = True #标记当前帧已经运行过全局BA优化

        self.last_t0 = t0
        self.last_t1 = self.n
    
    def __run_DBA(self, target, weight, lmbda, ii, jj, kk, t0, t1, eff_impl):
        """ 执行marginalization """
        if self.last_t1!=t1 or self.last_t0 != t0:
            if self.last_t0 >= t0:
                if eff_impl==False : #若不做GBA
                    t0 = self.last_t0
            else: # self.last_t0 < t0 也就是上一个t0小于当前的t0
                # print(f"Marginalization!!!!!!!!!!!{self.tstamp[t1-1]}")
                marg_paras = []
                # Construct a temporary factor graph (related to the old states) to obtain the marginalization information
                graph = gtsam.NonlinearFactorGraph()#构建一个新的图用于存放marginalization的因子

                # 索取要marginalize的索引
                marg_idx = torch.logical_and(self.last_t0<=self.cur_ii, self.cur_ii<t0)  # last_t0 <= ii < t0
                marg_idx2 = torch.logical_and(self.cur_ii<self.last_t1-2, self.cur_jj<self.last_t1-2) 
                marg_idx = torch.logical_and(marg_idx, marg_idx2)#获取要marginalize的索引
                # marg_idx = (self.cur_ii == self.last_t0)
                marg_ii = self.cur_ii[marg_idx]
                marg_jj = self.cur_jj[marg_idx]
                marg_kk = self.cur_kk[marg_idx]
                marg_t0 = self.last_t0 #上一个t0去掉
                marg_t1 = t0 + 1 #当前的t0+1
                if len(marg_ii) > 0:
                    marg_t0 = self.last_t0 
                    marg_t1 = torch.max(marg_jj).item()+1
                    marg_result = gtsam.Values()
                    for i in range(self.last_t0,marg_t1):
                        if i < t0:
                            marg_paras.append(X(i))
                        marg_result.insert(X(i), self.cur_result.atPose3(X(i)))

                    # 要marginalize的视觉因子
                    marg_target = self.cur_target[:,marg_idx]
                    marg_weight = self.cur_weight[:,marg_idx]
                    
                    # 接下来添加 marginalization的视觉因子
                    bafactor = fastba.BAFactor()#初始化类，准备用于构建视觉的因子
                    # 上面要确认获得的是marg_target, marg_weight,marg_ii, marg_jj, marg_t0, marg_t1,
                    bafactor.init(self.poses.data, self.patches, self.intrinsics,
                            marg_target, marg_weight, lmbda, marg_ii, marg_jj, marg_kk ,self.M, marg_t0, marg_t1, 2, eff_impl)
                    H = torch.zeros([(marg_t1-marg_t0)*6,(marg_t1-marg_t0)*6],dtype=torch.float64,device='cpu')
                    v = torch.zeros([(marg_t1-marg_t0)*6],dtype=torch.float64,device='cpu')
                    bafactor.hessian(H,v)#边缘化
                    
                    for i in range(6): H[i,i] += 0.00025  # for stability

                    # Hg,vg = BA2GTSAM(H,v,self.Tbc)
                    Hgg = gtsam.BA2GTSAM(H,v,self.Tbc)#将BA的Hessian和v转换为gtsam的Hessian和v
                    Hg = Hgg[0:(marg_t1-marg_t0)*6,0:(marg_t1-marg_t0)*6]
                    vg = Hgg[0:(marg_t1-marg_t0)*6,  (marg_t1-marg_t0)*6]
                    vis_factor = CustomHessianFactor(marg_result,Hg,vg)#构建视觉因子

                    # graph.push_back(vis_factor) # # #添加视觉因子到marginalization的图中

                # 添加其他因子到marginalization的图中
                for i in range(self.last_t0,marg_t1):
                    if i < t0:
                        if X(i) not in marg_paras:
                            marg_paras.append(X(i))
                        if not self.ignore_imu:#若不忽略imu
                            marg_paras.append(V(i))
                            marg_paras.append(B(i))
                            graph.push_back(gtsam.gtsam.CombinedImuFactor(\
                                        X(i),V(i),X(i+1),V(i+1),B(i),B(i+1),\
                                        self.state.preintegrations[i]))#添加IMU因子
                
                # 获取先验的地图
                keys = self.prior_factor_map.keys()
                for i in sorted(keys):
                    if i < t0:
                        for iii in range(len(self.prior_factor_map[i])):
                            graph.push_back(self.prior_factor_map[i][iii])
                    del self.prior_factor_map[i]
                # 若有marg_factor则添加到图中
                if not self.marg_factor == None:
                    graph.push_back(self.marg_factor)

                #优化获取marginalization的因子
                self.marg_factor = gtsam.marginalizeOut(graph,self.cur_result,marg_paras)

                # covariance inflation of IMU biases
                if self.reinit == True:#如果重新初始化
                    all_keys = self.marg_factor.keys()
                    for i in range(len(all_keys)):
                        if all_keys[i] == B(t0):
                            all_keys[i] = B(0)
                    graph = gtsam.NonlinearFactorGraph()
                    graph.push_back(self.marg_factor.rekey(all_keys))
                    b_l = gtsam.BetweenFactorConstantBias(B(0),B(t0),gtsam.imuBias.ConstantBias(np.array([.0,.0,.0]),np.array([.0,.0,.0])),\
                                                        gtsam.noiseModel.Diagonal.Sigmas(self.init_bias_sigma))
                    graph.push_back(b_l)
                    result_tmp = self.marg_factor.linearizationPoint()
                    result_tmp.insert(B(0),result_tmp.atConstantBias(B(t0)))
                    self.marg_factor = gtsam.marginalizeOut(graph,result_tmp,[B(0)])
                    self.reinit = False

            # print(f'last_t0 :{self.last_t0},last_t1: {self.last_t1}; t0: {t0}, t1: {t1}; t0_temp: {t0_temp}, t1_temp: {t1_temp}')
            if eff_impl==False : #若有GBA不更新
                self.last_t0 = t0
            self.last_t1 = t1

        """ optimization多传感器联合后端优化 """
        self.cur_graph = gtsam.NonlinearFactorGraph() # 当前gtsam优化时维护的图（与上面marginalization命名一样，但是含义不一样~）
        params = gtsam.LevenbergMarquardtParams()#;params.setMaxIterations(1)

        # 接下来开始添加各种因子到gtsam图中
        # imu factor
        if not self.ignore_imu:
            for i in range(t0,t1):
                if i > t0:
                    imu_factor = gtsam.gtsam.CombinedImuFactor(\
                        X(i-1),V(i-1),X(i),V(i),B(i-1),B(i),\
                        self.state.preintegrations[i-1])
                    self.cur_graph.add(imu_factor)#把imu预积分添加 
        
        # prior factor(此先验因子图感觉像是imu传入的)
        keys = self.prior_factor_map.keys()
        for i in sorted(keys):
            if i >= t0 and i < t1:
                for iii in range(len(self.prior_factor_map[i])):
                    self.cur_graph.push_back(self.prior_factor_map[i][iii])#添加先验因子
        
        # marginalization factor
        # if self.marg_factor is not None:
        if eff_impl==False and self.marg_factor is not None: #没有GBA且有边缘化因子
            self.cur_graph.push_back(self.marg_factor) #添加前面marginalization算出来的的因子
        
        # 初始化一系列gtsam中的参数（部分在marginalization中用到的~）
        # active_index    = torch.logical_and(ii>=t0,jj>=t0)#ii与jj的最大值为t0+10.所以此处应该是大于等于t0的为active_index
        self.cur_ii     = ii#[active_index]
        self.cur_jj     = jj#[active_index]
        self.cur_kk     = kk#[active_index]
        self.cur_target = target#[:,active_index]
        self.cur_weight = weight#[:,active_index]

        # 接下来开始构建视觉的约束因子并放入gtsam图中
        H = torch.zeros([(t1-t0)*6,(t1-t0)*6],dtype=torch.float64,device='cpu')
        v = torch.zeros([(t1-t0)*6],dtype=torch.float64,device='cpu')
        dx = torch.zeros([(t1-t0)*6],dtype=torch.float64,device='cpu') #用于获取gtsam的结果然后更新dba的

        bafactor = fastba.BAFactor()#初始化类，准备用于构建视觉的因子
        #进行初始化
        # bafactor.init(self.poses.data, self.patches, self.intrinsics, 
        #     target, weight, lmbda, ii, jj, kk, self.M, t0, t1, 2) #注意读入cuda代码不要写关键词
        # !下面对
        bafactor.init(self.poses.data, self.patches, self.intrinsics, 
            self.cur_target, self.cur_weight, lmbda, self.cur_ii, self.cur_jj, self.cur_kk, self.M, t0, t1, 2, eff_impl)
        
        """ multi-sensor DBA iterations """
        for iter in range(2):
            if iter > 0:
                self.cur_graph.resize(self.cur_graph.size()-1)#图的size原本为16，resize为15
            bafactor.hessian(H,v) # camera frame
            Hgg = gtsam.BA2GTSAM(H,v,self.Tbc)
            Hg = Hgg[0:(t1-t0)*6,0:(t1-t0)*6]
            vg = Hgg[0:(t1-t0)*6,(t1-t0)*6]

            initial = gtsam.Values()
            for i in range(t0,t1):# 给状态初始值
                initial.insert(X(i), self.state.wTbs[i]) # the indice need to be handled
            initial_vis = copy.deepcopy(initial)
            vis_factor = CustomHessianFactor(initial_vis,Hg,vg)
            self.cur_graph.push_back(vis_factor)#基于droid的结果构建视觉的因子

            if not self.ignore_imu:#如果不忽略IMU就把bias加入
                for i in range(t0,t1):
                    initial.insert(B(i),self.state.bs[i])
                    initial.insert(V(i),self.state.vs[i])
            
            optimizer = gtsam.LevenbergMarquardtOptimizer(self.cur_graph, initial, params)#初始化gtsam的优化器
            self.cur_result = optimizer.optimize()#使用gtsam进行优化

            # retraction and depth update
            for i in range(t0,t1):
                p0 = initial.atPose3(X(i))
                p1 = self.cur_result.atPose3(X(i))
                xi = gtsam.Pose3.Logmap(p0.inverse()*p1)
                dx[(i-t0)*6:(i-t0)*6+6] = torch.tensor(xi)
                if not self.ignore_imu:
                    self.state.bs[i] = self.cur_result.atConstantBias(B(i))
                    self.state.vs[i] = self.cur_result.atVector(V(i))
                self.state.wTbs[i] = self.cur_result.atPose3(X(i))
            dx = torch.tensor(gtsam.GTSAM2BA(dx,self.Tbc))#姿态的变化
            _ = bafactor.retract(dx)# ! （需要double check） 对pose与patch进行更新用于下一次的迭代
            gwp_TODO = 1; # TODO debug
        del bafactor #释放内存

    def update(self):
            """Enhanced update with network depth fusion"""
            
            # STEP 1: Network depth fusion (replaces align_patches_with_depth_priors)
            # EXACT ZED: Pass IMU status to control fusion strength
            if self.is_initialized:
                self.network.fuse_depth(self.patches, self.ix, self.n, imu_enabled=self.imu_enabled)
            
            # STEP 2: Standard DBA update cycle (unchanged)
            coords = self.reproject()#进行重投影

            with autocast(enabled=True):
                corr = self.corr(coords) #计算相关性，获取当前帧与上一帧之间的特征匹配信息。
                ctx = self.imap[:, self.pg.kk % (self.M * self.pmem)]
                self.pg.net, (delta, weight, _) = \
                    self.network.update(self.pg.net, ctx, corr, None, self.pg.ii, self.pg.jj, self.pg.kk)

            lmbda = torch.as_tensor([1e-4], device="cuda")
            weight = weight.float()
            target = coords[...,self.P//2,self.P//2] + delta.float()

            self.pg.target = target
            self.pg.weight = weight

            # STEP 3: Standard Bundle Adjustment (now operates on depth-aligned patches)
            with Timer("BA", enabled=self.enable_timing):
                try:
                    if self.imu_enabled:#如果使用imu
                        t1=self.n
                        eff_impl_flag=False

                        # 根据不同的情况决定t0，full_target，full_weight，full_ii，full_jj，full_kk，eff_impl_flag
                        if (self.pg.ii < self.n - self.cfg.REMOVAL_WINDOW - 1).any() and not self.ran_global_ba[self.n]:
                            # 如果ii中有小于n-REMOVAL_WINDOW-1的值（就是有回环匹配了），且当前帧没有运行过全局BA优化，则运行全局BA优化
                            eff_impl_flag=True #对于全局BA优化使用高效的实现
                            full_target = torch.cat((self.pg.target_inac, self.pg.target), dim=1)
                            full_weight = torch.cat((self.pg.weight_inac, self.pg.weight), dim=1)
                            full_ii = torch.cat((self.pg.ii_inac, self.pg.ii))
                            full_jj = torch.cat((self.pg.jj_inac, self.pg.jj))
                            full_kk = torch.cat((self.pg.kk_inac, self.pg.kk))

                            # self.pg.normalize()#! 归一化,目的是？
                            t0 = self.pg.ii.min().item()

                            self.ran_global_ba[self.n] = True #标记当前帧已经运行过全局BA优化
                            
                        else:#运行局部BA优化
                            t0 = self.n - self.cfg.OPTIMIZATION_WINDOW if self.is_initialized else 1
                            t0 = max(t0, 1)
                            full_target = self.pg.target
                            full_weight = self.pg.weight
                            full_ii =  self.pg.ii
                            full_jj = self.pg.jj
                            full_kk = self.pg.kk
                            eff_impl_flag=False #对于局部BA优化使用低效的实现

                        # CHANGED: Use standard __run_DBA (no post-BA depth constraints)
                        self.__run_DBA(target=full_target, weight=full_weight, lmbda=lmbda, ii=full_ii, jj=full_jj,kk=full_kk, t0=t0, t1=t1, eff_impl=eff_impl_flag)
                        
                    else:
                        # Standard BA for visual-only mode with pre-BA alignment
                        if (self.pg.ii < self.n - self.cfg.REMOVAL_WINDOW - 1).any() and not self.ran_global_ba[self.n]:
                            self.__run_global_BA()
                        else:
                            t0 = self.n - self.cfg.OPTIMIZATION_WINDOW if self.is_initialized else 1
                            t0 = max(t0, 1)
                            fastba.BA(self.poses, self.patches, self.intrinsics, 
                                target, weight, lmbda, self.pg.ii, self.pg.jj, self.pg.kk, t0, self.n, M=self.M, iterations=2, eff_impl=False)
                            
                            self.last_t0 = t0 
                            self.last_t1 = self.n

                except Exception as e:
                    print(f"Warning BA failed: {e}")
                
                # 更新点云
                points = pops.point_cloud(SE3(self.poses), self.patches[:, :self.m], self.intrinsics, self.ix[:self.m])
                points = (points[...,1,1,:3] / points[...,1,1,3:]).reshape(-1, 3)
                self.pg.points_[:len(points)] = points[:]

    # ============================================================================
    # ENHANCED UPDATE: Add RoMeO-style depth constraints after BA
    # ============================================================================
    
    def update_with_romeo_constraints(self):
        """Enhanced update method with post-BA depth regularization (similar to RoMeO)"""
        coords = self.reproject()#进行重投影

        with autocast(enabled=True):
            corr = self.corr(coords) #计算相关性，获取当前帧与上一帧之间的特征匹配信息。
            ctx = self.imap[:, self.pg.kk % (self.M * self.pmem)]
            self.pg.net, (delta, weight, _) = \
                self.network.update(self.pg.net, ctx, corr, None, self.pg.ii, self.pg.jj, self.pg.kk)

        lmbda = torch.as_tensor([1e-4], device="cuda")
        weight = weight.float()
        target = coords[...,self.P//2,self.P//2] + delta.float()

        self.pg.target = target
        self.pg.weight = weight

        # Bundle adjustment进行BA优化 + RoMeO constraints
        with Timer("BA", enabled=self.enable_timing):
            try:
                if self.imu_enabled:#如果使用imu
                    t1=self.n
                    eff_impl_flag=False

                    # 根据不同的情况决定t0，full_target，full_weight，full_ii，full_jj，full_kk，eff_impl_flag
                    if (self.pg.ii < self.n - self.cfg.REMOVAL_WINDOW - 1).any() and not self.ran_global_ba[self.n]:
                        # 如果ii中有小于n-REMOVAL_WINDOW-1的值（就是有回环匹配了），且当前帧没有运行过全局BA优化，则运行全局BA优化
                        eff_impl_flag=True #对于全局BA优化使用高效的实现
                        full_target = torch.cat((self.pg.target_inac, self.pg.target), dim=1)
                        full_weight = torch.cat((self.pg.weight_inac, self.pg.weight), dim=1)
                        full_ii = torch.cat((self.pg.ii_inac, self.pg.ii))
                        full_jj = torch.cat((self.pg.jj_inac, self.pg.jj))
                        full_kk = torch.cat((self.pg.kk_inac, self.pg.kk))

                        t0 = self.pg.ii.min().item()
                        self.ran_global_ba[self.n] = True #标记当前帧已经运行过全局BA优化
                        
                        # Apply BA with gentle RoMeO constraints for global optimization
                        self.__run_DBA_with_depth_constraints(target=full_target, weight=full_weight, lmbda=lmbda, ii=full_ii, jj=full_jj,kk=full_kk, t0=t0, t1=t1, eff_impl=eff_impl_flag)
                        
                    else:#运行局部BA优化
                        t0 = self.n - self.cfg.OPTIMIZATION_WINDOW if self.is_initialized else 1
                        t0 = max(t0, 1)
                        full_target = self.pg.target
                        full_weight = self.pg.weight
                        full_ii =  self.pg.ii
                        full_jj = self.pg.jj
                        full_kk = self.pg.kk
                        
                        eff_impl_flag=False #对于局部BA优化使用低效的实现

                        # Apply BA with RoMeO constraints for local optimization  
                        self.__run_DBA_with_depth_constraints(target=full_target, weight=full_weight, lmbda=lmbda, ii=full_ii, jj=full_jj,kk=full_kk, t0=t0, t1=t1, eff_impl=eff_impl_flag)

                else:
                    #运行全局BA优化
                    # run global bundle adjustment if there exist long-range edges
                    if (self.pg.ii < self.n - self.cfg.REMOVAL_WINDOW - 1).any() and not self.ran_global_ba[self.n]:
                        # 如果ii中有小于n-REMOVAL_WINDOW-1的值（就是有回环匹配了），且当前帧没有运行过全局BA优化，则运行全局BA优化
                        self.__run_global_BA()
                    else:#运行局部BA优化
                        t0 = self.n - self.cfg.OPTIMIZATION_WINDOW if self.is_initialized else 1
                        t0 = max(t0, 1)
                        
                        # Apply standard BA iterations (NO post-BA depth fusion)
                        for ba_iter in range(2):
                            fastba.BA(self.poses, self.patches, self.intrinsics, 
                                target, weight, lmbda, self.pg.ii, self.pg.jj, self.pg.kk, t0, self.n, M=self.M, iterations=1, eff_impl=False)
                        
                        # 额外需要进行记录的
                        self.last_t0 = t0 
                        self.last_t1 = self.n

            except Exception as e:
                print(f"Warning BA failed: {e}")
            
            # 更新点云
            points = pops.point_cloud(SE3(self.poses), self.patches[:, :self.m], self.intrinsics, self.ix[:self.m])
            points = (points[...,1,1,:3] / points[...,1,1,3:]).reshape(-1, 3)
            self.pg.points_[:len(points)] = points[:]

    def __run_DBA_with_depth_constraints(self, target, weight, lmbda, ii, jj, kk, t0, t1, eff_impl):
        """Standard DBA (depth fusion already applied pre-BA in update())"""
        # Run standard DBA - fusion already happened in update()
        self.__run_DBA(target, weight, lmbda, ii, jj, kk, t0, t1, eff_impl)

    # ============================================================================
    # END ENHANCED UPDATE
    # ============================================================================

    def flow_viz_step(self):
        # [DEBUG]
        # dij = (self.ii - self.jj).abs()
        # assert (dij==0).sum().item() == len(torch.unique(self.kk)) 
        # [DEBUG]

        coords_est = pops.transform(SE3(self.poses), self.patches, self.intrinsics, self.ii, self.jj, self.kk) # p_ij (B,close_edges,P,P,2)
        self.flow_data[self.counter-1] = {"ii": self.ii, "jj": self.jj, "kk": self.kk,\
                                          "coords_est": coords_est, "img": self.image_, "n": self.n}

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.imshow(self.image_)
        # plt.show()
                
    def __edges_all(self):
        return flatmeshgrid(
            torch.arange(0, self.m, device="cuda"),
            torch.arange(0, self.n, device="cuda"), indexing='ij')

    def __edges_forw(self):
        r=self.cfg.PATCH_LIFETIME  # default: 13
        t0 = self.M * max((self.n - r), 0)
        t1 = self.M * max((self.n - 1), 0)
        return flatmeshgrid(
            torch.arange(t0, t1, device="cuda"),
            torch.arange(self.n-1, self.n, device="cuda"), indexing='ij')

    def __edges_back(self):
        r=self.cfg.PATCH_LIFETIME  # default: 13
        t0 = self.M * max((self.n - 1), 0)
        t1 = self.M * max((self.n - 0), 0)
        return flatmeshgrid(
            torch.arange(t0, t1, device="cuda"),
            torch.arange(max(self.n-r, 0), self.n, device="cuda"), indexing='ij')
    

    def init_IMU(self):
        """ initialize IMU states """
        cur_t = float(self.tlist[self.pg.tstamps_[self.t0]])#获取当前时间戳,self.t0只是计数上关键帧为0，对应获取全局的时间索引

        # find the first IMU data（在第一帧时间后的第一个IMU数据）
        for i in range(len(self.all_imu)):
            # if self.all_imu[i][0] < cur_t - 1e-6: continue
            if self.all_imu[i][0] < cur_t: continue
            else:
                self.cur_imu_ii = i #记录当前IMU数据的索引
                break

        # 对于t0到t1范围内
        for i in range(self.t0,self.t1):
            tt = self.tlist[self.pg.tstamps_[i]]#获取当前帧的时间戳
            if i == self.t0:#如果是第一帧
                self.state.init_first_state(cur_t,np.zeros(3),\
                                            np.eye(3),\
                                            np.zeros(3))#将状态初始化为0（除了时间）
                # 然后插入IMU数据
                self.state.append_imu(self.all_imu[self.cur_imu_ii][0],\
                                        self.all_imu[self.cur_imu_ii][4:7],\
                                        self.all_imu[self.cur_imu_ii][1:4]/180*math.pi)
                self.cur_imu_ii += 1
                self.is_init = True
            else:
                cur_t = float(self.tlist[self.pg.tstamps_[i]])
                while self.all_imu[self.cur_imu_ii][0] < cur_t: #在当前时间之前的imu数据都加入
                    self.state.append_imu(self.all_imu[self.cur_imu_ii][0],\
                                            self.all_imu[self.cur_imu_ii][4:7],\
                                            self.all_imu[self.cur_imu_ii][1:4]/180*math.pi)
                    self.cur_imu_ii += 1
                self.state.append_imu(cur_t,\
                                            self.all_imu[self.cur_imu_ii][4:7],\
                                            self.all_imu[self.cur_imu_ii][1:4]/180*math.pi)#此处插入一次当前时间以及大于等于当前时间的IMU观测量，应该是为了保证连续性
                self.state.append_img(cur_t)#进行状态的插入及更新（此处与img无关）

                # 下面应该是为了连续性再插入一个IMU数据
                self.state.append_imu(self.all_imu[self.cur_imu_ii][0],\
                                self.all_imu[self.cur_imu_ii][4:7],\
                                self.all_imu[self.cur_imu_ii][1:4]/180*math.pi)
                
                self.cur_imu_ii += 1

            #初始化为相机到IMU的变换矩阵
            Twc = np.matmul(np.array([[1,0,0,0],\
                                     [0,1,0,0],\
                                     [0,0,1,0.02*i],\
                                     [0,0,0,1]]),self.Ti1c) #  perturb the camera poses, which benefits the robustness of initial BA
            TTT = torch.tensor(np.linalg.inv(Twc))#将齐次变换矩阵 Twc 的逆矩阵转换为 PyTorch 张量。也就是Tcw
            q = torch.tensor(Rotation.from_matrix(TTT[:3, :3]).as_quat())#将旋转矩阵转换为四元数
            t = TTT[:3,3]
            if not self.imu_enabled:#如果不使用imu（若为false，初始化为false，所以会执行）
                self.pg.poses_[i] = torch.cat([t,q])#进行赋值
                # gwp_donothing=1


    def __initialize(self):
        """ initialize the DEIO system """
        self.t0 = 0 #起始帧的索引
        self.t1 = self.n #结束帧的索引（应该也就是当前的输入数据的索引）

        # 初始化imu
        self.init_IMU()

        # 下面进行图的更新
        self.imu_enabled = False #此处图的更新不使用imu
        for itr in range(12):
            self.update()

        # # initialization complete标记初始化成功
        self.is_initialized = True # 初始化完成的标志位

    def compute_output_scale_once(self):
        """Compute scale once after VI initialization"""
        if len(self.network.depth_maps) < 5 or not self.imu_enabled:
            return None
        
        # Get depth quality from first 10 frames
        recent_frames = list(self.network.depth_maps.keys())[:10]
        qualities = [
            ((self.network.depth_maps[f] > 1.0) & 
             (self.network.depth_maps[f] < 80.0)).float().mean().item() * 100
            for f in recent_frames
        ]
        depth_quality = np.mean(qualities)
        
        # Get IMU excitation from first 10 frames
        total_rot = sum(
            np.linalg.norm(self.state.preintegrations[i].deltaRij().xyz())
            for i in range(min(10, len(self.state.preintegrations)))
            if i < len(self.state.preintegrations)
        )
        imu_excitation = total_rot / 10
        
        if depth_quality < 50.0:
            print(f"⚠️ Depth quality {depth_quality:.1f}% below threshold")
            return None
        
        scale = (self.scale_formula_a * depth_quality + 
                 self.scale_formula_b * imu_excitation + 
                 self.scale_formula_c)
        
        print(f"✅ Computed output scale: {scale:.3f} (depth_q={depth_quality:.1f}%, imu_exc={imu_excitation:.3f})")
        return scale

    def get_pose_ref(self, tt:float):
        tt_found = self.all_gt_keys[bisect.bisect(self.all_gt_keys,tt)]
        return tt_found, self.all_gt[tt_found]
    
    def VisualIMUAlignment(self, t0, t1, ignore_lever, disable_scale = True): #warda
    
        if disable_scale is None:
            disable_scale = getattr(self, 'disable_scale_in_vi_alignment', True)
        poses = SE3(self.pg.poses_)
        wTcs = poses.inv().matrix().cpu().numpy()

        if not ignore_lever:
            wTbs = np.matmul(wTcs,self.Tbc.inverse().matrix())
        else:
            T_tmp = self.Tbc.inverse().matrix()
            T_tmp[0:3,3] = 0.0
            wTbs = np.matmul(wTcs,T_tmp)
        cost = 0.0

        # solveGyroscopeBias
        A = np.zeros([3,3])
        b = np.zeros(3)
        H1 =np.zeros([15,6], order='F', dtype=np.float64)
        H2 =np.zeros([15,3], order='F', dtype=np.float64)
        H3 =np.zeros([15,6], order='F', dtype=np.float64)
        H4 =np.zeros([15,3], order='F', dtype=np.float64)
        H5 =np.zeros([15,6], order='F', dtype=np.float64) # navstate wrt. bias
        H6 =np.zeros([15,6], order='F', dtype=np.float64)
        for i in range(t0,t1-1):
            pose_i = gtsam.Pose3(wTbs[i])
            pose_j = gtsam.Pose3(wTbs[i+1])
            Rij = np.matmul(pose_i.rotation().matrix().T,pose_j.rotation().matrix())
            imu_factor = gtsam.gtsam.CombinedImuFactor(0,1,2,3,4,5,self.state.preintegrations[i])
            err = imu_factor.evaluateErrorCustom(pose_i,self.state.vs[i],\
                                                 pose_j,self.state.vs[i+1],\
                self.state.bs[i],self.state.bs[i+1],\
                    H1,H2,H3,H4,H5,H6)
            tmp_A = H5[0:3,3:6]
            tmp_b = err[0:3]
            cost +=  np.dot(tmp_b,tmp_b)
            A += np.matmul(tmp_A.T,tmp_A)
            b += np.matmul(tmp_A.T,tmp_b)
        bg = -np.matmul(np.linalg.inv(A),b)

        for i in range(0,t1-1):
            pim = gtsam.PreintegratedCombinedMeasurements(self.state.params,\
                  gtsam.imuBias.ConstantBias(np.array([.0,.0,.0]),bg))
            for iii in range(len(self.state.preintegrations_meas[i])):
                dd = self.state.preintegrations_meas[i][iii]
                if dd[2] > 0: pim.integrateMeasurement(dd[0],dd[1],dd[2])
            self.state.preintegrations[i] = pim
            self.state.bs[i] = gtsam.imuBias.ConstantBias(np.array([.0,.0,.0]),bg)
        print('bg: ',bg)
        
        # linearAlignment
        all_frame_count = t1 - t0
        n_state = all_frame_count * 3 + 3 + 1
        A = np.zeros([n_state,n_state])
        b = np.zeros(n_state)
        i_count = 0
        for i in range(t0,t1-1):
            pose_i = gtsam.Pose3(wTbs[i])
            pose_j = gtsam.Pose3(wTbs[i+1])
            R_i = pose_i.rotation().matrix()
            t_i = pose_i.translation()
            R_j = pose_j.rotation().matrix()
            t_j = pose_j.translation()
            pim = self.state.preintegrations[i]
            tic = self.Tbc.translation()

            tmp_A = np.zeros([6,10])
            tmp_b = np.zeros(6)
            dt = pim.deltaTij()
            tmp_A[0:3,0:3] = -dt * np.eye(3,3)
            tmp_A[0:3,6:9] = R_i.T * dt * dt / 2
            tmp_A[0:3,9] = np.matmul(R_i.T, t_j-t_i) / 100.0
            tmp_b[0:3] = pim.deltaPij()
            tmp_A[3:6,0:3] = -np.eye(3,3)
            tmp_A[3:6,3:6] = np.matmul(R_i.T, R_j)
            tmp_A[3:6,6:9] = R_i.T * dt
            tmp_b[3:6] = pim.deltaVij()

            r_A = np.matmul(tmp_A.T,tmp_A)
            r_b = np.matmul(tmp_A.T,tmp_b)

            A[i_count*3:i_count*3+6,i_count*3:i_count*3+6] += r_A[0:6,0:6]
            b[i_count*3:i_count*3+6] += r_b[0:6]
            A[-4:,-4:] += r_A[-4:,-4:]
            b[-4:] += r_b[-4:]
            
            A[i_count*3:i_count*3+6,n_state-4:] += r_A[0:6,-4:]
            A[n_state-4:,i_count*3:i_count*3+6] += r_A[-4:,0:6]
            i_count += 1
        
        A = A * 1000.0
        b = b * 1000.0
        x = np.matmul(np.linalg.inv(A),b)
        s = x[n_state-1] / 100.0

        g = x[-4:-1]

        # RefineGravity
        g0 = g / np.linalg.norm(g) * 9.81
        lx = np.zeros(3)
        ly = np.zeros(3)
        n_state = all_frame_count * 3 + 2 + 1
        A = np.zeros([n_state,n_state])
        b = np.zeros(n_state)

        for k in range(4):
            aa = g / np.linalg.norm(g)
            tmp = np.array([.0,.0,1.0])

            bb = (tmp - np.dot(aa,tmp) * aa)
            bb /= np.linalg.norm(bb)
            cc = np.cross(aa,bb)
            bc = np.zeros([3,2])
            bc[0:3,0] = bb
            bc[0:3,1] = cc
            lxly = bc
            
            i_count = 0
            for i in range(t0,t1-1):
                pose_i = gtsam.Pose3(wTbs[i])
                pose_j = gtsam.Pose3(wTbs[i+1])
                R_i = pose_i.rotation().matrix()
                t_i = pose_i.translation()
                R_j = pose_j.rotation().matrix()
                t_j = pose_j.translation()
                tmp_A = np.zeros([6,9])
                tmp_b = np.zeros(6)
                pim = self.state.preintegrations[i]
                dt = pim.deltaTij()

                tmp_A[0:3,0:3] = -dt *np.eye(3,3)
                tmp_A[0:3,6:8] = np.matmul(R_i.T,lxly) * dt * dt /2 
                tmp_A[0:3,8]   = np.matmul(R_i.T,t_j - t_i) / 100.0
                tmp_b[0:3] = pim.deltaPij() - np.matmul(R_i.T,g0) * dt * dt / 2

                tmp_A[3:6,0:3] = -np.eye(3)
                tmp_A[3:6,3:6] = np.matmul(R_i.T,R_j)
                tmp_A[3:6,6:8] = np.matmul(R_i.T,lxly) * dt
                tmp_b[3:6] = pim.deltaVij() - np.matmul(R_i.T,g0) * dt

                r_A = np.matmul(tmp_A.T,tmp_A)
                r_b = np.matmul(tmp_A.T,tmp_b)

                A[i_count*3:i_count*3+6,i_count*3:i_count*3+6] += r_A[0:6,0:6]
                b[i_count*3:i_count*3+6] += r_b[0:6]
                A[-3:,-3:] += r_A[-3:,-3:]
                b[-3:] += r_b[-3:]

                A[i_count*3:i_count*3+6,n_state-3:] += r_A[0:6,-3:]
                A[n_state-3:,i_count*3:i_count*3+6] += r_A[-3:,0:6]
                i_count += 1
            
            A = A * 1000.0
            b = b * 1000.0
            x = np.matmul(np.linalg.inv(A),b)
            dg = x[-3:-1]
            g0 = g0 + np.matmul(lxly,dg)
            g0 = g0 / np.linalg.norm(g0) * 9.81
            s = x[-1] / 100.0
        print(s,g0,x)

        if disable_scale:
            s = 1.0
            
        # print('g,s:',g,s)
        print(f'\033[31m the calculate g {g} and scaler {s} \033[0m ')
        if math.fabs(np.linalg.norm(g) - 9.81) < 0.5 and s > 0:
            print('V-I successfully initialized!')
        
        # visualInitialAlign
        wTbs[:,0:3,3] *= s # !!!!!!!!!!!!!!!!!!!!!!!!
        for i in range(0, t1-t0):
            self.state.vs[i+t0] = np.matmul(wTbs[i+t0,0:3,0:3],x[i*3:i*3+3])
        
        # g2R
        ng1 = g0/ np.linalg.norm(g0)
        ng2 = np.array([0,0,1.0])
        R0 = trans.FromTwoVectors(ng1,ng2)
        yaw = trans.R2ypr(R0)[0]
        R0 = np.matmul(trans.ypr2R(np.array([-yaw,0,0])),R0)

        # align for visualization
        ppp =  np.matmul(R0,wTbs[t1-1,0:3,3])
        RRR =  np.matmul(R0,wTbs[t1-1,0:3,0:3])

        if self.all_gt is not None: # align the initial poses for visualization
            tt_found,dd = self.get_pose_ref(self.tlist[self.pg.tstamps_[t1-1]]-1e-3)
            self.refTw = np.matmul(dd['T'],np.linalg.inv(wTbs[t1-1]))
            self.refTw[0:3,0:3] = trans.att2m([0,0,trans.m2att(self.refTw[0:3,0:3])[2]])

        g = np.matmul(R0,g0)
        for i in range(0,t1):
            wTbs[i,0:3,3] = np.matmul(R0,wTbs[i,0:3,3])
            wTbs[i,0:3,0:3] = np.matmul(R0,wTbs[i,0:3,0:3])
            self.state.vs[i] = np.matmul(R0, self.state.vs[i])
            self.state.wTbs[i] = gtsam.Pose3(wTbs[i])

        self.vi_init_t1 = t1
        self.vi_init_time = self.tlist[self.pg.tstamps_[t1-1]]

        if not ignore_lever:
            wTcs = np.matmul(wTbs,self.Tbc.matrix())
        else:
            T_tmp = self.Tbc.matrix()
            T_tmp[0:3,3] = 0.0
            wTcs = np.matmul(wTbs,T_tmp)
    
        for i in range(0,t1):
            TTT = np.linalg.inv(wTcs[i])
            q = torch.tensor(Rotation.from_matrix(TTT[:3, :3]).as_quat())
            t = torch.tensor(TTT[:3,3])
            self.pg.poses_[i] = torch.cat([t,q])
            # self.disps[i] /= s
            
            print(f"🔍 VI Alignment computed scale s={s:.3f}")
            self.patches_gt_[i,:,2] /= s #对所有的patch的深度进行改写
            self.pg.patches_[i,:,2] /= s
        
        # # 对于所有非关键帧的pose
        # s = torch.tensor(s).to(dtype=self.pg.poses_.dtype, device=self.pg.poses_.device)
        # for t, (t0, dP) in self.pg.delta.items():
        #     self.pg.delta[t] = (t0, dP.scale(s))
    
    def init_VI(self):
        """ initialize the V-I system, referring to VIN-Fusion """
        sum_g = np.zeros(3,dtype = np.float64)
        ccount = 0
        for i in range(self.t1 - 8 ,self.t1-1):
            dt = self.state.preintegrations[i].deltaTij()
            tmp_g = self.state.preintegrations[i].deltaVij()/dt
            sum_g += tmp_g
            ccount += 1
        aver_g = sum_g * 1.0 / ccount
        var_g = 0.0
        for i in range(self.t1 - 8 ,self.t1-1):
            dt = self.state.preintegrations[i].deltaTij()
            tmp_g = self.state.preintegrations[i].deltaVij()/dt
            var_g += np.linalg.norm(tmp_g - aver_g)**2
        var_g =math.sqrt(var_g/ccount)
        if var_g < 0.25: #若方差小于0.25,就证明IMU的激励不够
            print("IMU excitation not enough!",var_g)
        else:
            poses = SE3(self.pg.poses_)
            self.plt_pos = [[],[]]
            self.plt_pos_ref = [[],[]]
            for i in range(0,self.t1):
                ppp = np.matmul(poses[i].cpu().inv().matrix(),np.linalg.inv(self.Ti1c))[0:3,3]
                self.plt_pos[0].append(ppp[0])
                self.plt_pos[1].append(ppp[1])
                if self.all_gt is not None:#如果有gt数据此处就会用：
                    tt_found,dd = self.get_pose_ref(self.tlist[self.pg.tstamps_[i]]-1e-3)
                    self.plt_pos_ref[0].append(dd['T'][0,3])
                    self.plt_pos_ref[1].append(dd['T'][1,3])     

            if not self.visual_only:#如果有IMU
                self.VisualIMUAlignment(self.t1 - 8 ,self.t1, ignore_lever= True)
                self.update()#更新图
                self.VisualIMUAlignment(self.t1 - 8 ,self.t1, ignore_lever= False)
                self.update()#更新图
                self.VisualIMUAlignment(self.t1 - 8 ,self.t1, ignore_lever= False)
                self.imu_enabled = True #完成视觉惯性对齐后再开启IMU(之后BA update中才会用到imu了~)
            else:#下面不用管
                # 报错
                raise ValueError("Visual only initialization in init_VI???")
                self.visual_only_init = True #只用视觉，不用imu

            self.set_prior(self.last_t0,self.t1)

            self.plt_pos = [[],[]]
            self.plt_pos_ref = [[],[]]
            for i in range(0,self.t1):
                TTT = self.state.wTbs[i].matrix()
                ppp = TTT[0:3,3]
                qqq = Rotation.from_matrix(TTT[:3, :3]).as_quat()
                # 结果写入
                # self.result_file.writelines('%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n'%(self.tlist[i],ppp[0],ppp[1],ppp[2]\
                #                             ,qqq[0],qqq[1],qqq[2],qqq[3]))
                self.poses_save.append([self.tlist[i],ppp[0],ppp[1],ppp[2],qqq[0],qqq[1],qqq[2],qqq[3]])#将当前的位姿保存到列表中， #x,y,z xyzw
                
                TTTref = np.matmul(self.refTw,TTT) # for visualization
                ppp = TTTref[0:3,3]
                qqq = Rotation.from_matrix(TTTref[:3, :3]).as_quat()
                self.plt_pos[0].append(ppp[0])
                self.plt_pos[1].append(ppp[1])
                if self.all_gt is not None:
                    tt_found,dd = self.get_pose_ref(self.tlist[self.pg.tstamps_[i]]-1e-3)
                    self.plt_pos_ref[0].append(dd['T'][0,3])
                    self.plt_pos_ref[1].append(dd['T'][1,3])

            for itr in range(1):
                self.update()

    def VIO_update(self):
        """ perform VIO/EIO update """
        self.t1 = self.n

        # 如果有IMU数据，且当前的时间与self.video.vi_init_time的时间差大于5s，就重置reinit为True
        if self.imu_enabled and (self.tlist[self.pg.tstamps_[self.t1-1]] - self.vi_init_time > 5.0):
            self.reinit = True
            self.vi_init_time = 1e9

        ## new frame comes, append IMU（插入imu信息）
        cur_t = float(self.tlist[self.pg.tstamps_[self.t1-1]])#获取当前图像时间戳

        while self.all_imu[self.cur_imu_ii][0] < cur_t:#若当前imu的索引时间小于当前时间
            # 插入IMU数据
            self.state.append_imu(self.all_imu[self.cur_imu_ii][0],\
                                    self.all_imu[self.cur_imu_ii][4:7],\
                                    self.all_imu[self.cur_imu_ii][1:4]/180*math.pi)
            self.cur_imu_ii += 1 #更新IMU索引，一直加入

        # 插入imu数据（此时self.all_imu[self.cur_imu_ii][0]>=cur_t，但是插入时间cur_t及下一帧的IMU数据似乎可以保证时间的连续性）
        self.state.append_imu(cur_t,\
                                    self.all_imu[self.cur_imu_ii][4:7],\
                                    self.all_imu[self.cur_imu_ii][1:4]/180*math.pi)
        self.state.append_img(cur_t)#插入完IMU后更新一下状态

        # 为了确保连续性，再次插入imu数据，保证次数IMU中数据必然大于等于当前时间戳
        self.state.append_imu(self.all_imu[self.cur_imu_ii][0],\
                        self.all_imu[self.cur_imu_ii][4:7],\
                        self.all_imu[self.cur_imu_ii][1:4]/180*math.pi)
        self.cur_imu_ii += 1

        ## predict pose (<5 ms)
        if self.imu_enabled:#如果使用了imu
            Twc = (self.state.wTbs[-1] * self.Tbc).matrix()#最新的Twb*Tbc
            TTT = torch.tensor(np.linalg.inv(Twc))#获取它的逆，也就是Tcw
            q = torch.tensor(Rotation.from_matrix(TTT[:3, :3]).as_quat())#获取四元数
            t = TTT[:3,3]
            self.pg.poses_[self.t1-1] = torch.cat([t,q])#对当前的pose进行初始化

        # Choose which update method to use based on stereo availability

        self.update()#进行更新操作

        # 将pose结果保存输出
        poses = SE3(self.pg.poses_)#获取pose
        TTT = np.matmul(poses[self.t1-1].cpu().inv().matrix(),np.linalg.inv(self.Ti1c))#获取最新一帧并且转换为body frame下
        # 若使用了imu或者只使用了视觉并且已经初始化了
        if self.imu_enabled or (self.visual_only and self.visual_only_init):
        
            ppp = TTT[0:3,3]
            ppp = ppp*1.0 #warda
            qqq = Rotation.from_matrix(TTT[:3, :3]).as_quat()
            self.poses_save.append([cur_t,ppp[0],ppp[1],ppp[2],qqq[0],qqq[1],qqq[2],qqq[3]])#将当前的位姿保存到列表中， #x,y,z xyzw

        self.keyframe()#关键帧的管理
        self.t1=self.n#更新t1,因为前面关键帧处可能受到影响了？（此处应该主要影响初始化？？？）

        ## 尝试视觉惯性初始化.try initializing VI（vi_warmup为视觉初始化的帧数为12）
        if self.t1 > self.vi_warmup and self.vi_init_t1 < 0: #帧数大于12且初始化时间小于0
            if self.visual_only==1: #不使用IMU，这是传入的参数
                self.visual_only_init = True
            else:
                self.init_VI() #只有当使用了IMU的时候才会进行视觉初始化

        #结束当前帧的处理

    def __call__(self, tstamp, image, intrinsics, scale=1.0, psmnet_depth=None):
        """ track new voxel frame - ENHANCED with stereo depth integration """

        # ============================================================================
        # STEREO DEPTH INTEGRATION: Store current stereo depth
        # ============================================================================
        self.current_psmnet_depth = psmnet_depth

        if self.cfg.CLASSIC_LOOP_CLOSURE:#如果开启了经典的闭环检测（就是图像匹配）
            self.long_term_lc(image, self.n)

        if (self.n+1) >= self.N:
            raise Exception(f'The buffer size is too small. You can increase it using "--buffer {self.N*2}"')

        if self.viewer is not None:
            self.viewer.update_image(image)

        if self.viz_flow:
            self.image_ = image.detach().cpu().permute((1, 2, 0)).numpy()

        if not self.evs:#如果不使用事件,就是正常的图像归一化操作
            image = 2 * (image[None,None] / 255.0) - 0.5 
        else:
            image = image[None,None]
            
            if self.n == 0:
                nonzero_ev = (image != 0.0)
                zero_ev = ~nonzero_ev
                num_nonzeros = nonzero_ev.sum().item()
                num_zeros = zero_ev.sum().item()
                # [DEBUG]
                # print("nonzero-zero-ratio", num_nonzeros, num_zeros, num_nonzeros / (num_zeros + num_nonzeros))
                if num_nonzeros / (num_zeros + num_nonzeros) < 2e-2: # TODO eval hyperparam (add to config.py)
                    print(f"skip voxel at {tstamp} due to lack of events!")
                    return

            b, n, v, h, w = image.shape
            flatten_image = image.view(b,n,-1)
            
            if self.cfg.NORM.lower() == 'none':
                pass
            elif self.cfg.NORM.lower() == 'rescale' or self.cfg.NORM.lower() == 'norm':
                # Normalize (rescaling) neg events into [-1,0) and pos events into (0,1] sequence-wise
                # Preserve pos-neg inequality (quantity only)
                pos = flatten_image > 0.0
                neg = flatten_image < 0.0
                vx_max = torch.Tensor([1]).to("cuda") if pos.sum().item() == 0 else flatten_image[pos].max(dim=-1, keepdim=True)[0]
                vx_min = torch.Tensor([1]).to("cuda") if neg.sum().item() == 0 else flatten_image[neg].min(dim=-1, keepdim=True)[0]
                # [DEBUG]
                # print("vx_max", vx_max.item())
                # print("vx_min", vx_min.item())
                if vx_min.item() == 0.0 or vx_max.item() == 0.0:
                    # no information for at least one polarity
                    print(f"empty voxel at {tstamp}!")
                    return
                flatten_image[pos] = flatten_image[pos] / vx_max
                flatten_image[neg] = flatten_image[neg] / -vx_min
            elif self.cfg.NORM.lower() == 'standard' or self.cfg.NORM.lower() == 'std':
                # Data standardization of events only
                # Does not preserve pos-neg inequality
                # see https://github.com/uzh-rpg/rpg_e2depth/blob/master/utils/event_tensor_utils.py#L52
                nonzero_ev = (flatten_image != 0.0)
                num_nonzeros = nonzero_ev.sum(dim=-1)
                if torch.all(num_nonzeros > 0):
                    # compute mean and stddev of the **nonzero** elements of the event tensor
                    # we do not use PyTorch's default mean() and std() functions since it's faster
                    # to compute it by hand than applying those funcs to a masked array

                    mean = torch.sum(flatten_image, dim=-1, dtype=torch.float32) / num_nonzeros  # force torch.float32 to prevent overflows when using 16-bit precision
                    stddev = torch.sqrt(torch.sum(flatten_image ** 2, dim=-1, dtype=torch.float32) / num_nonzeros - mean ** 2)
                    mask = nonzero_ev.type_as(flatten_image)
                    flatten_image = mask * (flatten_image - mean[...,None]) / stddev[...,None]
            else:
                print(f"{self.cfg.NORM} not implemented")
                raise NotImplementedError

            image = flatten_image.view(b,n,v,h,w)

        if image.shape[-1] == 346:
            image = image[..., 1:-1] # hack for MVSEC, FPV,...

        # TODO patches with depth is available (val)
        with autocast(enabled=self.cfg.MIXED_PRECISION):
            if not self.evs:  # VONet (images)
                fmap, gmap, imap, patches, _, clr = \
                    self.network.patchify(image,
                        patches_per_image=self.cfg.PATCHES_PER_FRAME, 
                        return_color=True,
                        centroid_sel_strat=self.cfg.CENTROID_SEL_STRAT)  # Use VONet parameters
            else:  # eVONet (events)
                fmap, gmap, imap, patches, _, clr = \
                    self.network.patchify(image,
                        patches_per_image=self.cfg.PATCHES_PER_FRAME, 
                        return_color=True,
                        scorer_eval_mode=self.cfg.SCORER_EVAL_MODE,
                        scorer_eval_use_grid=self.cfg.SCORER_EVAL_USE_GRID)
        
        ### update state attributes ###
        self.tlist.append(tstamp)#时间戳，全局时间的时间戳
        self.pg.tstamps_[self.n] = self.counter#只是数字，统计的为关键帧对应的全局时间的索引
        self.pg.intrinsics_[self.n] = intrinsics / self.RES
        
        # color info for visualization
        if not self.evs:
            clr = (clr[0,:,[2,1,0]] + 0.5) * (255.0 / 2)
            self.pg.colors_[self.n] = clr.to(torch.uint8)
        else:
            clr = (clr[0,:,[0,0,0]] + 0.5) * (255.0 / 2)
            self.pg.colors_[self.n] = clr.to(torch.uint8)
            

        self.pg.index_[self.n + 1] = self.n + 1
        self.pg.index_map_[self.n + 1] = self.m + self.M 

        if self.n > 1:
            if self.cfg.MOTION_MODEL == 'DAMPED_LINEAR':
                P1 = SE3(self.pg.poses_[self.n-1])
                P2 = SE3(self.pg.poses_[self.n-2])

                # To deal with varying camera hz
                *_, a,b,c = [1]*3 + self.tlist
                fac = (c-b) / (b-a)

                xi = self.cfg.MOTION_DAMPING * fac * (P1 * P2.inv()).log()
                tvec_qvec = (SE3.exp(xi) * P1).data
                self.pg.poses_[self.n] = tvec_qvec
            else:
                tvec_qvec = self.poses[self.n-1]
                self.pg.poses_[self.n] = tvec_qvec

        # ============================================================================
        # DEPTH HANDLING: Use network methods (network owns all depth logic)
        # ============================================================================
        
        psmnet_success = False
        
        # If PSMNet depth available, pass to network
        if psmnet_depth is not None:
            try:
                # CRITICAL FIX: Pass patches so network can sample depths at patch locations
                # This stores depths at ORIGINAL patch locations (frozen during BA)
                self.network.add_depth_map(self.n, psmnet_depth, patches)
                
                # Network handles initialization (frames 0-7)
                if not self.is_initialized:
                    patches, psmnet_success = self.network.initialize_from_depth(patches, psmnet_depth)
                    if psmnet_success:
                        print(f"🎯 Frame {self.n}: Network initialized with PSMNet metric depths!")
                
            except Exception as e:
                print(f"❌ Frame {self.n}: PSMNet error: {e}")
        
        # Fallback to original DBA initialization if PSMNet failed or not available
        if not psmnet_success:
            # Original DBA depth initialization
            patches[:,:,2] = torch.rand_like(patches[:,:,2,0,0,None,None])
            if self.is_initialized:
                s = torch.median(self.pg.patches_[self.n-3:self.n,:,2])
                patches[:,:,2] = s
                if self.n % 50 == 0:  # Periodic logging
                    estimated_depth = 1.0 / s.item() if s.item() > 0 else float('inf')
                    print(f"📝 Frame {self.n}: Using median depth from previous frames: {estimated_depth:.2f}m")

        # ============================================================================
        # END DEPTH HANDLING
        # ============================================================================

        self.pg.patches_[self.n] = patches

        ### update network attributes ###
        self.imap_[self.n % self.pmem] = imap.squeeze()
        self.gmap_[self.n % self.pmem] = gmap.squeeze()
        self.fmap1_[:, self.n % self.mem] = F.avg_pool2d(fmap[0], 1, 1)
        self.fmap2_[:, self.n % self.mem] = F.avg_pool2d(fmap[0], 4, 4)

        self.counter += 1

        if self.n > 0 and not self.is_initialized:#视觉还没初始化
            thres = 2.0 if scale == 1.0 else scale ** 2 # TODO adapt thres for lite version
            if self.motion_probe() < thres: # TODO: replace by 8 pixels flow criterion (as described in 3.3 Initialization)
                self.pg.delta[self.counter - 1] = (self.counter - 2, Id[0])
                return

        self.n += 1 # add one (key)frame
        self.m += self.M # add patches per (key)frames to patch number

        if self.cfg.LOOP_CLOSURE: #如果开启了闭环检测（这应该是DPVO中实现的闭环检测）
            if self.n - self.last_global_ba >= self.cfg.GLOBAL_OPT_FREQ:
                """ Add loop closure factors """
                lii, ljj = self.pg.edges_loop() #获取闭环检测的边
                if lii.numel() > 0:
                    self.last_global_ba = self.n #标记上一次全局BA优化的帧数，用于控制全局BA优化的频率
                    self.append_factors(lii, ljj)#添加闭环检测的边

        # relative pose
        self.append_factors(*self.__edges_forw())
        self.append_factors(*self.__edges_back())

        if self.n == 8 and not self.is_initialized:#还没初始化且满足8帧
            self.__initialize()#进行视觉与惯性的初始化         
        elif self.is_initialized:
            self.VIO_update()#进行VIO的更新,同时包含了上面两个函数

        if self.cfg.CLASSIC_LOOP_CLOSURE:#如果开启了经典的闭环检测
            self.long_term_lc.attempt_loop_closure(self.n)#尝试进行闭环检测
            self.long_term_lc.lc_callback()

        if self.viz_flow:
            self.flow_viz_step()
