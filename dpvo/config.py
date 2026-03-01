from yacs.config import CfgNode as CN
_C = CN()

# Existing DPVO parameters
_C.BUFFER_SIZE = 4096
_C.CENTROID_SEL_STRAT = 'RANDOM'
_C.PATCHES_PER_FRAME = 80
_C.REMOVAL_WINDOW = 20
_C.OPTIMIZATION_WINDOW = 12
_C.PATCH_LIFETIME = 12
_C.KEYFRAME_INDEX = 4
_C.KEYFRAME_THRESH = 12.5
_C.MOTION_MODEL = 'DAMPED_LINEAR'
_C.MOTION_DAMPING = 0.5
_C.MIXED_PRECISION = True
_C.LOOP_CLOSURE = False
_C.BACKEND_THRESH = 64.0
_C.MAX_EDGE_AGE = 1000
_C.GLOBAL_OPT_FREQ = 15
_C.CLASSIC_LOOP_CLOSURE = False
_C.LOOP_CLOSE_WINDOW_SIZE = 3
_C.LOOP_RETR_THRESH = 0.04

# Add missing IMU parameters
_C.ENALBE_IMU = True  # Enable IMU
_C.ENALBE_INV = False
_C.Ti1c = [1.0, 0.0, 0.0, 0.0, 
           0.0, 1.0, 0.0, 0.0, 
           0.0, 0.0, 1.0, 0.0, 
           0.0, 0.0, 0.0, 1.0]  # Identity matrix as default
_C.accel_noise_sigma = 2.0000e-3
_C.gyro_noise_sigma = 1.6968e-04
_C.accel_bias_sigma = 3.0000e-3
_C.gyro_bias_sigma = 1.9393e-05

# Add missing parameters that might be needed
_C.NORM = 'std'
_C.PATCH_SELECTOR = 'scorer'
_C.SCORER_EVAL_MODE = 'multi'
_C.SCORER_EVAL_USE_GRID = True
_C.GRADIENT_BIAS = False

cfg = _C



# Logging controls (add to your config)
_C.LOG_TRAIN_DATA = True            # turn on/off logging
_C.LOG_DIR = "./train_logs"         # where to save .npz shards
_C.LOG_EVERY = 1                    # log every N frames (1 = every frame)
_C.LOG_MAX_DEPTH_MAPS = 30          # keep last K depth maps in CPU ring buffer
_C.LOG_COMPRESS = True              # np.savez_compressed vs savez
_C.SEQ_NAME = "kitti_seq"           # optional; used in filenames

# cfg defaults
_C.ENABLE_DEPTH_INIT   = True   # allow injecting depth into patches during init
_C.ENABLE_DEPTH_PRIORS = True   # allow RoMeO-style depth regularization in BA
_C.base_strength = 0.08

