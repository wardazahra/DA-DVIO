# DA-DVIO: Depth-Aware Deep Visual-Inertial Odometry

DA-DVIO is a multi-modal sensor fusion framework that extends [DPVO](https://github.com/princeton-vl/DPVO) (Deep Patch Visual Odometry) with stereo depth supervision and IMU integration. The system addresses scale drift in monocular visual odometry by incorporating PSMNet-generated stereo depth priors during training and runtime IMU measurements for metric-scale trajectory estimation. Evaluated on KITTI odometry sequences and ZED camera recordings.

---

## Overview

| Component | Description |
|---|---|
| Base network | DPVO (Deep Patch Visual Odometry) |
| Depth supervision | PSMNet stereo depth priors (training + runtime blending) |
| IMU integration | Factor graph optimization via GTSAM (from DEIO) |
| Training sequences | KITTI 00, 05, 07 |
| Test sequences | KITTI 08, 09, ZED z02, z08, z11 |

---

## Installation

DA-DVIO builds on top of two separate codebases: **DPVO** (for the visual odometry backbone and training) and **DEIO** (for IMU integration). Each requires its own conda environment.

### Prerequisites

- Ubuntu 20.04 or 22.04
- CUDA 11 or 12
- CuDNN
- Anaconda/Miniconda

---

### Environment 1: DPVO (Visual Odometry + Training)

This environment is used for training DA-DVIO and running visual-only inference.

```bash
# Clone DPVO
git clone https://github.com/princeton-vl/DPVO.git --recursive
cd DPVO

# Create and activate conda environment
conda env create -f environment.yml
conda activate dpvo

# Install Eigen (required for bundle adjustment)
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
unzip eigen-3.4.0.zip -d thirdparty

# Install DPVO package
pip install .

# Download pretrained models (~2GB)
./download_models_and_data.sh
```

> **Note:** The pretrained `dpvo.pth` checkpoint is required as the initialization point for DA-DVIO training.

---

### Environment 2: DEIO (IMU Integration)

This environment is used for runtime IMU fusion. DEIO requires GTSAM with Python bindings.

```bash
# Clone DEIO
git clone https://github.com/arclab-hku/DEIO.git --recursive
cd DEIO

# Create conda environment (Python 3.10)
conda create -n DEIO python=3.10
conda activate DEIO

# Install PyTorch (adjust cuda version as needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install GTSAM with Python bindings
# (requires cmake, build-essential)
cd thirdparty/gtsam
mkdir build && cd build
cmake .. -DGTSAM_BUILD_PYTHON=1 -DGTSAM_PYTHON_VERSION=3.10
make -j8
make python-install
cd ../../..

# Install remaining dependencies
pip install -e .
```

> **Note:** GTSAM compilation can take 15–30 minutes. Make sure `cmake >= 3.22` and `gcc >= 11` are installed.

---

### Additional Dependencies (both environments)

```bash
pip install evo          # trajectory evaluation
pip install opencv-python
pip install matplotlib numpy scipy
```

---

## Dataset Preparation

### KITTI Odometry

Download the KITTI odometry dataset from the [official website](https://www.cvlibs.net/datasets/kitti/eval_odometry.php). You will need:
- Color images (left camera)
- Ground truth poses
- Calibration files

### ZED Custom Dataset

ZED dataset can be downloaded from kaggle in preprocessed form
https://www.kaggle.com/datasets/wardazahra/ntust-zed-dataset

Expected structure:
```
kitti_odometry/
├── sequences/
│   ├── 00/
│   │   ├── image_2/
│   │   ├── calib.txt
│   │   └── times.txt
│   ├── 05/
│   ├── 07/
│   ├── 08/
│   └── 09/
└── poses/
    ├── 00.txt
    └── ...
```

### PSMNet Depth Maps

Pre-generate stereo depth maps using [PSMNet](https://github.com/JiaRenChang/PSMNet) for all training and test sequences and store them alongside the image sequences.

---

## Training

Training is done in the `dpvo` conda environment.

```bash
conda activate dpvo

python train_fixed_final_corrected.py \
    --name da_dvio_v1 \
    --ckpt /path/to/dpvo.pth \
    --datapath /path/to/kitti_odometry \
    --psmnet_dir /path/to/psmnet_depths \
    --steps 2500 \
    --lr 2e-5 \
    --pose_weight 10.0 \
    --flow_weight 0.1 \
    --depth_weight 1.0 \
    --scale_weight 0.3
```

Key training arguments:

| Argument | Default | Description |
|---|---|---|
| `--ckpt` | None | Path to pretrained DPVO checkpoint |
| `--steps` | 2500 | Total training steps |
| `--lr` | 2e-5 | Learning rate |
| `--pose_weight` | 10.0 | Weight for pose loss |
| `--flow_weight` | 0.1 | Weight for flow loss |
| `--depth_weight` | 1.0 | Weight for depth loss |
| `--scale_weight` | 0.3 | Weight for scale consistency loss |
| `--val_freq` | 100 | Validation every N steps |
| `--patience` | 5 | Early stopping patience |

Checkpoints are saved every 100 steps to `checkpoints/`. The best model (lowest validation loss) is saved as `checkpoints/<name>_best.pth`.

---

## Evaluation

### KITTI (Visual Only — dpvo environment)

```bash
conda activate dpvo
python evaluate_kitti.py \
    --network checkpoints/da_dvio_v1_best.pth \
    --datapath /path/to/kitti_odometry \
    --trials 1 \
    --plot \
    --save_trajectory
```

### KITTI with IMU (DEIO environment)

```bash
conda activate DEIO
python eval_da_dvio_kitti.py \
    --network checkpoints/da_dvio_v1_best.pth \
    --datapath /path/to/kitti_odometry \
    --imu_datapath /path/to/kitti_raw \
    --plot \
    --save_trajectory
```

---

## Results

### KITTI Odometry (ATE RMSE, meters) — SE(3) Alignment, No Scale Correction

| Method | Seq 08 | Seq 09 |
|---|---|---|
| DPVO Visual Only | — | — |
| DPVO + IMU | — | — |
| DA-DVIO (Ours) | — | — |

> Results to be filled after final evaluation.

---

## Repository Structure

```
DA-DVIO/
├── train_fixed_final_corrected.py   # Main training script
├── dpvo/                            # DPVO backbone (from DPVO repo)
├── checkpoints/                     # Saved model checkpoints
├── results/                         # Trajectory outputs
└── README.md
```

---

## Acknowledgements

This work builds upon:
- [DPVO](https://github.com/princeton-vl/DPVO) — Deep Patch Visual Odometry (Teed et al., NeurIPS 2023)
- [DEIO](https://github.com/arclab-hku/DEIO) — Deep Event Inertial Odometry (Guan et al., ICCV 2025)
- [PSMNet](https://github.com/JiaRenChang/PSMNet) — Pyramid Stereo Matching Network

---

## Citation

If you use this work, please cite:

```bibtex
@mastersthesis{warda2025dadvio,
  title     = {DA-DVIO: Depth-Aware Deep Visual-Inertial Odometry},
  author    = {Warda},
  year      = {2025}
}
```
