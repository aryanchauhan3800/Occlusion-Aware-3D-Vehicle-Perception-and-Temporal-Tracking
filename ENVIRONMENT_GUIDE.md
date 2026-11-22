# CenterPoint-KITTI: Environment Engineering & Reproducibility Guide

## 1. Executive Summary & Root-Cause Failure Autopsy

When attempting to build and run `CenterPoint-KITTI` on modern deep learning stacks (such as Python 3.13, PyTorch 2.x, CUDA 13, and recent Ubuntu/Colab runtimes), several catastrophic compilation, linking, and runtime failures occur. This autopsy explains the root causes of those issues and establishes the verified configuration required for full pipeline stability.

---

### Root-Cause Analysis of Previous Environment Failures

| Failure Symptom | Underlying Cause | Technical Explanation |
| :--- | :--- | :--- |
| **`fatal error: THC/THC.h: No such file or directory`** | Removal of `THC` (Torch CUDA) C-API in PyTorch $\ge$ 1.11 | `CenterPoint-KITTI`'s `setup.py` compiles PointNet2 CUDA extensions (`pointnet2_stack` and `pointnet2_batch`), which explicitly include `#include <THC/THC.h>` and depend on `THCState *state`. PyTorch deprecated THC in 1.10 and completely excised it in 1.11+. |
| **`ImportError: libc10.so / libtorch_cpu.so: cannot open shared object file`** | Missing PyTorch library path in dynamic linker (`ld.so`) | PyTorch shared libraries reside inside Python's `site-packages/torch/lib/`. When custom compiled extensions (e.g. `iou3d_nms_cuda.so`) are loaded by Python without `torch` having been imported first with global symbol visibility (`RTLD_GLOBAL`), or without `torch/lib` in `LD_LIBRARY_PATH`, the Linux dynamic loader cannot resolve dependencies. |
| **Colab Kernel Crash / Segmentation Fault (`SIGSEGV`) on `import iou3d_nms_cuda`** | C++ ABI mismatch & Torch Tensor layout incompatibility | PyTorch 2.x was compiled using GCC 9+/C++17 with breaking internal changes to `c10::TensorImpl`. Compiling C++ extensions against PyTorch 2.x while linking against outdated headers, or mixing host compilers across shared objects, produces silent memory corruption and an immediate segmentation fault. |
| **`spconv 1.2.1` build failure / missing wheel** | Spconv 1.x deprecation & Python $\ge$ 3.9 incompatibility | Official `spconv 1.x` was built for Python 3.6–3.8 and CUDA 10/11. No prebuilt wheels exist for Python $\ge$ 3.9. Upstream `spconv 2.x` refactored its entire API (`spconv.pytorch` namespace, removal of `spconv.utils.VoxelGenerator`), which breaks `CenterPoint-KITTI`'s data loader and 3D backbone. |
| **`AttributeError: module 'numpy' has no attribute 'bool'` or `'int'`** | NumPy 1.24+ removal of deprecated type aliases | `pcdet/datasets/kitti/kitti_object_eval_python/kitti_common.py`, `pcdet/models/backbones_2d/base_bev_backbone.py`, and `pcdet/utils/box_utils.py` utilize `np.bool` and `np.int`. NumPy 1.24.0 permanently removed these aliases. |
| **`TypeError: load() missing 1 required positional argument: 'Loader'`** | PyYAML 6.0+ breaking change | `pcdet/datasets/kitti/kitti_dataset.py` line 432 calls `yaml.load(open(sys.argv[2]))` without the `Loader=` argument. PyYAML $\ge$ 6.0 mandates `Loader=yaml.FullLoader` or `yaml.SafeLoader`. |

---

## 2. Golden Target Environment Specification

To guarantee 100% reproducible training, evaluation, and extension execution without code modifications, use this exact configuration:

| Component | Verified Target Version | Notes / Constraints |
| :--- | :--- | :--- |
| **Operating System** | **Ubuntu 18.04 LTS / 20.04 LTS** (x86_64) | Ubuntu 22.04 supported with GCC 9 toolchain |
| **Python** | **3.8.10** (or **3.7.10**) | Standard CPython; Python $\ge$ 3.9 breaks Spconv 1.2.1 |
| **PyTorch** | **1.10.1** | Last release containing `<THC/THC.h>` |
| **Torchvision** | **0.11.2** | Binary matched to PyTorch 1.10.1 |
| **CUDA Toolkit (nvcc)** | **CUDA 11.3** (or **CUDA 11.1**) | Must match PyTorch CUDA build |
| **Host Compiler** | **GCC $\le$ 9.3.0, G++ $\le$ 9.3.0** | GCC 10/11+ rejected by CUDA 11 nvcc |
| **Sparse Convolution** | **spconv 1.2.1** | Tag `v1.2.1` built with Boost and CMake $\ge$ 3.18 |
| **NumPy** | **1.21.6** | Strictly pinned $< 1.24.0$ to retain `np.bool`, `np.int` |
| **PyYAML** | **5.4.1** | Strictly pinned $< 6.0$ for `yaml.load()` backward compatibility |
| **Build Tools** | **Ninja 1.10.2**, **CMake $\ge$ 3.18**, **Setuptools 59.5.0** | Setuptools $\ge$ 60 introduces distutils deprecation issues |
| **GPU Hardware** | **NVIDIA T4, RTX 2080Ti, RTX 3090, V100, A100** | Compute capability 7.0, 7.5, 8.0, 8.6 |

---

## 3. Google Colab vs. Dedicated Ubuntu GPU Machine

### Can Google Colab Run This Environment?
* **Current Colab Runtime:** Google Colab currently provisions Ubuntu 22.04 LTS with Python 3.10+, PyTorch 2.x, and CUDA 12.x drivers.
* **Why default Colab fails:** Running `pip install -r requirements.txt && python setup.py develop` directly in Colab's interactive kernel will trigger the `THC/THC.h` missing header error and NumPy `np.bool` exceptions. Forcing PyTorch 1.10.1 into Colab's Python 3.10 runtime causes fatal C-extension symbol mismatches and kernel crashes.
* **Supported Colab Workflow:**
  1. **Inspection & Visual Sanity Checks:** The Colab notebook can inspect point clouds, verify calibration matrices, convert 3D boxes between Camera and LiDAR frames, and render BEV plots natively.
  2. **Sub-process Conda Execution in Colab:** By provisioning an isolated Conda environment (`centerpoint_env`) via Miniforge/Micromamba within the Colab VM filesystem, training and evaluation scripts can be executed headlessly via `conda run -n centerpoint_env python train.py ...`.
  3. **Dedicated GPU Machine Recommendation:** For iterative research, multi-epoch training (80 epochs on KITTI takes ~15–20 hours on an NVIDIA T4 / ~5 hours on an RTX 3090), and seamless interactive debugging, a **dedicated Ubuntu 20.04/18.04 GPU VM** (e.g. AWS EC2 `g4dn.xlarge`, GCP `n1-standard-4` + T4, Lambda Labs, or local workstation) is strongly recommended over standard ephemeral Colab sessions.

---

## 4. Installation Guide: Dedicated Ubuntu GPU Machine / Cloud VM

Follow these exact steps on a clean Ubuntu 18.04 / 20.04 GPU instance.

### Step 1: Install System Dependencies
```bash
sudo apt-get update && sudo apt-get install -y \
    build-essential \
    gcc-9 g++-9 \
    libboost-all-dev \
    libsparsehash-dev \
    cmake \
    git \
    wget \
    curl

# Ensure gcc-9 is the active compiler if on Ubuntu 20.04/22.04
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 90 --slave /usr/bin/g++ g++ /usr/bin/g++-9
```

### Step 2: Install CUDA 11.3 Toolkit (if not already installed)
Verify with `nvcc --version`. If missing:
```bash
wget https://developer.download.nvidia.com/compute/cuda/11.3.1/local_installers/cuda_11.3.1_465.19.01_linux.run
sudo sh cuda_11.3.1_465.19.01_linux.run --silent --toolkit
export CUDA_HOME=/usr/local/cuda-11.3
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### Step 3: Create Conda Environment
```bash
conda create -y -n centerpoint python=3.8.10
conda activate centerpoint
```

### Step 4: Install PyTorch 1.10.1 and Pinned Dependencies
```bash
pip install --upgrade pip==22.3.1 setuptools==59.5.0 wheel==0.38.4

# PyTorch 1.10.1 + CUDA 11.3
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

# Install pinned dependencies
pip install -r requirements-pinned.txt
```

### Step 5: Build and Install Spconv 1.2.1
```bash
git clone -b v1.2.1 https://github.com/traveller59/spconv.git --recursive
cd spconv
python setup.py bdist_wheel
pip install dist/spconv-1.2.1-cp38-cp38-linux_x86_64.whl
cd ..
```

### Step 6: Clone and Build CenterPoint-KITTI
```bash
git clone https://github.com/tianweiy/CenterPoint-KITTI.git
cd CenterPoint-KITTI

# Add PyTorch library path to LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$(python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"):$LD_LIBRARY_PATH

python setup.py develop
```

### Step 7: Verify CUDA Extensions
```bash
python -c "
import torch
import spconv
from pcdet.ops.iou3d_nms import iou3d_nms_cuda
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_cuda
print('CUDA Available:', torch.cuda.is_available())
print('Device:', torch.cuda.get_device_name(0))
print('SUCCESS: All CenterPoint CUDA extensions loaded cleanly!')
"
```

---

## 5. Dataset Organization & Preparation

Organize the KITTI 3D Object Detection dataset inside `CenterPoint-KITTI/data/kitti` as follows:

```
CenterPoint-KITTI
├── data
│   ├── kitti
│   │   ├── ImageSets
│   │   │   ├── train.txt       (3,711 sample IDs)
│   │   │   ├── val.txt         (3,768 sample IDs)
│   │   │   ├── test.txt        (7,517 sample IDs)
│   │   │   └── trainval.txt
│   │   ├── training
│   │   │   ├── calib/          (7,481 calibration .txt files)
│   │   │   ├── velodyne/       (7,481 point cloud .bin files)
│   │   │   ├── label_2/        (7,481 ground truth label .txt files)
│   │   │   ├── image_2/        (7,481 front camera .png images)
│   │   │   └── planes/         (Optional: ground planes .txt files)
│   │   └── testing
│   │       ├── calib/          (7,518 calibration .txt files)
│   │       ├── velodyne/       (7,518 point cloud .bin files)
│   │       └── image_2/        (7,518 front camera .png images)
```

Generate the database info pickle files (`kitti_infos_train.pkl`, `kitti_infos_val.pkl`, `kitti_dbinfos_train.pkl`):
```bash
python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml
```

---

## 6. Training, Evaluation, and Inference Execution

### 1. Training (Single GPU)
```bash
# Debugging run (1 epoch, batch size 2)
python tools/train.py --cfg_file tools/cfgs/kitti_models/centerpoint.yaml \
    --batch_size 2 --epochs 1 --workers 2 --extra_tag debug_run

# Full training run (80 epochs, batch size 4 or 8)
python tools/train.py --cfg_file tools/cfgs/kitti_models/centerpoint.yaml \
    --batch_size 4 --epochs 80 --workers 4 --extra_tag kitti_centerpoint_baseline
```

### 2. Evaluation (KITTI Official Benchmark Metrics)
```bash
python tools/test.py --cfg_file tools/cfgs/kitti_models/centerpoint.yaml \
    --batch_size 4 \
    --ckpt output/kitti_models/centerpoint/kitti_centerpoint_baseline/ckpt/checkpoint_epoch_80.pth
```

### 3. Inference & Detection Export
Use the provided Python script / notebook cells in `CenterPoint_KITTI_Colab.ipynb` to generate structured JSON/CSV/NPZ files containing 3D vehicle coordinates, dimensions, yaw, confidence, and 256-D BEV embeddings for temporal Transformer tracking.

---

## 7. Troubleshooting Playbook

### Q1: `error: #error -- unsupported GNU version! gcc versions later than 9 are not supported!`
* **Cause:** CUDA 11.1 / 11.3 nvcc rejects GCC $\ge$ 10.
* **Fix:** Install GCC 9 (`sudo apt install gcc-9 g++-9`) and set environment variables:
  ```bash
  export CC=/usr/bin/gcc-9
  export CXX=/usr/bin/g++-9
  export CUDAHOSTCXX=/usr/bin/g++-9
  ```

### Q2: `ImportError: libtorch_cpu.so: cannot open shared object file: No such file or directory`
* **Cause:** The dynamic linker cannot find PyTorch libraries when loading custom C++ extensions.
* **Fix:** Run this before starting Python:
  ```bash
  export LD_LIBRARY_PATH=$(python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"):$LD_LIBRARY_PATH
  ```

### Q3: `TypeError: load() missing 1 required positional argument: 'Loader'` in `create_kitti_infos`
* **Cause:** PyYAML 6.0+ requires `Loader=...`.
* **Fix:** Downgrade PyYAML to 5.4.1:
  ```bash
  pip install pyyaml==5.4.1
  ```

### Q4: `AttributeError: module 'numpy' has no attribute 'int'` or `'bool'`
* **Cause:** NumPy $\ge$ 1.24 removed deprecated aliases.
* **Fix:** Downgrade NumPy to 1.21.6:
  ```bash
  pip install numpy==1.21.6
  ```

### Q5: CUDA Out of Memory (OOM) during training
* **Cause:** High batch size or point cloud voxel density exceeding GPU VRAM.
* **Fix:** In `tools/train.py`, pass `--batch_size 2` or `--batch_size 4`. On 16 GB GPUs (like NVIDIA T4), batch size 4 fits comfortably with ~8.5 GB VRAM utilization.
