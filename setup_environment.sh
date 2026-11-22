#!/usr/bin/env bash
# ==============================================================================
# Setup Environment for CenterPoint-KITTI
# Target: Ubuntu 18.04 / 20.04 (x86_64), Python 3.8, PyTorch 1.10.1, CUDA 11.3
# ==============================================================================

set -e

# ANSI Color codes for readability
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================================${NC}"
echo -e "${BLUE}    CenterPoint-KITTI Environment Automated Installer   ${NC}"
echo -e "${BLUE}======================================================${NC}"

# Step 0: Check System Architecture and GPU
echo -e "\n${YELLOW}[Step 0/6] Inspecting Host System...${NC}"

if [[ "$(uname -s)" != "Linux" ]]; then
    echo -e "${RED}[ERROR] This installation script is designed for Linux (Ubuntu 18.04 / 20.04 / 22.04).${NC}"
    echo -e "${RED}Current OS: $(uname -s). Please run in an Ubuntu GPU VM or container.${NC}"
    exit 1
fi

if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}[ERROR] 'nvidia-smi' not found! An NVIDIA GPU is required.${NC}"
    exit 1
fi

echo -e "${GREEN}[OK] NVIDIA GPU detected:${NC}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

# Check nvcc / CUDA toolkit
if command -v nvcc &> /dev/null; then
    NVCC_VER=$(nvcc --version | grep "release" | awk '{print $5}' | tr -d ',')
    echo -e "${GREEN}[OK] Detected nvcc version: ${NVCC_VER}${NC}"
    export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
elif [ -d "/usr/local/cuda-11.3" ]; then
    export CUDA_HOME=/usr/local/cuda-11.3
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
    echo -e "${GREEN}[OK] Using CUDA toolkit at /usr/local/cuda-11.3${NC}"
elif [ -d "/usr/local/cuda-11.1" ]; then
    export CUDA_HOME=/usr/local/cuda-11.1
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
    echo -e "${GREEN}[OK] Using CUDA toolkit at /usr/local/cuda-11.1${NC}"
elif [ -d "/usr/local/cuda" ]; then
    export CUDA_HOME=/usr/local/cuda
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
    echo -e "${GREEN}[OK] Using CUDA toolkit at /usr/local/cuda${NC}"
else
    echo -e "${YELLOW}[WARNING] CUDA toolkit (nvcc) not found in PATH or standard locations.${NC}"
    echo -e "${YELLOW}CUDA headers are required to compile spconv 1.2.1 and pcdet CUDA extensions.${NC}"
fi

# Step 1: Install System Dependencies (Boost, CMake, Build Tools)
echo -e "\n${YELLOW}[Step 1/6] Checking system build dependencies...${NC}"
if command -v apt-get &> /dev/null && [ "$EUID" -eq 0 ]; then
    echo "Installing libboost-all-dev, libsparsehash-dev, and build tools via apt..."
    apt-get update -qq && apt-get install -y -qq \
        build-essential \
        libboost-all-dev \
        libsparsehash-dev \
        cmake \
        git \
        wget \
        curl
elif command -v apt-get &> /dev/null && command -v sudo &> /dev/null; then
    echo "Installing build tools via sudo apt..."
    sudo apt-get update -qq && sudo apt-get install -y -qq \
        build-essential \
        libboost-all-dev \
        libsparsehash-dev \
        cmake \
        git \
        wget \
        curl
else
    echo "Skipping apt installation (either non-root or non-apt system). Ensure Boost and CMake are present."
fi

# Step 2: Conda Environment Setup
echo -e "\n${YELLOW}[Step 2/6] Configuring Python Conda Environment...${NC}"

ENV_NAME="centerpoint"

if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    if conda info --envs | grep -q "^${ENV_NAME} "; then
        echo -e "${GREEN}[OK] Conda environment '${ENV_NAME}' already exists. Activating...${NC}"
        conda activate ${ENV_NAME}
    else
        echo "Creating Conda environment '${ENV_NAME}' with Python 3.8.10..."
        conda create -y -n ${ENV_NAME} python=3.8.10
        conda activate ${ENV_NAME}
    fi
else
    echo -e "${YELLOW}[WARNING] Conda not found. Proceeding with active Python environment: $(which python)${NC}"
    PYTHON_MAJOR_MINOR=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    if [[ "$PYTHON_MAJOR_MINOR" != "3.7" && "$PYTHON_MAJOR_MINOR" != "3.8" ]]; then
        echo -e "${RED}[ERROR] Python version $PYTHON_MAJOR_MINOR is unsupported! Please use Python 3.7 or 3.8.${NC}"
        exit 1
    fi
fi

# Step 3: Install PyTorch and Pinned Dependencies
echo -e "\n${YELLOW}[Step 3/6] Installing PyTorch 1.10.1 and pinned dependencies...${NC}"
pip install --upgrade pip==22.3.1 setuptools==59.5.0 wheel==0.38.4

# Install PyTorch 1.10.1 with CUDA 11.3 support
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

# Install pinned dependencies
if [ -f "requirements-pinned.txt" ]; then
    pip install -r requirements-pinned.txt
else
    pip install \
        numpy==1.21.6 \
        numba==0.54.1 \
        llvmlite==0.37.0 \
        pyyaml==5.4.1 \
        easydict==1.9 \
        tensorboardX==2.5.1 \
        scikit-image==0.19.3 \
        tqdm==4.64.1 \
        open3d==0.15.2 \
        matplotlib==3.5.3 \
        pandas==1.3.5 \
        scipy==1.7.3 \
        pillow==9.4.0 \
        shapely==1.8.5.post1 \
        ninja==1.10.2.4
fi

# Step 4: Install Spconv 1.2.1
echo -e "\n${YELLOW}[Step 4/6] Installing spconv 1.2.1...${NC}"

python -c "import spconv" &> /dev/null && SPCONV_INSTALLED=1 || SPCONV_INSTALLED=0

if [ $SPCONV_INSTALLED -eq 1 ]; then
    SPCONV_VER=$(python -c "import spconv; print(getattr(spconv, '__version__', 'unknown'))")
    echo -e "${GREEN}[OK] spconv is already installed (version: ${SPCONV_VER}).${NC}"
else
    echo "spconv not found. Building spconv 1.2.1 from source..."
    TMP_DIR=$(mktemp -d)
    pushd "$TMP_DIR"
    git clone -b v1.2.1 https://github.com/traveller59/spconv.git --recursive
    cd spconv
    python setup.py bdist_wheel
    pip install dist/spconv-*.whl
    popd
    rm -rf "$TMP_DIR"
    echo -e "${GREEN}[OK] spconv 1.2.1 installed successfully.${NC}"
fi

# Step 5: Setup CenterPoint-KITTI and Compile CUDA Extensions
echo -e "\n${YELLOW}[Step 5/6] Compiling CenterPoint CUDA Extensions...${NC}"

REPO_DIR="CenterPoint-KITTI"

if [ ! -d "$REPO_DIR" ] && [ ! -f "setup.py" ]; then
    echo "Cloning CenterPoint-KITTI repository..."
    git clone https://github.com/tianweiy/CenterPoint-KITTI.git "$REPO_DIR"
    cd "$REPO_DIR"
elif [ -f "setup.py" ]; then
    echo "Already inside repository directory."
else
    cd "$REPO_DIR"
fi

# Ensure PyTorch library paths are available to linker
TORCH_LIB_PATH=$(python -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
export LD_LIBRARY_PATH=${TORCH_LIB_PATH}:${LD_LIBRARY_PATH:-}

echo "Running 'python setup.py develop'..."
python setup.py develop

# Step 6: Verification and Import Diagnostics
echo -e "\n${YELLOW}[Step 6/6] Verifying Environment and Compiled Extensions...${NC}"

python << 'EOF'
import sys

def verify():
    errors = []
    
    # 1. PyTorch & CUDA
    try:
        import torch
        print(f"[OK] PyTorch version: {torch.__version__}")
        print(f"[OK] CUDA Available in PyTorch: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[OK] Device: {torch.cuda.get_device_name(0)}")
            print(f"[OK] PyTorch CUDA Version: {torch.version.cuda}")
        else:
            errors.append("CUDA is not available in PyTorch.")
    except Exception as e:
        errors.append(f"PyTorch import error: {e}")

    # 2. NumPy
    try:
        import numpy as np
        print(f"[OK] NumPy version: {np.__version__}")
        # Verify legacy type aliases needed by OpenPCDet
        _ = np.bool(True)
        _ = np.int(1)
        print("[OK] NumPy legacy types (np.bool, np.int) verified.")
    except Exception as e:
        errors.append(f"NumPy compatibility check error: {e}")

    # 3. spconv
    try:
        import spconv
        print(f"[OK] spconv imported successfully.")
    except Exception as e:
        errors.append(f"spconv import error: {e}")

    # 4. pcdet IoU NMS CUDA extension
    try:
        from pcdet.ops.iou3d_nms import iou3d_nms_cuda
        print("[OK] pcdet.ops.iou3d_nms.iou3d_nms_cuda imported successfully.")
    except Exception as e:
        errors.append(f"iou3d_nms_cuda import error: {e}")

    # 5. pcdet RoIAware Pool3D CUDA extension
    try:
        from pcdet.ops.roiaware_pool3d import roiaware_pool3d_cuda
        print("[OK] pcdet.ops.roiaware_pool3d.roiaware_pool3d_cuda imported successfully.")
    except Exception as e:
        errors.append(f"roiaware_pool3d_cuda import error: {e}")

    # Summary
    if errors:
        print("\n" + "="*50)
        print("[VERIFICATION FAILED] The following errors were found:")
        for err in errors:
            print(f"  - {err}")
        print("="*50)
        sys.exit(1)
    else:
        print("\n" + "="*50)
        print("[ALL CHECKS PASSED] Environment is fully operational!")
        print("="*50)

if __name__ == '__main__':
    verify()
EOF

echo -e "\n${GREEN}======================================================${NC}"
echo -e "${GREEN}    CenterPoint-KITTI setup completed successfully!   ${NC}"
echo -e "${GREEN}======================================================${NC}"
