# ORB-SLAM3 Setup Guide for MobilePoser

This guide explains how to install and configure ORB-SLAM3 for use with MobilePoser's Visual-Inertial SLAM integration.

## Prerequisites

- Ubuntu 20.04 or 22.04
- CMake 3.10+
- GCC/G++ 7+
- Python 3.8+
- CUDA (optional, for GPU acceleration)

## Dependencies Installation

### 1. Install System Dependencies

```bash
sudo apt update
sudo apt install -y build-essential cmake git pkg-config
sudo apt install -y libeigen3-dev libopencv-dev libglew-dev
sudo apt install -y libboost-all-dev libssl-dev
sudo apt install -y libpython3-dev python3-pip python3-numpy
```

### 2. Install Pangolin (for visualization)

```bash
cd /tmp
git clone https://github.com/stevenlovegrove/Pangolin.git
cd Pangolin
mkdir build && cd build
cmake ..
make -j$(nproc)
sudo make install
```

### 3. Build ORB-SLAM3

```bash
# Clone ORB-SLAM3 into the third_party directory
cd /home/naoto/docker_workspace/MobilePoser/third_party
git clone https://github.com/UZ-SLAMLab/ORB_SLAM3.git

# Build ORB-SLAM3
cd ORB_SLAM3
chmod +x build.sh
./build.sh
```

### 4. Install pyOrbSlam3 Python Wrapper

```bash
# Clone pyOrbSlam3
cd /home/naoto/docker_workspace/MobilePoser/third_party
git clone https://github.com/jskinn/pyOrbSlam3.git

# Build the Python wrapper
cd pyOrbSlam3
mkdir build && cd build
cmake .. -DPYTHON_EXECUTABLE=$(which python3)
make -j$(nproc)

# Install the Python module
cd ..
pip install -e .
```

## Configuration

### Camera Calibration

MobilePoser includes default camera configurations for the Nymeria dataset. If using a different dataset, update the camera parameters in:

- `/home/naoto/docker_workspace/MobilePoser/mobileposer/slam_configs/`

### IMU Calibration

For Visual-Inertial SLAM, ensure your IMU calibration parameters are correctly set:

```yaml
# IMU noise parameters (example)
IMU.NoiseGyro: 1.7e-4       # rad/s
IMU.NoiseAcc: 2.0e-3        # m/s^2
IMU.GyroWalk: 1.9393e-05    # rad/s^2
IMU.AccWalk: 3.0000e-03     # m/s^3
IMU.Frequency: 60           # Hz
```

## Usage

### Training with SLAM

```bash
# Train with real ORB-SLAM3 Visual-Inertial mode
python mobileposer/train_slam.py \
    --slam-type vi \
    --finetune nymeria \
    --batch-size 32

# Train with Adaptive SLAM (automatically selects best mode)
python mobileposer/train_slam.py \
    --slam-type adaptive \
    --finetune nymeria
```

### Inference with SLAM

```bash
# Run adaptive head pose ensemble
python adaptive_head_demo.py \
    --sequence /path/to/nymeria/sequence \
    --weights checkpoints/weights.pth \
    --max-frames 300

# Run with specific SLAM type
python head_pose_example.py \
    --sequence /path/to/nymeria/sequence \
    --weights checkpoints/weights.pth \
    --slam-type orb_slam3_vi
```

## Troubleshooting

### ImportError: No module named 'pyOrbSlam'

1. Ensure pyOrbSlam3 is built correctly
2. Add to Python path: `export PYTHONPATH=/home/naoto/docker_workspace/MobilePoser/third_party/pyOrbSlam3:$PYTHONPATH`

### SLAM initialization fails

1. Check ORB vocabulary file exists: `/third_party/ORB_SLAM3/Vocabulary/ORBvoc.txt`
2. Verify camera calibration parameters match your dataset
3. Ensure RGB frames are being loaded correctly

### Low tracking quality

1. Ensure good lighting conditions in RGB video
2. Check camera motion is not too fast
3. Verify IMU-camera synchronization
4. Adjust ORB feature extraction parameters

## Docker Support

For easier setup, use the provided Docker configuration:

```bash
cd /home/naoto/docker_workspace/MobilePoser/docker
docker-compose up -d
docker exec -it mobileposer-slam bash
```

The Docker image includes all dependencies pre-installed.

## Verification

To verify SLAM is working correctly:

```bash
# Run test script
python test_real_orbslam3_integration.py

# Check SLAM integration status
python -c "from mobileposer.models.real_orbslam3 import create_real_orbslam3_interface; slam = create_real_orbslam3_interface(); print('SLAM available:', slam.initialize())"
```

## Performance Tips

1. **GPU Acceleration**: Build ORB-SLAM3 with CUDA support for faster feature extraction
2. **Multi-threading**: Adjust thread count in SLAM settings based on CPU cores
3. **Feature Count**: Balance accuracy vs speed with `ORBextractor.nFeatures`
4. **Frame Rate**: Process every N frames if real-time performance is not required

## References

- [ORB-SLAM3 Paper](https://arxiv.org/abs/2007.11898)
- [ORB-SLAM3 GitHub](https://github.com/UZ-SLAMLab/ORB_SLAM3)
- [pyOrbSlam3 GitHub](https://github.com/jskinn/pyOrbSlam3)