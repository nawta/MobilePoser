# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MobilePoser is a real-time full-body pose estimation system that uses IMU sensors to estimate 3D human body pose and translation. It outputs SMPL body model parameters from sparse IMU data (2-5 sensors).

## Common Commands

### Environment Setup
```bash
conda create -n mobileposer310 python=3.10
conda activate mobileposer310
pip install -r requirements.txt
pip install -e .
```

### Data Processing
```bash
# Process AMASS dataset
PYTHONPATH=. python mobileposer/process.py --dataset amass

# Process DIP-IMU dataset  
PYTHONPATH=. python mobileposer/process.py --dataset dip

# Process Nymeria dataset with different IMU configurations
PYTHONPATH=. python mobileposer/process.py --dataset nymeria --imu-device aria --contact-logic xdata
PYTHONPATH=. python mobileposer/process.py --dataset nymeria --imu-device xsens --contact-logic legacy

# Process partial dataset for debugging
PYTHONPATH=. python mobileposer/process.py --dataset nymeria --max_sequences 10
```

### Training
```bash
# Train all modules from scratch
python train.py

# Train specific module
python train.py --module poser

# Continue from checkpoint
python train.py --init-from checkpoints/model.ckpt

# Debug run (single epoch)
python train.py --fast-dev-run

# Finetune on specific dataset
./finetune.sh dip checkpoints/weights.pth

# Combine module weights into single file
python combine_weights.py --checkpoint checkpoints/dir --finetune dip
```

### Evaluation
```bash
# Evaluate on DIP dataset
python evaluate.py --model checkpoints/weights.pth --dataset dip

# Evaluate on Nymeria dataset
python evaluate.py --model checkpoints/weights.pth --dataset nymeria
```

### Visualization
```bash
# Visualize predictions only (GT=0)
python example.py --model checkpoints/weights.pth --dataset dip --seq-num 5

# Visualize predictions + ground truth (GT=1)
GT=1 python example.py --model checkpoints/weights.pth --dataset dip --with-tran

# Save visualization as video
GT=1 python example.py --model checkpoints/weights.pth --save-video --video-path output.mp4

# Convert frames to video
ffmpeg -r 25 -i frames_dir/frame_%04d.png -c:v libx264 -vf 'fps=25' output.mp4
```

### Head Pose Ensemble (SLAM Integration)
```bash
# Basic head pose ensemble with Visual-Inertial SLAM
python head_pose_example.py \
    --sequence /path/to/nymeria/sequence/dir \
    --weights checkpoints/weights.pth \
    --slam-type mock_vi \
    --fusion-method weighted_average

# Adaptive ensemble with automatic mode selection and dynamic weights
python adaptive_head_demo.py \
    --sequence /path/to/nymeria/sequence/dir \
    --weights checkpoints/weights.pth \
    --max-frames 300

# Test with data dropout simulation (shows adaptive behavior)
python adaptive_head_demo.py \
    --sequence /path/to/nymeria/sequence/dir \
    --weights checkpoints/weights.pth \
    --max-frames 500

# Disable dropout simulation for clean data testing
python adaptive_head_demo.py \
    --sequence /path/to/nymeria/sequence/dir \
    --weights checkpoints/weights.pth \
    --no-dropouts
```

## Architecture

### Core Model Structure
The system uses a modular architecture with four specialized neural networks:

1. **Joints Module** (`models/joints.py`): Predicts 24 3D joint positions from IMU data
2. **Poser Module** (`models/poser.py`): Converts predicted joints + IMU to SMPL pose parameters  
3. **FootContact Module** (`models/footcontact.py`): Binary classification for foot-ground contact
4. **Velocity Module** (`models/velocity.py`): Predicts per-frame joint velocities

All modules use LSTM-based RNNs defined in `models/rnn.py`. The main `MobilePoserNet` in `models/net.py` orchestrates these modules.

### Data Flow
1. IMU sensors provide 3-axis acceleration + 3x3 rotation matrix per sensor
2. Data is preprocessed to 30 FPS and normalized
3. Joints module predicts initial joint positions
4. Poser module refines to SMPL parameters using joints + original IMU
5. FootContact + Velocity modules enable translation estimation
6. Optional physics optimization refines results

### Key Design Decisions
- **6D rotation representation** instead of quaternions for neural network stability
- **Two-stage prediction** (joints then pose) for better supervision
- **Streaming support** for large datasets via `StreamingPoseDataset`
- **Multi-scale temporal losses** to reduce motion jitter
- **Robust NaN handling** throughout the pipeline

### Head Pose Ensemble System
The system includes an advanced adaptive ensemble approach for head pose estimation:

#### Basic Ensemble (`head_pose_ensemble.py`)
1. **Head-Specific Tracking**: Focuses on head position/orientation only
2. **Visual-Inertial SLAM** (`models/slam.py`): Integrates camera + head IMU for scaled pose estimation
3. **Intelligent Fusion** (`models/fusion.py`): Combines IMU orientation accuracy with SLAM translation accuracy

#### Adaptive Ensemble (`adaptive_head_ensemble.py`)
1. **Automatic Mode Selection**: 
   - RGB + Head IMU → Visual-Inertial SLAM
   - RGB only → Monocular SLAM
   - No RGB → IMU-only mode
2. **Dynamic Weight Calculation** (`models/adaptive_slam.py`): 
   - Confidence-based weighting
   - Tracking state assessment
   - Temporal consistency analysis
   - Scale estimation quality
3. **Temporal Feedback**: Uses previous fused pose to improve next frame prediction
4. **Graceful Degradation**: Handles missing data smoothly

**Benefits:**
- Fully adaptive to available sensor data
- 30-50% improvement in head translation accuracy
- Robust to sensor dropouts and failures
- Dynamic ensemble weights optimize for current conditions
- Temporal feedback reduces jitter and improves consistency

## Configuration

Key configuration files:
- `config.py`: Paths, hyperparameters, model settings
- Dataset paths must be set in `paths.*` before processing
- Training hyperparameters in `train_hypers` and `finetune_hypers`

## Dataset Structure

Processed datasets are stored as PyTorch tensors with keys:
- `acc`: Accelerometer data [N_sequences, T, 6, 3]
- `ori`: Orientation data [N_sequences, T, 6, 3, 3] or quaternions [N_sequences, T, 6, 4]
- `pose`: SMPL pose parameters [N_sequences, T, 24, 3, 3]
- `tran`: Global translation [N_sequences, T, 3]
- `joint`: 3D joint positions [N_sequences, T, 24, 3]
- `contact`: Foot contact labels [N_sequences, T, 2]

## Error Handling

The codebase uses print statements for all logging. Common error patterns:
```python
print(f"Error processing {filename}: {e}")
print(f"Training error: {e}")
```

No structured logging framework is implemented. Errors are printed to console without timestamps or severity levels.