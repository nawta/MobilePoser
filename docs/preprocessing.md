# MobilePoser Dataset Preprocessing Documentation

This document explains the preprocessing steps for each dataset used in MobilePoser.

## AMASS Dataset

The `process_amass()` function performs the following preprocessing steps:

1. **Loading Data**: Loads SMPL pose data from AMASS dataset (.npz files)
2. **Downsampling**: Reduces framerate from original to target FPS (30 fps)
3. **Coordinate Transformation**: Converts axis-angle representation to rotation matrices
4. **IMU Synthesis**: Creates synthetic IMU data (accelerations and orientations) for 6 body locations
5. **Foot Contact Detection**: Computes foot-ground contact probabilities based on foot movement
6. **Data Saving**: Stores processed data in .pt format for training and evaluation

## DIP_IMU Dataset

The `process_dipimu()` function performs the following preprocessing steps:

1. **Loading Data**: Loads IMU data and SMPL parameters from DIP_IMU dataset
2. **Pose Conversion**: Converts SMPL pose parameters to rotation matrices
3. **Forward Kinematics**: Computes joint positions and global rotations
4. **Data Organization**: Separates data into training and test sets
5. **Data Saving**: Stores processed data in .pt format for training and evaluation

## IMUPoser Dataset

The `process_imuposer()` function performs the following preprocessing steps:

1. **Loading Data**: Loads IMU data and SMPL parameters from IMUPoser dataset
2. **Pose Conversion**: Converts SMPL pose parameters to rotation matrices
3. **Forward Kinematics**: Computes joint positions and global rotations
4. **Data Saving**: Stores processed data in .pt format for training and evaluation

## TotalCapture Dataset

The `process_totalcapture()` function performs the following preprocessing steps:

1. **Loading Data**: Loads IMU data and ground truth poses from TotalCapture dataset
2. **Calibration**: Applies calibration to IMU data
3. **Foot Contact Detection**: Computes foot-ground contact probabilities
4. **Acceleration Bias Removal**: Removes bias from acceleration data
5. **Data Saving**: Stores processed data in .pt format for evaluation

## Nymeria Dataset

The `process_nymeria()` function performs the following preprocessing steps:

1. **Loading Data**: Loads XSens motion data from .npz files in the Nymeria dataset
2. **Quaternion Conversion**: Converts quaternions to axis-angle representation
3. **Downsampling**: Reduces framerate from original (likely 240Hz) to target FPS (30 fps)
4. **Coordinate Alignment**: Aligns coordinate systems between Nymeria and SMPL
5. **IMU Synthesis**: Creates synthetic IMU data (accelerations and orientations) for 6 body locations
6. **Foot Contact Detection**: Computes foot-ground contact probabilities based on foot movement
7. **Data Saving**: Stores processed data in .pt format for training and evaluation

## Common Processing Elements

All dataset processing functions share these common elements:

1. **SMPL Model**: Uses the SMPL parametric model for forward kinematics
2. **IMU Locations**: Focuses on 6 key body locations (left wrist, right wrist, left thigh, right thigh, head, pelvis)
3. **Data Format**: Standardizes all datasets to the same format for training and evaluation
4. **Rotation Representation**: Converts various rotation representations to rotation matrices
5. **Output Format**: Saves processed data as PyTorch tensors (.pt files)
