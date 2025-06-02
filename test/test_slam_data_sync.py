#!/usr/bin/env python3
"""Test RGB-IMU synchronization in SLAM dataset."""

import os
import sys
sys.path.append('/home/naoto/docker_workspace/MobilePoser')

import torch
from pathlib import Path
from mobileposer.slam_data_streaming import StreamingSlamPoseDataset

# Test parameters
finetune = "nymeria_aria_xdata_train"
data_folder = Path("/home/naoto/docker_workspace/MobilePoser/datasets/processed_datasets")
rgb_root = Path("/mnt/nas2/naoto/nymeria_dataset/data_video_main_rgb")

print("Testing RGB-IMU Synchronization")
print("=" * 60)

# Load a sample data file to check sequence names
sample_file = data_folder / "nymeria_aria_xdata_train_000.pt"
if sample_file.exists():
    print(f"\nLoading sample file: {sample_file}")
    data = torch.load(sample_file)
    
    # Check what's in the data
    print("\nData structure:")
    for key in data.keys():
        if isinstance(data[key], list):
            print(f"  {key}: list of {len(data[key])} items")
        else:
            print(f"  {key}: {type(data[key])}")
    
    # Check sequence names
    if 'sequence_name' in data:
        seq_names = data['sequence_name']
        print(f"\nFound {len(seq_names)} sequences:")
        for i, name in enumerate(seq_names[:5]):  # First 5
            print(f"  [{i}] {name}")
            
            # Check if corresponding RGB video exists
            video_path = rgb_root / name / "video_main_rgb.mp4"
            exists = video_path.exists()
            print(f"      RGB video exists: {exists}")
            if exists:
                print(f"      Path: {video_path}")
    else:
        print("\nNo 'sequence_name' key found in data!")
        
    # Check data shapes and timing
    if 'acc' in data and len(data['acc']) > 0:
        first_acc = data['acc'][0]
        print(f"\nFirst sequence shape: {first_acc.shape}")
        print(f"Duration at 30 FPS: {first_acc.shape[0] / 30:.2f} seconds")
        print(f"Number of frames: {first_acc.shape[0]}")
        
else:
    print(f"Sample file not found: {sample_file}")

# Test dataset creation
print("\n" + "=" * 60)
print("Testing StreamingSlamPoseDataset")
print("=" * 60)

try:
    dataset = StreamingSlamPoseDataset(
        fold='train',
        finetune=finetune,
        stream_buffer_size=1,  # Small buffer for testing
        slam_type='mock',  # Use mock to avoid SLAM initialization
        slam_enabled=True
    )
    
    print(f"\nDataset created successfully")
    print(f"Data folder: {dataset.data_folder}")
    print(f"RGB root: {dataset.rgb_root}")
    print(f"Number of data files: {len(dataset.data_files)}")
    print(f"Data files: {dataset.data_files[:3]}...")
    
    # Test iteration
    print("\nTesting data iteration (first batch)...")
    for i, batch in enumerate(dataset):
        if i == 0:
            print(f"\nBatch structure:")
            print(f"  Number of elements: {len(batch)}")
            for j, elem in enumerate(batch):
                if elem is not None:
                    if isinstance(elem, torch.Tensor):
                        print(f"  [{j}] Tensor shape: {elem.shape}")
                    else:
                        print(f"  [{j}] Type: {type(elem)}")
                else:
                    print(f"  [{j}] None")
            
            # Check if SLAM data is present
            if len(batch) > 6:  # Should have SLAM data as last element
                slam_data = batch[-1]
                if slam_data is not None:
                    print(f"\nSLAM data present! Shape: {slam_data.shape}")
                else:
                    print("\nSLAM data is None (RGB not found or SLAM disabled)")
        
        if i >= 2:  # Check a few batches
            break
    
except Exception as e:
    print(f"\nError creating dataset: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Summary:")
print("- Check if sequence names in .pt files match RGB video directories")
print("- Verify RGB videos exist at expected paths")
print("- Ensure timestamp calculation (frame_idx / 30.0) is correct")
print("- SLAM data should be non-None if RGB videos are found")