#!/usr/bin/env python3
"""Test script to verify SLAM data loading with streaming."""

import torch
import numpy as np
from pathlib import Path
import time

from mobileposer.slam_data_streaming import StreamingSlamPoseDataset, SlamPoseDataModule
from mobileposer.slam_selector import slam_selector
from mobileposer.config import finetune_hypers

def test_slam_dataset():
    """Test the streaming SLAM dataset."""
    print("=" * 60)
    print("Testing SLAM Data Loading")
    print("=" * 60)
    
    # Check SLAM availability
    slam_selector.print_status()
    print()
    
    # Create dataset
    print("Creating streaming SLAM dataset...")
    # Use mock SLAM since ORB-SLAM3 is not available
    dataset = StreamingSlamPoseDataset(
        fold='train',
        finetune='nymeria_aria_xdata_train',  # Correct name to match data files
        slam_enabled=True,
        slam_type='mock',  # Use mock for testing
        stream_buffer_size=2,
        cache_slam_results=True
    )
    
    # Test iteration
    print("\nTesting data iteration...")
    start_time = time.time()
    
    try:
        # Get first few batches
        data_iter = iter(dataset)
        for i in range(3):
            print(f"\nBatch {i+1}:")
            data = next(data_iter)
            
            if len(data) == 7:  # Training mode
                imu, pose, joint, tran, vel, contact, slam_pose = data
                print(f"  IMU shape: {imu.shape}")
                print(f"  Pose shape: {pose.shape if pose is not None else 'None'}")
                print(f"  Joint shape: {joint.shape if joint is not None else 'None'}")
                print(f"  Translation shape: {tran.shape if tran is not None else 'None'}")
                print(f"  Velocity shape: {vel.shape if vel is not None else 'None'}")
                print(f"  Contact shape: {contact.shape if contact is not None else 'None'}")
                print(f"  SLAM head pose shape: {slam_pose.shape if slam_pose is not None else 'None'}")
                
                # Check for NaN values
                has_nan = False
                for name, tensor in [("IMU", imu), ("Pose", pose), ("Joint", joint), 
                                   ("Translation", tran), ("Velocity", vel), 
                                   ("Contact", contact), ("SLAM", slam_pose)]:
                    if tensor is not None and torch.isnan(tensor).any():
                        print(f"  WARNING: NaN values in {name}")
                        has_nan = True
                
                if not has_nan:
                    print("  ✓ No NaN values detected")
            
            elapsed = time.time() - start_time
            print(f"  Time elapsed: {elapsed:.2f}s")
        
        print("\n✓ Dataset iteration successful!")
        
    except Exception as e:
        print(f"\n✗ Error during iteration: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        dataset.cleanup()
    
    print("\n" + "=" * 60)

def test_slam_datamodule():
    """Test the SLAM data module with Lightning."""
    print("\nTesting SLAM DataModule...")
    print("=" * 60)
    
    # Create data module
    data_module = SlamPoseDataModule(
        finetune='nymeria_aria_xdata_train',  # Correct name to match data files
        streaming=True,
        slam_enabled=True,
        slam_type='mock',  # Use mock for testing
        cache_slam_results=True
    )
    
    # Setup
    data_module.setup('fit')
    
    # Test train dataloader
    print("\nTesting train dataloader...")
    train_loader = data_module.train_dataloader()
    print(f"Batch size: {train_loader.batch_size}")
    print(f"Num workers: {train_loader.num_workers}")
    
    try:
        # Get first batch
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= 2:  # Test first 2 batches
                break
            
            (inputs, input_lengths), (outputs, output_lengths) = batch
            
            print(f"\nBatch {batch_idx + 1}:")
            print(f"  Input shape: {inputs.shape}")
            print(f"  Input lengths: {input_lengths[:5]}...")  # First 5
            
            for key, tensor in outputs.items():
                if tensor is not None:
                    print(f"  {key} shape: {tensor.shape}")
            
            # Check SLAM poses
            if 'slam_head_poses' in outputs and outputs['slam_head_poses'] is not None:
                print(f"  ✓ SLAM head poses included!")
            else:
                print(f"  ✗ No SLAM head poses")
        
        print("\n✓ DataModule test successful!")
        
    except Exception as e:
        print(f"\n✗ DataModule error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        data_module.teardown('fit')
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    # Test dataset first
    test_slam_dataset()
    
    # Then test data module
    test_slam_datamodule()
    
    print("\nAll tests completed!")