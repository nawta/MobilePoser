#!/usr/bin/env python3
"""Test script to verify real ORB-SLAM3 integration."""

import torch
import numpy as np
from pathlib import Path
import time
import os

# Set up environment
os.environ['LD_LIBRARY_PATH'] = '/usr/lib/x86_64-linux-gnu:/usr/local/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
os.environ['LD_PRELOAD'] = '/usr/lib/x86_64-linux-gnu/libstdc++.so.6'
import sys
sys.path.append('/home/naoto/docker_workspace/MobilePoser/third_party/pyOrbSlam3/pyOrbSlam3/build')

from mobileposer.slam_selector import slam_selector
from mobileposer.models.real_orbslam3 import RealOrbSlam3Interface

def test_orbslam3_direct():
    """Test ORB-SLAM3 directly."""
    print("=" * 60)
    print("Testing ORB-SLAM3 Direct")
    print("=" * 60)
    
    # Check availability
    slam_selector.print_status()
    print()
    
    # Try to create ORB-SLAM3 interface
    try:
        print("Creating RealOrbSlam3Interface...")
        slam = RealOrbSlam3Interface(mode="monocular")
        print("✓ ORB-SLAM3 interface created successfully!")
        
        # Test with dummy data
        print("\nTesting with dummy frame...")
        dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        timestamp = 0.0
        
        result = slam.process_frame(dummy_frame, timestamp)
        print(f"Result: {result}")
        
        # Shutdown
        slam.shutdown()
        print("✓ ORB-SLAM3 shutdown successfully!")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)

def test_slam_with_nymeria():
    """Test SLAM with actual Nymeria data."""
    print("\nTesting SLAM with Nymeria Data")
    print("=" * 60)
    
    try:
        from mobileposer.slam_data_streaming import StreamingSlamPoseDataset
        
        print("Creating streaming dataset with adaptive SLAM...")
        dataset = StreamingSlamPoseDataset(
            fold='train',
            finetune='nymeria_aria_xdata_train',
            slam_enabled=True,
            slam_type='adaptive',  # Use adaptive SLAM
            stream_buffer_size=1,
            cache_slam_results=True
        )
        
        print("\nProcessing first batch...")
        data_iter = iter(dataset)
        
        # Get first batch
        start_time = time.time()
        data = next(data_iter)
        elapsed = time.time() - start_time
        
        if len(data) == 7:  # Training mode
            imu, pose, joint, tran, vel, contact, slam_pose = data
            print(f"\n✓ Data loaded successfully in {elapsed:.2f}s")
            print(f"  IMU shape: {imu.shape}")
            if slam_pose is not None:
                print(f"  SLAM head pose shape: {slam_pose.shape}")
                print(f"  SLAM pose sample: {slam_pose[0][:6]}")  # First 6 values
            else:
                print("  No SLAM pose (might need actual video data)")
        
        # Cleanup
        dataset.cleanup()
        print("\n✓ Test completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    # Test direct ORB-SLAM3
    test_orbslam3_direct()
    
    # Test with Nymeria data
    test_slam_with_nymeria()
    
    print("\nAll tests completed!")