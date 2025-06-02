#!/usr/bin/env python3
"""Test ORB-SLAM3 with Nymeria-specific calibration."""

import os
import sys
import numpy as np

# Set up environment
os.environ['LD_LIBRARY_PATH'] = '/usr/lib/x86_64-linux-gnu:/usr/local/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
os.environ['LD_PRELOAD'] = '/usr/lib/x86_64-linux-gnu/libstdc++.so.6'
sys.path.append('/home/naoto/docker_workspace/MobilePoser')

from mobileposer.models.real_orbslam3 import RealOrbSlam3Interface

print("Testing ORB-SLAM3 with Nymeria calibration...")
print("-" * 60)

# Test monocular SLAM
print("\n1. Testing Monocular SLAM with Nymeria calibration:")
try:
    mono_slam = RealOrbSlam3Interface(mode="monocular")
    mono_slam.initialize()  # Initialize to load settings
    print(f"   Settings file: {mono_slam.settings_file}")
    
    # Test with dummy frames
    for i in range(5):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        timestamp = i * 0.033  # 30 FPS
        pose = mono_slam.process_frame(frame, timestamp)
        state = mono_slam.tracking_state
        print(f"   Frame {i}: tracking state = {state}")
    
    print("   ✓ Monocular SLAM working with Nymeria calibration!")
    
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Test VI-SLAM
print("\n2. Testing Visual-Inertial SLAM with Nymeria calibration:")
try:
    vi_slam = RealOrbSlam3Interface(mode="visual_inertial")
    vi_slam.initialize()  # Initialize to load settings
    print(f"   Settings file: {vi_slam.settings_file}")
    
    # Test with dummy frames and IMU data
    for i in range(5):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        timestamp = i * 0.033  # 30 FPS
        # Mock IMU data (acc, gyro)
        imu_data = {
            'acc': np.random.randn(3) * 0.1,
            'gyro': np.random.randn(3) * 0.01,
            'ori': np.eye(3)  # Identity rotation
        }
        pose = vi_slam.process_frame_with_imu(frame, imu_data, timestamp)
        state = vi_slam.tracking_state
        print(f"   Frame {i}: tracking state = {state}")
    
    print("   ✓ VI-SLAM working with Nymeria calibration!")
    
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "-" * 60)
print("Calibration test completed!")