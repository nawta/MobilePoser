#!/usr/bin/env python3
"""Test ORB-SLAM3 initialization directly."""

import os
import sys
import numpy as np
import cv2

# Set up environment
os.environ['LD_LIBRARY_PATH'] = '/usr/lib/x86_64-linux-gnu:/usr/local/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
os.environ['LD_PRELOAD'] = '/usr/lib/x86_64-linux-gnu/libstdc++.so.6'
sys.path.append('/home/naoto/docker_workspace/MobilePoser/third_party/pyOrbSlam3/pyOrbSlam3/build')

import pyOrbSlam

# Paths
vocab_path = "/home/naoto/docker_workspace/MobilePoser/third_party/pyOrbSlam3/pyOrbSlam3/modules/ORB_SLAM3/Vocabulary/ORBvoc.txt"
settings_path = "/home/naoto/docker_workspace/MobilePoser/mobileposer/slam_configs/nymeria_mono.yaml"

print("Testing ORB-SLAM3 initialization...")
print(f"Vocabulary: {vocab_path}")
print(f"Settings: {settings_path}")
print(f"Vocabulary exists: {os.path.exists(vocab_path)}")
print(f"Settings exists: {os.path.exists(settings_path)}")

try:
    # Try to create ORB-SLAM3 instance
    print("\nCreating ORB-SLAM3 instance...")
    slam = pyOrbSlam.OrbSlam(vocab_path, settings_path, "Mono", False)
    print("✓ ORB-SLAM3 created successfully!")
    
    # Test processing a dummy frame
    print("\nTesting with dummy frame...")
    frame = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
    timestamp = 0.0
    
    pose = slam.process(frame, timestamp)
    print(f"Pose result: {pose}")
    print(f"Tracking state: {slam.GetTrackingState()}")
    
    # Shutdown
    slam.Shutdown()
    print("\n✓ Test completed successfully!")
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()