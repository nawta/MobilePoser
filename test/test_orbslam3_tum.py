#!/usr/bin/env python3
"""Test ORB-SLAM3 with a known working config."""

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
settings_path = "/home/naoto/docker_workspace/MobilePoser/third_party/pyOrbSlam3/pyOrbSlam3/modules/ORB_SLAM3/Examples/Monocular/TUM1.yaml"

print("Testing ORB-SLAM3 with TUM1 config...")
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
    # TUM1 uses 640x480 resolution
    frame = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
    timestamp = 0.0
    
    pose = slam.process(frame, timestamp)
    print(f"Pose result type: {type(pose)}")
    if pose is not None:
        print(f"Pose shape: {pose.shape if hasattr(pose, 'shape') else 'N/A'}")
    print(f"Tracking state: {slam.GetTrackingState()}")
    
    # Process a few more frames
    print("\nProcessing more frames...")
    for i in range(5):
        frame = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        timestamp = i * 0.1
        pose = slam.process(frame, timestamp)
        state = slam.GetTrackingState()
        print(f"Frame {i}: tracking state = {state}")
    
    # Shutdown
    print("\nShutting down...")
    slam.shutdown()  # lowercase 's'
    print("✓ Test completed successfully!")
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()