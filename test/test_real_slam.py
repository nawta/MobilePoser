#!/usr/bin/env python3
"""
Test real ORB-SLAM3 functionality without mock implementation.
"""

import numpy as np
import cv2
from pathlib import Path
import sys
import time

sys.path.append('/home/naoto/docker_workspace/MobilePoser')

from mobileposer.models.real_orbslam3 import RealOrbSlam3Interface, HAS_PYORBSLAM

def create_test_image():
    """Create a test image with features."""
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Add checkerboard pattern for features
    square_size = 40
    for i in range(0, 480, square_size):
        for j in range(0, 640, square_size):
            if (i // square_size + j // square_size) % 2 == 0:
                image[i:i+square_size, j:j+square_size] = 255
                
    # Add some circles
    for _ in range(15):
        x = np.random.randint(50, 590)
        y = np.random.randint(50, 430)
        radius = np.random.randint(10, 30)
        cv2.circle(image, (x, y), radius, (128, 128, 128), 2)
        
    return image

def test_real_slam():
    """Test real ORB-SLAM3."""
    print("=" * 60)
    print("Real ORB-SLAM3 Test")
    print("=" * 60)
    
    # Check if pyOrbSlam is available
    print(f"\npyOrbSlam available: {HAS_PYORBSLAM}")
    
    if not HAS_PYORBSLAM:
        print("✗ pyOrbSlam not available. Cannot test real SLAM.")
        return False
        
    # Create SLAM interface
    print("\nCreating RealOrbSlam3Interface...")
    slam = RealOrbSlam3Interface(
        mode="monocular",
        enable_viewer=False  # Disable viewer for headless operation
    )
    
    # Check paths
    print(f"\nVocabulary path: {slam.vocabulary_path}")
    print(f"  Exists: {Path(slam.vocabulary_path).exists()}")
    
    # Initialize SLAM
    print("\nInitializing ORB-SLAM3...")
    try:
        if slam.initialize():
            print("✅ ORB-SLAM3 initialized successfully!")
        else:
            print("✗ Failed to initialize ORB-SLAM3")
            return False
    except Exception as e:
        print(f"✗ Error initializing: {e}")
        return False
        
    # Process test frames
    print("\nProcessing test frames...")
    test_image = create_test_image()
    
    for i in range(10):
        timestamp = i * 0.033  # 30 FPS
        
        # Process frame
        result = slam.process_frame(test_image, timestamp)
        
        if result:
            print(f"\nFrame {i}:")
            print(f"  Tracking state: {slam.tracking_state}")
            print(f"  Confidence: {result['confidence']:.3f}")
            if result['pose'] is not None:
                print(f"  Pose shape: {result['pose'].shape}")
                # Print first row of pose matrix
                print(f"  Pose[0]: {result['pose'][0]}")
        else:
            print(f"\nFrame {i}: No result")
            
        # Add slight motion to the image for next frame
        test_image = np.roll(test_image, 5, axis=1)  # Shift horizontally
        
    # Get trajectory
    print("\nGetting trajectory...")
    trajectory = slam.get_trajectory()
    if trajectory is not None:
        print(f"✅ Trajectory shape: {trajectory.shape}")
    else:
        print("✗ No trajectory available")
        
    # Shutdown
    print("\nShutting down SLAM...")
    slam.shutdown()
    print("✅ SLAM shutdown complete")
    
    return True

def test_slam_factory():
    """Test SLAM factory with real implementation."""
    print("\n" + "=" * 60)
    print("Testing SLAM Factory")
    print("=" * 60)
    
    from mobileposer.models.slam import create_slam_interface
    
    try:
        # Create real SLAM
        print("\nCreating real SLAM interface...")
        slam = create_slam_interface("real")
        print(f"✅ Created: {type(slam).__name__}")
        
        # Initialize
        if slam.initialize():
            print("✅ Real SLAM initialized")
            
            # Process a test frame
            test_image = create_test_image()
            result = slam.process_frame(test_image, 0.0)
            
            if result:
                print("✅ Real SLAM processed frame successfully")
            else:
                print("⚠️  Real SLAM returned no result (may need more frames)")
                
            slam.shutdown()
        else:
            print("✗ Failed to initialize real SLAM")
            
    except Exception as e:
        print(f"✗ Error: {e}")

if __name__ == "__main__":
    # Test real SLAM
    success = test_real_slam()
    
    # Test factory
    test_slam_factory()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ Real ORB-SLAM3 is working! No more mock SLAM needed!")
    else:
        print("✗ Real ORB-SLAM3 test failed")
    print("=" * 60)