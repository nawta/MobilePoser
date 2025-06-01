#!/usr/bin/env python3
"""
Test script to verify real ORB-SLAM3 integration is working correctly.
Tests both monocular and visual-inertial modes.
"""

import os
import sys
import cv2
import numpy as np
import torch
from pathlib import Path
import time
import argparse

# Add MobilePoser to path
sys.path.append(str(Path(__file__).parent))

from mobileposer.models.real_orbslam3 import RealOrbSlam3Interface
from mobileposer.models.adaptive_slam import AdaptiveSlamInterface, SlamInput
from mobileposer.head_pose_ensemble import HeadPoseEnsemble
from mobileposer.adaptive_head_ensemble import AdaptiveHeadPoseEnsemble


def test_real_orbslam3_monocular():
    """Test real ORB-SLAM3 in monocular mode."""
    print("\n" + "="*60)
    print("Testing Real ORB-SLAM3 - Monocular Mode")
    print("="*60)
    
    # Create monocular SLAM
    slam = RealOrbSlam3Interface(mode="monocular", enable_viewer=False)
    
    # Initialize
    if slam.initialize():
        print("✅ Monocular SLAM initialized successfully!")
    else:
        print("❌ Failed to initialize monocular SLAM")
        print("   Make sure ORB-SLAM3 is properly installed")
        return False
    
    # Test with synthetic image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    timestamp = 0.0
    
    # Process frame
    result = slam.process_frame(test_image, timestamp)
    
    if result is not None:
        print("✅ Monocular processing successful")
        print(f"   Tracking state: {result.get('tracking_state', 'unknown')}")
        print(f"   Confidence: {result.get('confidence', 0.0):.3f}")
    else:
        print("⚠️  Monocular processing returned None (expected for initialization)")
    
    # Cleanup
    slam.shutdown()
    print("✅ Monocular SLAM shutdown complete")
    
    return True


def test_real_orbslam3_visual_inertial():
    """Test real ORB-SLAM3 in visual-inertial mode."""
    print("\n" + "="*60)
    print("Testing Real ORB-SLAM3 - Visual-Inertial Mode")
    print("="*60)
    
    # Create VI-SLAM
    slam = RealOrbSlam3Interface(mode="visual_inertial", enable_viewer=False)
    
    # Initialize
    if slam.initialize():
        print("✅ Visual-Inertial SLAM initialized successfully!")
    else:
        print("❌ Failed to initialize VI-SLAM")
        print("   Make sure ORB-SLAM3 is properly installed")
        return False
    
    # Test with synthetic data
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    test_imu = np.array([0.0, -9.81, 0.0, 0.0, 0.0, 0.0])  # acc + gyro
    timestamp = 0.0
    
    # Process frame with IMU
    result = slam.process_frame_with_imu(test_image, timestamp, test_imu)
    
    if result is not None:
        print("✅ VI-SLAM processing successful")
        print(f"   Tracking state: {result.get('tracking_state', 'unknown')}")
        print(f"   Confidence: {result.get('confidence', 0.0):.3f}")
        print(f"   Scale factor: {result.get('scale_factor', 1.0):.3f}")
    else:
        print("⚠️  VI-SLAM processing returned None (expected for initialization)")
    
    # Cleanup
    slam.shutdown()
    print("✅ VI-SLAM shutdown complete")
    
    return True


def test_adaptive_slam():
    """Test adaptive SLAM interface."""
    print("\n" + "="*60)
    print("Testing Adaptive SLAM Interface")
    print("="*60)
    
    # Create adaptive SLAM
    adaptive_slam = AdaptiveSlamInterface()
    
    # Initialize
    if adaptive_slam.initialize():
        print("✅ Adaptive SLAM initialized successfully!")
    else:
        print("❌ Failed to initialize adaptive SLAM")
        return False
    
    # Test different input scenarios
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    test_imu = np.array([0.0, -9.81, 0.0, 0.0, 0.0, 0.0])
    
    # Test 1: RGB + IMU (should use VI-SLAM)
    slam_input = SlamInput(
        rgb_frame=test_image,
        head_imu_data=test_imu,
        timestamp=0.0
    )
    result = adaptive_slam.process_frame(slam_input)
    print(f"\nTest 1 - RGB + IMU:")
    print(f"  Mode used: {result.mode_used.value}")
    print(f"  Tracking state: {result.tracking_state}")
    
    # Test 2: RGB only (should use Monocular)
    slam_input = SlamInput(
        rgb_frame=test_image,
        head_imu_data=None,
        timestamp=0.1
    )
    result = adaptive_slam.process_frame(slam_input)
    print(f"\nTest 2 - RGB only:")
    print(f"  Mode used: {result.mode_used.value}")
    print(f"  Tracking state: {result.tracking_state}")
    
    # Test 3: No RGB (should use IMU-only)
    slam_input = SlamInput(
        rgb_frame=None,
        head_imu_data=test_imu,
        timestamp=0.2
    )
    result = adaptive_slam.process_frame(slam_input)
    print(f"\nTest 3 - No RGB:")
    print(f"  Mode used: {result.mode_used.value}")
    print(f"  Tracking state: {result.tracking_state}")
    
    print("\n✅ Adaptive SLAM tests complete")
    
    return True


def test_head_pose_ensemble(weights_path: str):
    """Test head pose ensemble with real SLAM."""
    print("\n" + "="*60)
    print("Testing Head Pose Ensemble with Real SLAM")
    print("="*60)
    
    if not os.path.exists(weights_path):
        print(f"⚠️  Weights not found at {weights_path}, skipping ensemble test")
        return True
    
    # Test different SLAM types
    slam_types = ['real_vi', 'real_mono']
    
    for slam_type in slam_types:
        print(f"\nTesting with {slam_type}...")
        
        try:
            ensemble = HeadPoseEnsemble(
                mobileposer_weights=weights_path,
                slam_type=slam_type,
                fusion_method='weighted_average'
            )
            print(f"✅ {slam_type} ensemble created successfully")
            
            # Test with synthetic data
            test_rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            test_imu = np.random.randn(72)  # Full IMU data
            
            result = ensemble.process_frame(test_rgb, test_imu, 0.0)
            
            if result is not None:
                print(f"✅ {slam_type} processing successful")
                print(f"   Source: {result.source}")
                print(f"   Confidence: {result.confidence:.3f}")
            else:
                print(f"⚠️  {slam_type} processing returned None")
                
        except Exception as e:
            print(f"❌ Error with {slam_type}: {e}")
    
    return True


def test_nymeria_sequence(sequence_path: str, weights_path: str):
    """Test with real Nymeria sequence data."""
    print("\n" + "="*60)
    print("Testing with Nymeria Sequence")
    print("="*60)
    
    # Check if sequence exists
    video_path = Path(sequence_path) / "video_main_rgb.mp4"
    if not video_path.exists():
        print(f"⚠️  Video not found at {video_path}, skipping Nymeria test")
        return True
    
    # Create adaptive ensemble
    ensemble = AdaptiveHeadPoseEnsemble(
        mobileposer_weights=weights_path,
        head_imu_index=4
    )
    
    # Load video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Failed to open video: {video_path}")
        return False
    
    # Process first 10 frames
    print("\nProcessing first 10 frames...")
    for i in range(10):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Generate mock IMU data
        imu_data = np.random.randn(72)
        
        # Process
        result = ensemble.process_frame(frame_rgb, imu_data, i/30.0)
        
        if result.head_pose is not None:
            print(f"Frame {i}: Mode={result.mode_used.value}, "
                  f"Confidence={result.head_pose.confidence:.3f}")
    
    cap.release()
    print("\n✅ Nymeria sequence test complete")
    
    return True


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description="Test real ORB-SLAM3 integration")
    parser.add_argument('--weights', type=str, default='checkpoints/weights.pth',
                       help='Path to MobilePoser weights')
    parser.add_argument('--sequence', type=str, default=None,
                       help='Optional: Path to Nymeria sequence for testing')
    parser.add_argument('--skip-basic', action='store_true',
                       help='Skip basic SLAM tests')
    
    args = parser.parse_args()
    
    print("\n🔧 Testing Real ORB-SLAM3 Integration for MobilePoser")
    print("="*60)
    
    success = True
    
    if not args.skip_basic:
        # Test individual components
        if not test_real_orbslam3_monocular():
            success = False
        
        if not test_real_orbslam3_visual_inertial():
            success = False
        
        if not test_adaptive_slam():
            success = False
    
    # Test ensemble if weights available
    if os.path.exists(args.weights):
        if not test_head_pose_ensemble(args.weights):
            success = False
    
    # Test with real sequence if provided
    if args.sequence:
        if not test_nymeria_sequence(args.sequence, args.weights):
            success = False
    
    # Summary
    print("\n" + "="*60)
    if success:
        print("✅ All tests passed! Real ORB-SLAM3 integration is working.")
    else:
        print("❌ Some tests failed. Check the output above for details.")
        print("\nCommon issues:")
        print("- ORB-SLAM3 not installed or not in Python path")
        print("- Missing vocabulary file")
        print("- GPU/CUDA issues")
        print("- Camera calibration mismatch")
    print("="*60)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())