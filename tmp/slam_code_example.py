#!/usr/bin/env python3
"""
SLAM Processing Code Example - Shows actual usage
"""

import numpy as np
import cv2
from mobileposer.models.slam import create_slam_interface
from mobileposer.models.adaptive_slam import AdaptiveSlamInterface, SlamInput

def simple_slam_example():
    """Basic SLAM usage example."""
    print("=" * 60)
    print("BASIC SLAM EXAMPLE")
    print("=" * 60)
    
    # 1. Create SLAM interface
    slam = create_slam_interface("real")  # Creates real ORB-SLAM3
    
    # 2. Initialize SLAM
    print("\n1. Initializing SLAM...")
    slam.initialize()
    print("   ✓ Vocabulary loaded")
    print("   ✓ Camera calibration loaded")
    print("   ✓ Ready to process frames")
    
    # 3. Process a frame
    print("\n2. Processing RGB frame...")
    # In real usage, this would be an actual camera frame
    dummy_frame = np.ones((480, 640, 3), dtype=np.uint8) * 128  # Gray image
    timestamp = 0.0
    
    result = slam.process_frame(dummy_frame, timestamp)
    
    print("\n3. SLAM Output:")
    print(f"   - Has pose: {result['pose'] is not None}")
    print(f"   - Confidence: {result['confidence']:.3f}")
    print(f"   - Tracking state: {result.get('tracking_state', 'unknown')}")
    
    if result['pose'] is not None:
        print("\n   4x4 Pose Matrix:")
        print("   " + "-" * 40)
        for row in result['pose']:
            print(f"   [{row[0]:7.4f} {row[1]:7.4f} {row[2]:7.4f} {row[3]:7.4f}]")
        
        # Extract position and rotation
        position = result['pose'][:3, 3]
        rotation_matrix = result['pose'][:3, :3]
        
        print(f"\n   Position (X,Y,Z): ({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}) meters")
    
    # 4. Shutdown
    slam.shutdown()
    print("\n4. SLAM shutdown complete")

def adaptive_slam_example():
    """Adaptive SLAM with sensor fusion example."""
    print("\n" + "=" * 60)
    print("ADAPTIVE SLAM EXAMPLE")
    print("=" * 60)
    
    # 1. Create adaptive SLAM
    adaptive = AdaptiveSlamInterface()
    adaptive.initialize()
    print("\n1. Adaptive SLAM initialized")
    print("   ✓ Monocular SLAM ready")
    print("   ✓ Visual-Inertial SLAM ready")
    print("   ✓ Mode selection automatic")
    
    # 2. Process with different sensor configurations
    scenarios = [
        ("RGB only", True, False),
        ("RGB + IMU", True, True),
        ("IMU only", False, True),
        ("No sensors", False, False),
    ]
    
    print("\n2. Testing different sensor configurations:")
    print("-" * 50)
    
    for name, has_rgb, has_imu in scenarios:
        # Prepare input
        rgb_frame = np.ones((480, 640, 3), dtype=np.uint8) * 128 if has_rgb else None
        imu_data = np.array([0.1, 0.2, 9.8, 0.01, 0.02, 0.03]) if has_imu else None
        
        slam_input = SlamInput(
            rgb_frame=rgb_frame,
            head_imu_data=imu_data,
            timestamp=0.0
        )
        
        # Process
        output = adaptive.process_frame(slam_input)
        
        print(f"\n   {name}:")
        print(f"   - Mode used: {output.mode_used.value}")
        print(f"   - Has pose: {output.pose is not None}")
        print(f"   - Confidence: {output.confidence:.3f}")
        print(f"   - Scale confidence: {output.scale_confidence:.3f}")
    
    # 3. Show statistics
    stats = adaptive.get_statistics()
    print("\n3. Adaptive SLAM Statistics:")
    print(f"   - Frames processed: {stats['frames_processed']}")
    print(f"   - Mode switches: {stats['mode_switches']}")
    print(f"   - Current mode: {stats['current_mode']}")
    
    adaptive.shutdown()
    print("\n4. Adaptive SLAM shutdown complete")

def ensemble_fusion_example():
    """Example showing ensemble weight calculation."""
    print("\n" + "=" * 60)
    print("ENSEMBLE FUSION EXAMPLE")
    print("=" * 60)
    
    from mobileposer.models.adaptive_slam import EnsembleWeightCalculator
    
    calc = EnsembleWeightCalculator()
    
    print("\n1. Weight calculation for different scenarios:")
    print("-" * 50)
    
    # Scenario 1: SLAM lost
    print("\n   Scenario 1: SLAM Lost")
    slam_output = type('SlamOutput', (), {
        'confidence': 0.0,
        'tracking_state': 'lost',
        'pose': None,
        'mode_used': type('SlamMode', (), {'value': 'visual_inertial'})(),
        'scale_confidence': 0.0
    })()
    
    imu_conf = 0.8
    imu_weight, slam_weight = calc.calculate_weights(imu_conf, slam_output, True)
    print(f"   - IMU confidence: {imu_conf:.2f}")
    print(f"   - SLAM confidence: {slam_output.confidence:.2f}")
    print(f"   → IMU weight: {imu_weight:.2f}, SLAM weight: {slam_weight:.2f}")
    
    # Scenario 2: Both good
    print("\n   Scenario 2: Both tracking well")
    slam_output.confidence = 0.85
    slam_output.tracking_state = 'tracking'
    slam_output.pose = np.eye(4)
    slam_output.scale_confidence = 0.9
    
    imu_weight, slam_weight = calc.calculate_weights(imu_conf, slam_output, True)
    print(f"   - IMU confidence: {imu_conf:.2f}")
    print(f"   - SLAM confidence: {slam_output.confidence:.2f}")
    print(f"   → IMU weight: {imu_weight:.2f}, SLAM weight: {slam_weight:.2f}")
    
    # Show weight components
    print("\n2. Weight component breakdown:")
    components = calc._get_weight_components(imu_conf, slam_output, True)
    print(f"   - Confidence factor: IMU={components['confidence'][0]:.3f}, SLAM={components['confidence'][1]:.3f}")
    print(f"   - Tracking state factor: IMU={components['tracking_state'][0]:.3f}, SLAM={components['tracking_state'][1]:.3f}")
    print(f"   - Temporal consistency: IMU={components['temporal_consistency'][0]:.3f}, SLAM={components['temporal_consistency'][1]:.3f}")
    print(f"   - Scale quality: IMU={components['scale_quality'][0]:.3f}, SLAM={components['scale_quality'][1]:.3f}")

def data_format_example():
    """Show actual data formats and structures."""
    print("\n" + "=" * 60)
    print("DATA FORMAT EXAMPLE")
    print("=" * 60)
    
    print("\n1. Input Data Formats:")
    print("-" * 30)
    
    print("\n   RGB Frame:")
    print("   - Type: numpy.ndarray")
    print("   - Shape: (480, 640, 3)")
    print("   - Dtype: uint8")
    print("   - Range: 0-255")
    print("   - Format: BGR (OpenCV convention)")
    
    print("\n   IMU Data:")
    print("   - Type: numpy.ndarray") 
    print("   - Shape: (6,)")
    print("   - Contents: [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]")
    print("   - Units: m/s² for accelerometer, rad/s for gyroscope")
    print("   - Example: [0.1, 0.2, 9.81, 0.01, 0.02, 0.03]")
    
    print("\n2. Output Data Formats:")
    print("-" * 30)
    
    print("\n   Pose Matrix (SE3):")
    print("   - Type: numpy.ndarray")
    print("   - Shape: (4, 4)")
    print("   - Structure:")
    print("     [[r11, r12, r13, tx],")
    print("      [r21, r22, r23, ty],")
    print("      [r31, r32, r33, tz],")
    print("      [0,   0,   0,   1 ]]")
    print("   - R (3x3): Rotation matrix")
    print("   - t (3x1): Translation vector in meters")
    
    print("\n   6DOF Head Pose:")
    print("   - Position: (X, Y, Z) in meters")
    print("   - Orientation: (Roll, Pitch, Yaw) in radians")
    print("   - Extracted from pose matrix")
    
    print("\n   Confidence Score:")
    print("   - Range: 0.0 to 1.0")
    print("   - 0.0: No confidence (lost tracking)")
    print("   - 0.2: Initializing")
    print("   - 0.5+: Good tracking")
    print("   - 0.8+: Excellent tracking")

if __name__ == '__main__':
    print("SLAM PROCESSING CODE EXAMPLES\n")
    
    # Run examples
    try:
        simple_slam_example()
    except Exception as e:
        print(f"\nNote: Real SLAM example requires ORB-SLAM3 setup: {e}")
    
    adaptive_slam_example()
    ensemble_fusion_example()
    data_format_example()
    
    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)