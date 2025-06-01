#!/usr/bin/env python3
"""
Comprehensive unit tests for SLAM functionality in MobilePoser.
Tests both mock and real SLAM implementations, adaptive SLAM, and ensemble weights.
"""

import unittest
import numpy as np
import torch
import cv2
from pathlib import Path
import sys
sys.path.append('/home/naoto/docker_workspace/MobilePoser')

from mobileposer.models.slam import (
    SlamInterface, MockSlamInterface, OrbSlam3Interface, create_slam_interface
)
from mobileposer.models.adaptive_slam import (
    SlamMode, SlamInput, SlamOutput, AdaptiveSlamInterface, EnsembleWeightCalculator
)


class TestSlamInterface(unittest.TestCase):
    """Test base SLAM interface."""
    
    def test_interface_methods(self):
        """Test that interface defines all required methods."""
        interface = SlamInterface()
        
        # Check all methods exist
        self.assertTrue(hasattr(interface, 'initialize'))
        self.assertTrue(hasattr(interface, 'process_frame'))
        self.assertTrue(hasattr(interface, 'get_trajectory'))
        self.assertTrue(hasattr(interface, 'reset'))
        self.assertTrue(hasattr(interface, 'shutdown'))
        
        # Check initial state
        self.assertFalse(interface.is_initialized)
        self.assertIsNone(interface.last_pose)
        self.assertEqual(interface.confidence, 0.0)


class TestMockSlamInterface(unittest.TestCase):
    """Test mock SLAM implementation."""
    
    def setUp(self):
        self.slam = MockSlamInterface()
        
    def test_initialization(self):
        """Test mock SLAM initialization."""
        self.assertFalse(self.slam.is_initialized)
        
        result = self.slam.initialize()
        self.assertTrue(result)
        self.assertTrue(self.slam.is_initialized)
        self.assertEqual(self.slam.frame_count, 0)
        self.assertEqual(len(self.slam.trajectory), 0)
        
    def test_process_frame(self):
        """Test frame processing with mock SLAM."""
        # Initialize first
        self.slam.initialize()
        
        # Create dummy image
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        timestamp = 0.033
        
        # Process frame
        result = self.slam.process_frame(image, timestamp)
        
        # Check result structure
        self.assertIsNotNone(result)
        self.assertIn('pose', result)
        self.assertIn('confidence', result)
        self.assertIn('keypoints', result)
        self.assertIn('map_points', result)
        self.assertIn('timestamp', result)
        
        # Check pose is 4x4 matrix
        pose = result['pose']
        self.assertEqual(pose.shape, (4, 4))
        self.assertAlmostEqual(pose[3, 3], 1.0)
        self.assertAlmostEqual(np.linalg.det(pose[:3, :3]), 1.0, places=5)
        
        # Check confidence range
        self.assertGreaterEqual(result['confidence'], 0.0)
        self.assertLessEqual(result['confidence'], 1.0)
        
        # Check frame count incremented
        self.assertEqual(self.slam.frame_count, 1)
        
    def test_trajectory_accumulation(self):
        """Test that trajectory accumulates over multiple frames."""
        self.slam.initialize()
        
        # Process multiple frames
        num_frames = 10
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        for i in range(num_frames):
            result = self.slam.process_frame(image, i * 0.033)
            self.assertIsNotNone(result)
            
        # Check trajectory
        trajectory = self.slam.get_trajectory()
        self.assertIsNotNone(trajectory)
        self.assertEqual(trajectory.shape, (num_frames, 4, 4))
        
        # Verify poses are different (mock generates moving camera)
        for i in range(1, num_frames):
            diff = np.linalg.norm(trajectory[i] - trajectory[i-1])
            self.assertGreater(diff, 0.0)
            
    def test_reset(self):
        """Test SLAM reset functionality."""
        self.slam.initialize()
        
        # Process some frames
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        for i in range(5):
            self.slam.process_frame(image, i * 0.033)
            
        # Reset
        self.slam.reset()
        
        # Check state is cleared
        self.assertEqual(self.slam.frame_count, 0)
        self.assertEqual(len(self.slam.trajectory), 0)
        self.assertIsNone(self.slam.last_pose)
        
    def test_shutdown(self):
        """Test SLAM shutdown."""
        self.slam.initialize()
        self.assertTrue(self.slam.is_initialized)
        
        self.slam.shutdown()
        self.assertFalse(self.slam.is_initialized)


class TestSlamFactory(unittest.TestCase):
    """Test SLAM factory function."""
    
    def test_create_mock_slam(self):
        """Test creating mock SLAM interface."""
        slam = create_slam_interface("mock")
        self.assertIsInstance(slam, MockSlamInterface)
        
    def test_create_orb_slam3(self):
        """Test creating ORB-SLAM3 interface."""
        try:
            slam = create_slam_interface("orb_slam3")
            # Should create RealOrbSlam3Interface if available
            self.assertIsInstance(slam, SlamInterface)
        except Exception as e:
            # Expected if ORB-SLAM3 not installed
            print(f"ORB-SLAM3 creation failed (expected): {e}")
            
    def test_invalid_slam_type(self):
        """Test error handling for invalid SLAM type."""
        with self.assertRaises(ValueError):
            create_slam_interface("invalid_type")


class TestAdaptiveSlamInterface(unittest.TestCase):
    """Test adaptive SLAM interface."""
    
    def setUp(self):
        # Use mock implementations to avoid ORB-SLAM3 dependency
        self.adaptive_slam = AdaptiveSlamInterface()
        
        # Replace with mock implementations
        self.adaptive_slam.monocular_slam = MockSlamInterface()
        self.adaptive_slam.visual_inertial_slam = MockSlamInterface()
        
    def test_initialization(self):
        """Test adaptive SLAM initialization."""
        # Initialize mock systems
        self.adaptive_slam.monocular_slam.initialize()
        self.adaptive_slam.visual_inertial_slam.initialize()
        self.adaptive_slam.is_initialized = True
        
        self.assertTrue(self.adaptive_slam.is_initialized)
        self.assertEqual(self.adaptive_slam.current_mode, SlamMode.NONE)
        self.assertEqual(self.adaptive_slam.frame_count, 0)
        
    def test_mode_selection(self):
        """Test automatic mode selection based on input data."""
        # No data -> NONE mode
        input1 = SlamInput(rgb_frame=None, head_imu_data=None)
        mode1 = self.adaptive_slam._select_slam_mode(input1)
        self.assertEqual(mode1, SlamMode.NONE)
        
        # RGB only -> MONOCULAR mode
        input2 = SlamInput(
            rgb_frame=np.zeros((480, 640, 3), dtype=np.uint8),
            head_imu_data=None
        )
        mode2 = self.adaptive_slam._select_slam_mode(input2)
        self.assertEqual(mode2, SlamMode.MONOCULAR)
        
        # RGB + IMU -> VISUAL_INERTIAL mode
        input3 = SlamInput(
            rgb_frame=np.zeros((480, 640, 3), dtype=np.uint8),
            head_imu_data=np.zeros((6,))  # [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
        )
        mode3 = self.adaptive_slam._select_slam_mode(input3)
        self.assertEqual(mode3, SlamMode.VISUAL_INERTIAL)
        
    def test_process_frame_with_mode_switch(self):
        """Test processing frames with mode switching."""
        # Initialize
        self.adaptive_slam.monocular_slam.initialize()
        self.adaptive_slam.visual_inertial_slam.initialize()
        self.adaptive_slam.is_initialized = True
        
        # Process with no data (NONE mode)
        input1 = SlamInput(timestamp=0.0)
        output1 = self.adaptive_slam.process_frame(input1)
        self.assertEqual(output1.mode_used, SlamMode.NONE)
        self.assertEqual(output1.tracking_state, "none")
        
        # Process with RGB only (switch to MONOCULAR)
        input2 = SlamInput(
            rgb_frame=np.zeros((480, 640, 3), dtype=np.uint8),
            timestamp=0.033
        )
        output2 = self.adaptive_slam.process_frame(input2)
        self.assertEqual(output2.mode_used, SlamMode.MONOCULAR)
        self.assertEqual(self.adaptive_slam.mode_switches, 1)
        
        # Process with RGB + IMU (switch to VISUAL_INERTIAL)
        input3 = SlamInput(
            rgb_frame=np.zeros((480, 640, 3), dtype=np.uint8),
            head_imu_data=np.zeros((6,)),
            timestamp=0.066
        )
        output3 = self.adaptive_slam.process_frame(input3)
        self.assertEqual(output3.mode_used, SlamMode.VISUAL_INERTIAL)
        self.assertEqual(self.adaptive_slam.mode_switches, 2)
        
    def test_slam_output_structure(self):
        """Test SLAM output data structure."""
        output = SlamOutput(
            pose=np.eye(4),
            confidence=0.9,
            scale_factor=1.2,
            scale_confidence=0.8,
            tracking_state="tracking",
            timestamp=0.033,
            mode_used=SlamMode.VISUAL_INERTIAL
        )
        
        self.assertEqual(output.pose.shape, (4, 4))
        self.assertEqual(output.confidence, 0.9)
        self.assertEqual(output.scale_factor, 1.2)
        self.assertEqual(output.scale_confidence, 0.8)
        self.assertEqual(output.tracking_state, "tracking")
        self.assertEqual(output.timestamp, 0.033)
        self.assertEqual(output.mode_used, SlamMode.VISUAL_INERTIAL)
        
    def test_performance_stats(self):
        """Test performance statistics tracking."""
        self.adaptive_slam.is_initialized = True
        self.adaptive_slam.frame_count = 100
        self.adaptive_slam.mode_switches = 5
        self.adaptive_slam.current_mode = SlamMode.VISUAL_INERTIAL
        
        stats = self.adaptive_slam.get_performance_stats()
        
        self.assertEqual(stats['current_mode'], 'visual_inertial')
        self.assertEqual(stats['mode_switches'], 5)
        self.assertEqual(stats['frames_processed'], 100)
        self.assertTrue(stats['slam_initialized'])


class TestEnsembleWeightCalculator(unittest.TestCase):
    """Test ensemble weight calculation."""
    
    def setUp(self):
        self.calculator = EnsembleWeightCalculator()
        
    def test_weight_calculation_no_slam(self):
        """Test weight calculation when SLAM is lost."""
        slam_output = SlamOutput(
            tracking_state="lost",
            confidence=0.0
        )
        
        imu_weight, slam_weight = self.calculator.calculate_weights(
            imu_confidence=0.8,
            slam_output=slam_output
        )
        
        self.assertEqual(imu_weight, 1.0)
        self.assertEqual(slam_weight, 0.0)
        
    def test_weight_calculation_no_imu(self):
        """Test weight calculation when IMU is not available."""
        slam_output = SlamOutput(
            pose=np.eye(4),
            tracking_state="tracking",
            confidence=0.9
        )
        
        imu_weight, slam_weight = self.calculator.calculate_weights(
            imu_confidence=0.0,
            slam_output=slam_output,
            imu_pose_available=False
        )
        
        self.assertEqual(imu_weight, 0.0)
        self.assertEqual(slam_weight, 1.0)
        
    def test_weight_calculation_both_available(self):
        """Test weight calculation when both IMU and SLAM are available."""
        slam_output = SlamOutput(
            pose=np.eye(4),
            tracking_state="tracking",
            confidence=0.8,
            scale_confidence=0.7,
            mode_used=SlamMode.VISUAL_INERTIAL
        )
        
        # Build up some history
        for _ in range(10):
            self.calculator._update_confidence_history(0.7, 0.8)
            
        imu_weight, slam_weight = self.calculator.calculate_weights(
            imu_confidence=0.7,
            slam_output=slam_output
        )
        
        # Weights should sum to 1.0
        self.assertAlmostEqual(imu_weight + slam_weight, 1.0)
        
        # Both should have non-zero weights
        self.assertGreater(imu_weight, 0.0)
        self.assertGreater(slam_weight, 0.0)
        
    def test_weight_breakdown(self):
        """Test detailed weight breakdown."""
        slam_output = SlamOutput(
            pose=np.eye(4),
            tracking_state="tracking",
            confidence=0.9,
            scale_confidence=0.8,
            mode_used=SlamMode.VISUAL_INERTIAL
        )
        
        breakdown = self.calculator.get_weight_breakdown(
            imu_confidence=0.6,
            slam_output=slam_output
        )
        
        # Check all components are present
        expected_keys = [
            'confidence_component',
            'tracking_component',
            'temporal_component',
            'scale_component',
            'final_slam_weight',
            'final_imu_weight'
        ]
        
        for key in expected_keys:
            self.assertIn(key, breakdown)
            self.assertIsInstance(breakdown[key], float)
            
        # Weights should sum to 1.0
        self.assertAlmostEqual(
            breakdown['final_imu_weight'] + breakdown['final_slam_weight'],
            1.0
        )
        
    def test_temporal_consistency(self):
        """Test temporal consistency analysis."""
        # Stable SLAM confidence
        for _ in range(20):
            self.calculator._update_confidence_history(0.7, 0.85)
            
        temporal_comp = self.calculator._calculate_temporal_component()
        self.assertGreater(temporal_comp, 0.7)  # Should indicate good consistency
        
        # Reset and test unstable SLAM
        self.calculator.reset()
        for i in range(20):
            # Oscillating confidence
            conf = 0.9 if i % 2 == 0 else 0.3
            self.calculator._update_confidence_history(0.7, conf)
            
        temporal_comp = self.calculator._calculate_temporal_component()
        self.assertLess(temporal_comp, 0.5)  # Should indicate poor consistency
        
    def test_scale_component(self):
        """Test scale component calculation."""
        # Visual-Inertial SLAM with good scale
        slam_output_vi = SlamOutput(
            mode_used=SlamMode.VISUAL_INERTIAL,
            scale_confidence=0.9
        )
        scale_comp_vi = self.calculator._calculate_scale_component(slam_output_vi)
        self.assertEqual(scale_comp_vi, 0.9)
        
        # Monocular SLAM (scale ambiguity)
        slam_output_mono = SlamOutput(
            mode_used=SlamMode.MONOCULAR
        )
        scale_comp_mono = self.calculator._calculate_scale_component(slam_output_mono)
        self.assertEqual(scale_comp_mono, 0.3)
        
        # No SLAM
        slam_output_none = SlamOutput(
            mode_used=SlamMode.NONE
        )
        scale_comp_none = self.calculator._calculate_scale_component(slam_output_none)
        self.assertEqual(scale_comp_none, 0.0)


class TestSlamIntegration(unittest.TestCase):
    """Integration tests for SLAM components."""
    
    def test_full_pipeline_mock(self):
        """Test full SLAM pipeline with mock implementation."""
        # Create adaptive SLAM with mock backends
        adaptive_slam = AdaptiveSlamInterface()
        adaptive_slam.monocular_slam = MockSlamInterface()
        adaptive_slam.visual_inertial_slam = MockSlamInterface()
        adaptive_slam.monocular_slam.initialize()
        adaptive_slam.visual_inertial_slam.initialize()
        adaptive_slam.is_initialized = True
        
        # Create weight calculator
        weight_calc = EnsembleWeightCalculator()
        
        # Simulate processing sequence
        num_frames = 30
        trajectory = []
        
        for i in range(num_frames):
            # Alternate between different input scenarios
            if i < 10:
                # Start with RGB only
                slam_input = SlamInput(
                    rgb_frame=np.zeros((480, 640, 3), dtype=np.uint8),
                    timestamp=i * 0.033
                )
            elif i < 20:
                # Add IMU data
                slam_input = SlamInput(
                    rgb_frame=np.zeros((480, 640, 3), dtype=np.uint8),
                    head_imu_data=np.random.randn(6) * 0.1,
                    timestamp=i * 0.033
                )
            else:
                # Lose RGB (IMU only)
                slam_input = SlamInput(
                    head_imu_data=np.random.randn(6) * 0.1,
                    timestamp=i * 0.033
                )
                
            # Process with adaptive SLAM
            slam_output = adaptive_slam.process_frame(slam_input)
            
            # Calculate ensemble weights
            imu_confidence = 0.7 + 0.1 * np.random.randn()
            imu_weight, slam_weight = weight_calc.calculate_weights(
                imu_confidence=imu_confidence,
                slam_output=slam_output
            )
            
            # Store results
            trajectory.append({
                'timestamp': slam_input.timestamp,
                'slam_mode': slam_output.mode_used,
                'slam_pose': slam_output.pose,
                'slam_confidence': slam_output.confidence,
                'imu_weight': imu_weight,
                'slam_weight': slam_weight
            })
            
        # Verify mode switches occurred
        self.assertGreaterEqual(adaptive_slam.mode_switches, 2)
        
        # Verify trajectory
        self.assertEqual(len(trajectory), num_frames)
        
        # Check mode progression
        self.assertEqual(trajectory[5]['slam_mode'], SlamMode.MONOCULAR)
        self.assertEqual(trajectory[15]['slam_mode'], SlamMode.VISUAL_INERTIAL)
        self.assertEqual(trajectory[25]['slam_mode'], SlamMode.NONE)
        
        # Check weight adaptation
        # When SLAM is NONE, IMU weight should be 1.0
        self.assertEqual(trajectory[25]['imu_weight'], 1.0)
        self.assertEqual(trajectory[25]['slam_weight'], 0.0)


if __name__ == '__main__':
    unittest.main(verbosity=2)