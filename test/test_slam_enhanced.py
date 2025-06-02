#!/usr/bin/env python3
"""
Enhanced unit tests for SLAM functionality in MobilePoser.
Includes performance benchmarks, edge case testing, and real SLAM tests (when available).
"""

import unittest
import numpy as np
import torch
import cv2
from pathlib import Path
import sys
import time
import tempfile
import yaml
from unittest.mock import patch, MagicMock

sys.path.append('/home/naoto/docker_workspace/MobilePoser')

from mobileposer.models.slam import (
    SlamInterface, MockSlamInterface, create_slam_interface
)
from mobileposer.models.adaptive_slam import (
    SlamMode, SlamInput, SlamOutput, AdaptiveSlamInterface, EnsembleWeightCalculator
)

# Try to import real ORB-SLAM3
try:
    from mobileposer.models.real_orbslam3 import RealOrbSlam3Interface
    HAS_ORBSLAM3 = True
except ImportError:
    HAS_ORBSLAM3 = False
    print("Note: pyOrbSlam3 not available, skipping real SLAM tests")


class TestSlamPerformance(unittest.TestCase):
    """Performance benchmark tests for SLAM systems."""
    
    def setUp(self):
        self.slam = MockSlamInterface()
        self.slam.initialize()
        
    def test_frame_processing_speed(self):
        """Benchmark frame processing speed."""
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        num_frames = 100
        
        start_time = time.time()
        for i in range(num_frames):
            result = self.slam.process_frame(image, i * 0.033)
            self.assertIsNotNone(result)
        end_time = time.time()
        
        elapsed = end_time - start_time
        fps = num_frames / elapsed
        
        print(f"\nMock SLAM Performance:")
        print(f"  Processed {num_frames} frames in {elapsed:.2f}s")
        print(f"  Average FPS: {fps:.1f}")
        print(f"  Average time per frame: {elapsed/num_frames*1000:.2f}ms")
        
        # Assert reasonable performance (mock should be fast)
        self.assertGreater(fps, 100)  # Should process >100 FPS for mock
        
    def test_memory_usage(self):
        """Test memory usage with large trajectories."""
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Process many frames
        for i in range(1000):
            self.slam.process_frame(image, i * 0.033)
            
        trajectory = self.slam.get_trajectory()
        self.assertEqual(trajectory.shape[0], 1000)
        
        # Check memory efficiency
        trajectory_size = trajectory.nbytes / (1024 * 1024)  # MB
        print(f"\nTrajectory memory usage: {trajectory_size:.2f} MB for 1000 poses")
        
        # Each pose is 4x4 float64 = 128 bytes, so 1000 poses = ~0.122 MB
        self.assertLess(trajectory_size, 1.0)  # Should be less than 1MB


class TestSlamEdgeCases(unittest.TestCase):
    """Test SLAM behavior in edge cases."""
    
    def setUp(self):
        self.slam = MockSlamInterface()
        self.adaptive_slam = AdaptiveSlamInterface()
        
    def test_very_fast_motion(self):
        """Test SLAM with very fast camera motion."""
        self.slam.initialize()
        
        # Simulate fast motion by large pose changes
        poses = []
        for i in range(10):
            # Create image with motion blur simulation
            image = np.random.randint(0, 50, (480, 640, 3), dtype=np.uint8)
            result = self.slam.process_frame(image, i * 0.033)
            
            if result:
                poses.append(result['pose'])
                # Confidence should be lower for fast motion
                self.assertLess(result['confidence'], 0.9)
                
        self.assertEqual(len(poses), 10)
        
    def test_poor_lighting_conditions(self):
        """Test SLAM with poor lighting (dark/bright images)."""
        self.slam.initialize()
        
        # Test very dark image
        dark_image = np.zeros((480, 640, 3), dtype=np.uint8)
        dark_result = self.slam.process_frame(dark_image, 0.0)
        self.assertIsNotNone(dark_result)
        
        # Test very bright image
        bright_image = np.full((480, 640, 3), 255, dtype=np.uint8)
        bright_result = self.slam.process_frame(bright_image, 0.033)
        self.assertIsNotNone(bright_result)
        
        # Test low contrast image
        low_contrast = np.full((480, 640, 3), 128, dtype=np.uint8)
        low_contrast_result = self.slam.process_frame(low_contrast, 0.066)
        self.assertIsNotNone(low_contrast_result)
        
    def test_rapid_mode_switching(self):
        """Test adaptive SLAM with rapid mode switches."""
        # Use mock implementations
        self.adaptive_slam.monocular_slam = MockSlamInterface()
        self.adaptive_slam.visual_inertial_slam = MockSlamInterface()
        self.adaptive_slam.monocular_slam.initialize()
        self.adaptive_slam.visual_inertial_slam.initialize()
        self.adaptive_slam.is_initialized = True
        
        # Simulate rapid sensor availability changes
        mode_sequence = []
        for i in range(20):
            if i % 3 == 0:
                # No sensors
                slam_input = SlamInput(timestamp=i*0.033)
            elif i % 3 == 1:
                # RGB only
                slam_input = SlamInput(
                    rgb_frame=np.zeros((480, 640, 3), dtype=np.uint8),
                    timestamp=i*0.033
                )
            else:
                # RGB + IMU
                slam_input = SlamInput(
                    rgb_frame=np.zeros((480, 640, 3), dtype=np.uint8),
                    head_imu_data=np.zeros((6,)),
                    timestamp=i*0.033
                )
                
            output = self.adaptive_slam.process_frame(slam_input)
            mode_sequence.append(output.mode_used)
            
        # Check that mode switches occurred
        unique_modes = set(mode_sequence)
        self.assertEqual(len(unique_modes), 3)  # All three modes used
        self.assertGreaterEqual(self.adaptive_slam.mode_switches, 10)
        
    def test_corrupted_input_handling(self):
        """Test handling of corrupted or invalid inputs."""
        self.slam.initialize()
        
        # Test with None image
        result = self.slam.process_frame(None, 0.0)
        self.assertIsNone(result)
        
        # Test with wrong image shape
        wrong_shape = np.zeros((100, 100), dtype=np.uint8)
        result = self.slam.process_frame(wrong_shape, 0.0)
        # Mock implementation should handle this gracefully
        
        # Test with negative timestamp
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        result = self.slam.process_frame(image, -1.0)
        self.assertIsNotNone(result)  # Should still work
        
    def test_long_duration_tracking(self):
        """Test SLAM stability over long duration."""
        self.slam.initialize()
        weight_calc = EnsembleWeightCalculator()
        
        # Simulate 5 minutes of tracking
        duration = 300  # seconds
        fps = 30
        total_frames = duration * fps
        
        confidence_history = []
        
        for i in range(0, total_frames, 100):  # Sample every 100 frames
            image = np.zeros((480, 640, 3), dtype=np.uint8)
            result = self.slam.process_frame(image, i / fps)
            
            if result:
                confidence_history.append(result['confidence'])
                
                # Create SLAM output for weight calculation
                slam_output = SlamOutput(
                    pose=result['pose'],
                    confidence=result['confidence'],
                    tracking_state="tracking"
                )
                
                # Update weight calculator history
                weight_calc._update_confidence_history(0.7, result['confidence'])
                
        # Check stability
        avg_confidence = np.mean(confidence_history)
        std_confidence = np.std(confidence_history)
        
        print(f"\nLong duration tracking stats:")
        print(f"  Duration: {duration}s")
        print(f"  Frames sampled: {len(confidence_history)}")
        print(f"  Avg confidence: {avg_confidence:.3f}")
        print(f"  Std confidence: {std_confidence:.3f}")
        
        # Mock SLAM should maintain stable confidence
        self.assertGreater(avg_confidence, 0.7)
        self.assertLess(std_confidence, 0.2)


class TestSlamConfiguration(unittest.TestCase):
    """Test SLAM configuration loading and validation."""
    
    def test_configuration_loading(self):
        """Test loading SLAM configuration from YAML."""
        # Create temporary config file
        config_data = {
            'Camera.type': 'PinHole',
            'Camera.fx': 525.0,
            'Camera.fy': 525.0,
            'Camera.cx': 319.5,
            'Camera.cy': 239.5,
            'Camera.fps': 30.0,
            'ORBextractor.nFeatures': 1000
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
            
        try:
            # Test configuration can be loaded
            with open(config_path, 'r') as f:
                loaded_config = yaml.safe_load(f)
                
            self.assertEqual(loaded_config['Camera.fx'], 525.0)
            self.assertEqual(loaded_config['ORBextractor.nFeatures'], 1000)
            
        finally:
            Path(config_path).unlink()
            
    def test_nymeria_calibration_files(self):
        """Test that Nymeria calibration files exist and are valid."""
        mono_config = Path('/home/naoto/docker_workspace/MobilePoser/mobileposer/slam_configs/nymeria_mono_base.yaml')
        vi_config = Path('/home/naoto/docker_workspace/MobilePoser/mobileposer/slam_configs/nymeria_vi.yaml')
        
        if mono_config.exists():
            with open(mono_config, 'r') as f:
                mono_data = yaml.safe_load(f)
                
            # Check critical parameters
            self.assertIn('Camera1.fx', mono_data)
            self.assertIn('Camera1.fy', mono_data)
            self.assertIn('Camera1.cx', mono_data)
            self.assertIn('Camera1.cy', mono_data)
            self.assertAlmostEqual(mono_data['Camera1.fx'], 517.306, places=2)
            
        if vi_config.exists():
            with open(vi_config, 'r') as f:
                vi_data = yaml.safe_load(f)
                
            # Check VI-specific parameters
            self.assertIn('IMU.Frequency', vi_data)
            self.assertEqual(vi_data['IMU.Frequency'], 800.0)


class TestMemoryManagement(unittest.TestCase):
    """Test memory cleanup and resource management."""
    
    def test_slam_cleanup(self):
        """Test proper cleanup of SLAM resources."""
        # Create and destroy multiple SLAM instances
        for i in range(5):
            slam = MockSlamInterface()
            slam.initialize()
            
            # Process some frames
            image = np.zeros((480, 640, 3), dtype=np.uint8)
            for j in range(10):
                slam.process_frame(image, j * 0.033)
                
            # Cleanup
            slam.shutdown()
            
            # Verify state is cleaned
            self.assertFalse(slam.is_initialized)
            
    def test_adaptive_slam_cleanup(self):
        """Test cleanup of adaptive SLAM system."""
        adaptive = AdaptiveSlamInterface()
        adaptive.monocular_slam = MockSlamInterface()
        adaptive.visual_inertial_slam = MockSlamInterface()
        
        # Initialize
        adaptive.monocular_slam.initialize()
        adaptive.visual_inertial_slam.initialize()
        adaptive.is_initialized = True
        
        # Process frames
        for i in range(10):
            slam_input = SlamInput(
                rgb_frame=np.zeros((480, 640, 3), dtype=np.uint8),
                timestamp=i*0.033
            )
            adaptive.process_frame(slam_input)
            
        # Cleanup
        adaptive.shutdown()
        
        # Verify cleanup
        self.assertFalse(adaptive.is_initialized)
        self.assertFalse(adaptive.monocular_slam.is_initialized)
        self.assertFalse(adaptive.visual_inertial_slam.is_initialized)


@unittest.skipUnless(HAS_ORBSLAM3, "Requires pyOrbSlam3")
class TestRealOrbSlam3(unittest.TestCase):
    """Test real ORB-SLAM3 integration (only runs if pyOrbSlam3 is available)."""
    
    def setUp(self):
        self.vocab_path = "/home/naoto/docker_workspace/MobilePoser/third_party/pyOrbSlam3/pyOrbSlam3/modules/ORB_SLAM3/Vocabulary/ORBvoc.txt"
        self.config_path = "/home/naoto/docker_workspace/MobilePoser/mobileposer/slam_configs/nymeria_mono_base.yaml"
        
    def test_real_orbslam3_initialization(self):
        """Test initialization of real ORB-SLAM3."""
        slam = RealOrbSlam3Interface()
        
        # Set paths
        slam.vocabulary_path = self.vocab_path
        slam.settings_path = self.config_path
        
        # Initialize
        result = slam.initialize()
        self.assertTrue(result)
        self.assertTrue(slam.is_initialized)
        
        # Shutdown
        slam.shutdown()
        self.assertFalse(slam.is_initialized)
        
    def test_real_slam_frame_processing(self):
        """Test processing frames with real ORB-SLAM3."""
        slam = RealOrbSlam3Interface()
        slam.vocabulary_path = self.vocab_path
        slam.settings_path = self.config_path
        
        if slam.initialize():
            # Create test image with features
            image = self._create_test_image_with_features()
            
            # Process frame
            result = slam.process_frame(image, 0.0)
            
            if result:
                self.assertIn('pose', result)
                self.assertIn('confidence', result)
                self.assertEqual(result['pose'].shape, (4, 4))
                
            slam.shutdown()
            
    def _create_test_image_with_features(self):
        """Create a test image with detectable features."""
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add some features (corners, edges)
        cv2.rectangle(image, (100, 100), (200, 200), (255, 255, 255), 2)
        cv2.circle(image, (320, 240), 50, (255, 255, 255), 2)
        
        # Add some texture
        for i in range(10):
            x = np.random.randint(50, 590)
            y = np.random.randint(50, 430)
            cv2.circle(image, (x, y), 5, (255, 255, 255), -1)
            
        return image


class TestNymeriaIntegration(unittest.TestCase):
    """Test integration with Nymeria dataset (when data is available)."""
    
    def test_nymeria_data_loading(self):
        """Test loading Nymeria RGB video frames."""
        # Check if sample Nymeria data exists
        nymeria_root = Path("/mnt/nas2/naoto/nymeria_dataset/data_video_main_rgb")
        
        if not nymeria_root.exists():
            self.skipTest("Nymeria dataset not available")
            
        # Find a sample sequence
        sequences = list(nymeria_root.glob("*/video_main_rgb.mp4"))
        
        if sequences:
            video_path = sequences[0]
            
            # Try to load a frame
            cap = cv2.VideoCapture(str(video_path))
            
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                print(f"\nNymeria video info:")
                print(f"  Path: {video_path}")
                print(f"  FPS: {fps}")
                print(f"  Frames: {frame_count}")
                
                # Read first frame
                ret, frame = cap.read()
                if ret:
                    self.assertEqual(frame.shape[2], 3)  # RGB channels
                    print(f"  Frame shape: {frame.shape}")
                    
                cap.release()
                
                # Verify FPS is 15 as discovered
                self.assertEqual(fps, 15.0)


class TestConcurrentSlamInstances(unittest.TestCase):
    """Test running multiple SLAM instances concurrently."""
    
    def test_multiple_mock_instances(self):
        """Test multiple mock SLAM instances."""
        instances = []
        
        # Create multiple instances
        for i in range(3):
            slam = MockSlamInterface()
            slam.initialize()
            instances.append(slam)
            
        # Process different frames on each
        for i, slam in enumerate(instances):
            image = np.full((480, 640, 3), i * 50, dtype=np.uint8)
            
            for j in range(5):
                result = slam.process_frame(image, j * 0.033)
                self.assertIsNotNone(result)
                
        # Verify each has independent trajectory
        trajectories = [slam.get_trajectory() for slam in instances]
        
        # Trajectories should be different
        for i in range(len(trajectories)):
            for j in range(i + 1, len(trajectories)):
                # Check that trajectories are not identical
                diff = np.sum(np.abs(trajectories[i] - trajectories[j]))
                self.assertGreater(diff, 0.0)
                
        # Cleanup
        for slam in instances:
            slam.shutdown()


if __name__ == '__main__':
    # Run with verbosity level 2 for detailed output
    unittest.main(verbosity=2)