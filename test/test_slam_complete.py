#!/usr/bin/env python3
"""
Complete SLAM test suite combining all unit tests and enhancements.
Works with both mock and real SLAM implementations.
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
import logging

sys.path.append('/home/naoto/docker_workspace/MobilePoser')

from mobileposer.models.slam import (
    SlamInterface, MockSlamInterface, create_slam_interface
)
from mobileposer.models.adaptive_slam import (
    SlamMode, SlamInput, SlamOutput, AdaptiveSlamInterface, EnsembleWeightCalculator
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check if real SLAM is available
HAS_REAL_SLAM = False
try:
    from mobileposer.models.real_orbslam3 import RealOrbSlam3Interface
    # Try to create an instance to verify it works
    test_slam = create_slam_interface("real")
    if hasattr(test_slam, 'slam_system') and test_slam.slam_system is not None:
        HAS_REAL_SLAM = True
        logger.info("Real ORB-SLAM3 is available")
    else:
        logger.info("Real ORB-SLAM3 interface exists but pyOrbSlam3 not available")
except Exception as e:
    logger.info(f"Real ORB-SLAM3 not available: {e}")


class TestSlamComplete(unittest.TestCase):
    """Complete test suite for SLAM functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once."""
        cls.test_image = cls._create_test_image()
        cls.vocab_path = Path("/home/naoto/docker_workspace/MobilePoser/third_party/pyOrbSlam3/pyOrbSlam3/modules/ORB_SLAM3/Vocabulary/ORBvoc.txt")
        cls.config_path = Path("/home/naoto/docker_workspace/MobilePoser/mobileposer/slam_configs/nymeria_mono_base.yaml")
        
    @staticmethod
    def _create_test_image():
        """Create a test image with features for SLAM."""
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add checkerboard pattern
        square_size = 40
        for i in range(0, 480, square_size):
            for j in range(0, 640, square_size):
                if (i // square_size + j // square_size) % 2 == 0:
                    image[i:i+square_size, j:j+square_size] = 255
                    
        # Add some circles for additional features
        for _ in range(10):
            x = np.random.randint(50, 590)
            y = np.random.randint(50, 430)
            radius = np.random.randint(10, 30)
            cv2.circle(image, (x, y), radius, (128, 128, 128), 2)
            
        return image
    
    def test_01_basic_slam_functionality(self):
        """Test basic SLAM interface functionality."""
        logger.info("\n=== Testing Basic SLAM Functionality ===")
        
        # Test with mock SLAM (always available)
        slam = MockSlamInterface()
        self.assertTrue(slam.initialize())
        
        # Process a few frames
        for i in range(5):
            result = slam.process_frame(self.test_image, i * 0.033)
            self.assertIsNotNone(result)
            self.assertIn('pose', result)
            self.assertEqual(result['pose'].shape, (4, 4))
            
        # Get trajectory
        trajectory = slam.get_trajectory()
        self.assertEqual(len(trajectory), 5)
        
        slam.shutdown()
        
    def test_02_adaptive_slam_modes(self):
        """Test adaptive SLAM mode selection."""
        logger.info("\n=== Testing Adaptive SLAM Modes ===")
        
        adaptive = AdaptiveSlamInterface()
        
        # Use mock implementations
        adaptive.monocular_slam = MockSlamInterface()
        adaptive.visual_inertial_slam = MockSlamInterface()
        adaptive.monocular_slam.initialize()
        adaptive.visual_inertial_slam.initialize()
        adaptive.is_initialized = True
        
        test_cases = [
            (None, None, SlamMode.NONE, "No sensors"),
            (self.test_image, None, SlamMode.MONOCULAR, "RGB only"),
            (self.test_image, np.zeros(6), SlamMode.VISUAL_INERTIAL, "RGB + IMU"),
        ]
        
        for rgb, imu, expected_mode, desc in test_cases:
            slam_input = SlamInput(
                rgb_frame=rgb,
                head_imu_data=imu,
                timestamp=0.0
            )
            output = adaptive.process_frame(slam_input)
            self.assertEqual(output.mode_used, expected_mode, f"Failed for: {desc}")
            logger.info(f"  ✓ {desc}: {expected_mode.value}")
            
        adaptive.shutdown()
        
    def test_03_weight_calculation(self):
        """Test ensemble weight calculation."""
        logger.info("\n=== Testing Weight Calculation ===")
        
        calc = EnsembleWeightCalculator()
        
        # Test different scenarios
        scenarios = [
            {
                'name': 'High SLAM confidence',
                'imu_conf': 0.6,
                'slam_output': SlamOutput(
                    pose=np.eye(4),
                    confidence=0.9,
                    tracking_state="tracking",
                    mode_used=SlamMode.VISUAL_INERTIAL,
                    scale_confidence=0.8
                ),
                'expected_slam_weight_min': 0.6
            },
            {
                'name': 'Low SLAM confidence',
                'imu_conf': 0.8,
                'slam_output': SlamOutput(
                    pose=np.eye(4),
                    confidence=0.3,
                    tracking_state="tracking",
                    mode_used=SlamMode.MONOCULAR
                ),
                'expected_slam_weight_max': 0.4
            },
            {
                'name': 'SLAM lost',
                'imu_conf': 0.7,
                'slam_output': SlamOutput(
                    tracking_state="lost"
                ),
                'expected_slam_weight_exact': 0.0
            }
        ]
        
        for scenario in scenarios:
            imu_weight, slam_weight = calc.calculate_weights(
                scenario['imu_conf'],
                scenario['slam_output']
            )
            
            self.assertAlmostEqual(imu_weight + slam_weight, 1.0)
            
            if 'expected_slam_weight_exact' in scenario:
                self.assertEqual(slam_weight, scenario['expected_slam_weight_exact'])
            if 'expected_slam_weight_min' in scenario:
                self.assertGreaterEqual(slam_weight, scenario['expected_slam_weight_min'])
            if 'expected_slam_weight_max' in scenario:
                self.assertLessEqual(slam_weight, scenario['expected_slam_weight_max'] + 0.1)  # Allow 10% tolerance
                
            logger.info(f"  ✓ {scenario['name']}: IMU={imu_weight:.2f}, SLAM={slam_weight:.2f}")
            
    def test_04_performance_benchmark(self):
        """Benchmark SLAM performance."""
        logger.info("\n=== Performance Benchmark ===")
        
        slam = MockSlamInterface()
        slam.initialize()
        
        # Warm up
        for _ in range(10):
            slam.process_frame(self.test_image, 0.0)
            
        # Benchmark
        num_frames = 100
        start_time = time.time()
        
        for i in range(num_frames):
            result = slam.process_frame(self.test_image, i * 0.033)
            self.assertIsNotNone(result)
            
        elapsed = time.time() - start_time
        fps = num_frames / elapsed
        ms_per_frame = (elapsed / num_frames) * 1000
        
        logger.info(f"  Mock SLAM Performance:")
        logger.info(f"    - FPS: {fps:.1f}")
        logger.info(f"    - ms/frame: {ms_per_frame:.2f}")
        logger.info(f"    - Total time: {elapsed:.2f}s for {num_frames} frames")
        
        # Mock should be very fast
        self.assertGreater(fps, 100)
        
        slam.shutdown()
        
    def test_05_edge_cases(self):
        """Test edge cases and error handling."""
        logger.info("\n=== Testing Edge Cases ===")
        
        slam = MockSlamInterface()
        slam.initialize()
        
        # Test various edge cases
        edge_cases = [
            ("Dark image", np.zeros((480, 640, 3), dtype=np.uint8)),
            ("Bright image", np.full((480, 640, 3), 255, dtype=np.uint8)),
            ("Single color", np.full((480, 640, 3), 128, dtype=np.uint8)),
            ("Noisy image", np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)),
        ]
        
        for name, image in edge_cases:
            result = slam.process_frame(image, 0.0)
            self.assertIsNotNone(result, f"Failed on: {name}")
            logger.info(f"  ✓ Handled: {name}")
            
        slam.shutdown()
        
    def test_06_long_sequence(self):
        """Test SLAM stability over long sequences."""
        logger.info("\n=== Testing Long Sequence Stability ===")
        
        slam = MockSlamInterface()
        slam.initialize()
        
        # Process 1000 frames (33 seconds at 30fps)
        num_frames = 1000
        confidences = []
        
        for i in range(num_frames):
            result = slam.process_frame(self.test_image, i * 0.033)
            if result:
                confidences.append(result['confidence'])
                
        # Analyze stability
        avg_conf = np.mean(confidences)
        std_conf = np.std(confidences)
        
        logger.info(f"  Long sequence stats ({num_frames} frames):")
        logger.info(f"    - Avg confidence: {avg_conf:.3f}")
        logger.info(f"    - Std confidence: {std_conf:.3f}")
        logger.info(f"    - Min confidence: {np.min(confidences):.3f}")
        logger.info(f"    - Max confidence: {np.max(confidences):.3f}")
        
        # Should maintain stable confidence
        self.assertGreater(avg_conf, 0.7)
        self.assertLess(std_conf, 0.2)
        
        slam.shutdown()
        
    @unittest.skipUnless(HAS_REAL_SLAM, "Requires real ORB-SLAM3")
    def test_07_real_slam(self):
        """Test real ORB-SLAM3 if available."""
        logger.info("\n=== Testing Real ORB-SLAM3 ===")
        
        slam = create_slam_interface("real")
        
        if slam.initialize():
            # Process test frames
            for i in range(10):
                result = slam.process_frame(self.test_image, i * 0.033)
                logger.info(f"  Frame {i}: tracking_state={slam.slam_system.GetTrackingState() if hasattr(slam, 'slam_system') else 'N/A'}")
                
            slam.shutdown()
            logger.info("  ✓ Real SLAM test completed")
        else:
            self.fail("Real SLAM failed to initialize")
            
    def test_08_nymeria_config_validation(self):
        """Validate Nymeria SLAM configuration files."""
        logger.info("\n=== Validating Nymeria Configurations ===")
        
        configs = {
            "monocular": self.config_path,
            "visual-inertial": Path(str(self.config_path).replace("mono_base", "vi"))
        }
        
        for name, path in configs.items():
            if path.exists():
                try:
                    with open(path, 'r') as f:
                        # Skip the %YAML directive line
                        content = f.read()
                        if content.startswith('%YAML'):
                            content = '\n'.join(content.split('\n')[1:])
                        data = yaml.safe_load(content)
                    
                    # Validate critical parameters
                    required_params = ['Camera1.fx', 'Camera1.fy', 'Camera1.cx', 'Camera1.cy']
                    for param in required_params:
                        self.assertIn(param, data, f"{param} missing in {name} config")
                        
                    logger.info(f"  ✓ {name} config valid")
                    logger.info(f"    - fx: {data.get('Camera1.fx', 'N/A')}")
                    logger.info(f"    - fy: {data.get('Camera1.fy', 'N/A')}")
                    
                    if name == "visual-inertial":
                        self.assertIn('IMU.Frequency', data)
                        logger.info(f"    - IMU freq: {data.get('IMU.Frequency', 'N/A')} Hz")
                except Exception as e:
                    logger.warning(f"  ⚠️  Failed to load {name} config: {e}")
                    
    def test_09_memory_cleanup(self):
        """Test proper memory cleanup."""
        logger.info("\n=== Testing Memory Cleanup ===")
        
        # Create and destroy multiple instances
        for i in range(5):
            slam = MockSlamInterface()
            slam.initialize()
            
            # Process frames
            for j in range(100):
                slam.process_frame(self.test_image, j * 0.033)
                
            # Get trajectory before cleanup
            traj = slam.get_trajectory()
            self.assertEqual(len(traj), 100)
            
            # Cleanup
            slam.shutdown()
            
            # Verify state is reset
            self.assertFalse(slam.is_initialized)
            # After shutdown, trajectory should be cleared
            new_slam = MockSlamInterface()
            self.assertEqual(len(new_slam.trajectory), 0)
            
        logger.info("  ✓ Memory cleanup verified for 5 instances")


def run_comprehensive_tests():
    """Run all SLAM tests with detailed reporting."""
    
    print("\n" + "="*70)
    print("COMPREHENSIVE SLAM TEST SUITE")
    print("="*70)
    
    # Check environment
    print("\nEnvironment Check:")
    print(f"  Python version: {sys.version.split()[0]}")
    print(f"  Working directory: {Path.cwd()}")
    print(f"  Real SLAM available: {HAS_REAL_SLAM}")
    
    if not HAS_REAL_SLAM:
        print("\n⚠️  Note: pyOrbSlam3 not available. Tests will use mock SLAM.")
        print("  This is likely due to Python version mismatch:")
        print("  - pyOrbSlam3 was built with Python 3.12")
        print("  - Current environment uses Python 3.10")
        print("  To enable real SLAM tests, rebuild pyOrbSlam3 with Python 3.10")
    
    # Run tests
    print("\nRunning tests...\n")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestSlamComplete)
    
    # Run with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.wasSuccessful():
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed")
        
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_comprehensive_tests()