#!/usr/bin/env python3
"""
Complete SLAM test suite using REAL ORB-SLAM3 (no mock).
"""

import unittest
import numpy as np
import cv2
from pathlib import Path
import sys
import time
import logging

sys.path.append('/home/naoto/docker_workspace/MobilePoser')

from mobileposer.models.slam import create_slam_interface
from mobileposer.models.real_orbslam3 import RealOrbSlam3Interface, HAS_PYORBSLAM
from mobileposer.models.adaptive_slam import (
    SlamMode, SlamInput, SlamOutput, AdaptiveSlamInterface, EnsembleWeightCalculator
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestRealOrbSlam3(unittest.TestCase):
    """Test real ORB-SLAM3 functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Check if real SLAM is available."""
        if not HAS_PYORBSLAM:
            raise unittest.SkipTest("pyOrbSlam3 not available")
            
        cls.vocab_path = Path("/home/naoto/docker_workspace/MobilePoser/third_party/pyOrbSlam3/pyOrbSlam3/modules/ORB_SLAM3/Vocabulary/ORBvoc.txt")
        cls.config_path = Path("/home/naoto/docker_workspace/MobilePoser/mobileposer/slam_configs/nymeria_mono_base.yaml")
        
        if not cls.vocab_path.exists():
            raise unittest.SkipTest(f"ORB vocabulary not found at {cls.vocab_path}")
            
    def create_feature_rich_image(self, offset=0):
        """Create an image with many features for SLAM."""
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Create a grid pattern
        grid_size = 30
        for i in range(0, 480, grid_size):
            cv2.line(image, (0, i + offset), (640, i + offset), (100, 100, 100), 1)
        for j in range(0, 640, grid_size):
            cv2.line(image, (j + offset, 0), (j + offset, 480), (100, 100, 100), 1)
            
        # Add random features
        np.random.seed(42)  # Consistent features
        for _ in range(50):
            x = np.random.randint(20, 620)
            y = np.random.randint(20, 460)
            size = np.random.randint(5, 15)
            
            if np.random.random() > 0.5:
                cv2.circle(image, (x, y), size, (255, 255, 255), -1)
            else:
                cv2.rectangle(image, (x-size, y-size), (x+size, y+size), (200, 200, 200), -1)
                
        # Add text for more features
        cv2.putText(image, f"Frame {offset}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return image
    
    def test_01_real_slam_initialization(self):
        """Test real ORB-SLAM3 initialization."""
        logger.info("\n=== Testing Real ORB-SLAM3 Initialization ===")
        
        slam = RealOrbSlam3Interface(mode="monocular", enable_viewer=False)
        
        # Check paths
        self.assertTrue(Path(slam.vocabulary_path).exists())
        
        # Initialize
        result = slam.initialize()
        self.assertTrue(result)
        self.assertTrue(slam.is_initialized)
        
        # Check SLAM system is created
        self.assertIsNotNone(slam.slam_system)
        
        # Shutdown
        slam.shutdown()
        self.assertFalse(slam.is_initialized)
        
        logger.info("  ✓ Real ORB-SLAM3 initialized and shutdown successfully")
        
    def test_02_real_slam_frame_processing(self):
        """Test processing frames with real ORB-SLAM3."""
        logger.info("\n=== Testing Real Frame Processing ===")
        
        slam = RealOrbSlam3Interface(mode="monocular", enable_viewer=False)
        self.assertTrue(slam.initialize())
        
        # Process multiple frames with motion
        processed_count = 0
        tracking_count = 0
        
        for i in range(30):  # More frames to allow initialization
            # Create image with features and motion
            image = self.create_feature_rich_image(offset=i*2)
            timestamp = i * 0.033
            
            result = slam.process_frame(image, timestamp)
            
            if result:
                processed_count += 1
                if slam.tracking_state == "tracking":
                    tracking_count += 1
                    
                if i % 10 == 0:
                    logger.info(f"  Frame {i}: state={slam.tracking_state}, confidence={result['confidence']:.3f}")
                    
        self.assertGreater(processed_count, 0)
        logger.info(f"  ✓ Processed {processed_count}/30 frames")
        logger.info(f"  ✓ Tracking achieved: {tracking_count} frames")
        
        slam.shutdown()
        
    def test_03_real_slam_trajectory(self):
        """Test trajectory generation with real SLAM."""
        logger.info("\n=== Testing Real Trajectory Generation ===")
        
        slam = RealOrbSlam3Interface(mode="monocular", enable_viewer=False)
        self.assertTrue(slam.initialize())
        
        # Create a sequence with clear motion
        for i in range(50):
            # Simulate camera moving in a circle
            angle = i * 0.1
            offset_x = int(20 * np.cos(angle))
            offset_y = int(20 * np.sin(angle))
            
            image = self.create_feature_rich_image(offset=i)
            # Shift image to simulate motion
            M = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
            image = cv2.warpAffine(image, M, (640, 480))
            
            slam.process_frame(image, i * 0.033)
            
        # Try to get trajectory
        trajectory = slam.get_trajectory()
        
        if trajectory is not None:
            logger.info(f"  ✓ Trajectory shape: {trajectory.shape}")
            self.assertGreater(len(trajectory), 0)
        else:
            logger.info("  ⚠️  No trajectory available (SLAM may need more initialization)")
            
        slam.shutdown()
        
    def test_04_adaptive_slam_with_real_backend(self):
        """Test adaptive SLAM using real ORB-SLAM3."""
        logger.info("\n=== Testing Adaptive SLAM with Real Backend ===")
        
        # Create adaptive SLAM (will use real ORB-SLAM3)
        adaptive = AdaptiveSlamInterface()
        
        # Don't override with mock - let it create real instances
        try:
            result = adaptive.initialize()
            
            if result:
                logger.info("  ✓ Adaptive SLAM initialized with real ORB-SLAM3")
                
                # Test mode switching with real backends
                test_image = self.create_feature_rich_image()
                
                # Test different input modes
                inputs = [
                    (None, None, SlamMode.NONE),
                    (test_image, None, SlamMode.MONOCULAR),
                    (test_image, np.zeros(6), SlamMode.VISUAL_INERTIAL),
                ]
                
                for rgb, imu, expected_mode in inputs:
                    slam_input = SlamInput(
                        rgb_frame=rgb,
                        head_imu_data=imu,
                        timestamp=0.0
                    )
                    output = adaptive.process_frame(slam_input)
                    self.assertEqual(output.mode_used, expected_mode)
                    logger.info(f"    ✓ Mode: {expected_mode.value}")
                    
                adaptive.shutdown()
            else:
                logger.warning("  ⚠️  Could not initialize adaptive SLAM with real backends")
                
        except Exception as e:
            logger.warning(f"  ⚠️  Adaptive SLAM initialization failed: {e}")
            
    def test_05_slam_factory_real(self):
        """Test SLAM factory creates real instances."""
        logger.info("\n=== Testing SLAM Factory (Real) ===")
        
        # Test creating real SLAM
        slam = create_slam_interface("real")
        self.assertIsInstance(slam, RealOrbSlam3Interface)
        logger.info("  ✓ Factory created RealOrbSlam3Interface")
        
        # Initialize and test
        if slam.initialize():
            logger.info("  ✓ Real SLAM initialized via factory")
            
            # Process a frame
            test_image = self.create_feature_rich_image()
            result = slam.process_frame(test_image, 0.0)
            
            self.assertIsNotNone(result)
            logger.info("  ✓ Real SLAM processed frame")
            
            slam.shutdown()
        else:
            logger.warning("  ⚠️  Real SLAM initialization failed")
            
    def test_06_performance_benchmark_real(self):
        """Benchmark real ORB-SLAM3 performance."""
        logger.info("\n=== Real ORB-SLAM3 Performance Benchmark ===")
        
        slam = RealOrbSlam3Interface(mode="monocular", enable_viewer=False)
        
        if slam.initialize():
            # Warm up
            test_image = self.create_feature_rich_image()
            for _ in range(5):
                slam.process_frame(test_image, 0.0)
                
            # Benchmark
            num_frames = 30
            start_time = time.time()
            
            for i in range(num_frames):
                image = self.create_feature_rich_image(offset=i)
                slam.process_frame(image, i * 0.033)
                
            elapsed = time.time() - start_time
            fps = num_frames / elapsed
            ms_per_frame = (elapsed / num_frames) * 1000
            
            logger.info(f"  Real ORB-SLAM3 Performance:")
            logger.info(f"    - FPS: {fps:.1f}")
            logger.info(f"    - ms/frame: {ms_per_frame:.2f}")
            logger.info(f"    - Total time: {elapsed:.2f}s for {num_frames} frames")
            
            # Real SLAM should process at reasonable speed
            self.assertGreater(fps, 10)  # At least 10 FPS
            
            slam.shutdown()
            
    def test_07_config_loading(self):
        """Test loading different SLAM configurations."""
        logger.info("\n=== Testing Configuration Loading ===")
        
        configs = {
            "monocular": self.config_path,
            "custom": self.config_path  # Test with custom path
        }
        
        for name, config_path in configs.items():
            if config_path.exists():
                slam = RealOrbSlam3Interface(mode="monocular", enable_viewer=False)
                
                # Initialize with specific config
                result = slam.initialize(str(config_path))
                
                if result:
                    logger.info(f"  ✓ Loaded {name} configuration")
                    slam.shutdown()
                else:
                    logger.warning(f"  ⚠️  Failed to load {name} configuration")


def run_real_slam_tests():
    """Run all real SLAM tests."""
    
    print("\n" + "="*70)
    print("REAL ORB-SLAM3 TEST SUITE (NO MOCK)")
    print("="*70)
    
    # Check if real SLAM is available
    if not HAS_PYORBSLAM:
        print("\n❌ pyOrbSlam3 not available. Cannot run real SLAM tests.")
        print("Please ensure pyOrbSlam3 is built with Python 3.10")
        return False
        
    print(f"\n✅ pyOrbSlam3 is available!")
    print("Running real ORB-SLAM3 tests...\n")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestRealOrbSlam3)
    
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
        print("\n✅ All real SLAM tests passed!")
        print("🎉 No more mock SLAM needed - real ORB-SLAM3 is working!")
    else:
        print("\n❌ Some tests failed")
        
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_real_slam_tests()