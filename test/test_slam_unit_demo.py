#!/usr/bin/env python3
"""
Focused SLAM unit test demonstrating key functionality with real ORB-SLAM3.
"""

import unittest
import numpy as np
import cv2
import time
from pathlib import Path

from mobileposer.models.slam import create_slam_interface
from mobileposer.models.adaptive_slam import (
    AdaptiveSlamInterface, SlamInput, SlamMode, EnsembleWeightCalculator
)


class TestSlamKeyFunctionality(unittest.TestCase):
    """Unit tests demonstrating key SLAM functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_image = self._create_test_scene()
        
    def _create_test_scene(self):
        """Create a realistic test scene with good features."""
        # Create base image
        img = np.ones((480, 640, 3), dtype=np.uint8) * 50
        
        # Add grid pattern (like a floor)
        for i in range(0, 480, 40):
            cv2.line(img, (0, i), (640, i), (100, 100, 100), 1)
        for j in range(0, 640, 40):
            cv2.line(img, (j, 0), (j, 480), (100, 100, 100), 1)
            
        # Add some "furniture" (rectangles)
        cv2.rectangle(img, (100, 100), (200, 300), (200, 150, 100), -1)
        cv2.rectangle(img, (400, 200), (550, 350), (150, 200, 100), -1)
        
        # Add text as features
        cv2.putText(img, "SLAM TEST", (250, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
        
        # Add corner markers
        marker_size = 20
        cv2.rectangle(img, (0, 0), (marker_size, marker_size), (255, 255, 255), -1)
        cv2.rectangle(img, (640-marker_size, 0), (640, marker_size), (255, 255, 255), -1)
        cv2.rectangle(img, (0, 480-marker_size), (marker_size, 480), (255, 255, 255), -1)
        cv2.rectangle(img, (640-marker_size, 480-marker_size), (640, 480), (255, 255, 255), -1)
        
        return img
    
    def test_01_slam_initialization_and_state(self):
        """Test SLAM initialization and state management."""
        print("\n🔧 Test 1: SLAM Initialization and State")
        
        # Create real SLAM
        slam = create_slam_interface("real")
        
        # Check initial state
        self.assertFalse(slam.is_initialized)
        self.assertIsNone(slam.last_pose)
        
        # Initialize
        result = slam.initialize()
        self.assertTrue(result, "SLAM should initialize successfully")
        self.assertTrue(slam.is_initialized)
        
        print("✅ SLAM initialized with real ORB-SLAM3")
        
        # Clean up
        slam.shutdown()
        self.assertFalse(slam.is_initialized)
        print("✅ SLAM shutdown cleanly")
        
    def test_02_frame_processing_pipeline(self):
        """Test the complete frame processing pipeline."""
        print("\n📹 Test 2: Frame Processing Pipeline")
        
        slam = create_slam_interface("real")
        slam.initialize()
        
        results = []
        
        # Process sequence of frames with simulated motion
        for i in range(20):
            # Simulate camera motion (pan right)
            offset = i * 10
            shifted_img = np.roll(self.test_image, offset, axis=1)
            
            # Process frame
            timestamp = i * 0.033  # 30 FPS
            result = slam.process_frame(shifted_img, timestamp)
            
            if result:
                results.append({
                    'frame': i,
                    'has_pose': result['pose'] is not None,
                    'confidence': result['confidence'],
                    'state': slam.tracking_state if hasattr(slam, 'tracking_state') else 'unknown'
                })
        
        # Verify results
        self.assertGreater(len(results), 0, "Should process some frames")
        
        # Print summary
        print(f"✅ Processed {len(results)}/{20} frames")
        if results:
            avg_confidence = np.mean([r['confidence'] for r in results])
            print(f"   Average confidence: {avg_confidence:.3f}")
            states = set(r['state'] for r in results)
            print(f"   States observed: {states}")
        
        slam.shutdown()
        
    def test_03_adaptive_slam_mode_switching(self):
        """Test adaptive SLAM mode switching."""
        print("\n🔄 Test 3: Adaptive SLAM Mode Switching")
        
        adaptive = AdaptiveSlamInterface()
        
        # Initialize with real SLAM backends
        if not adaptive.initialize():
            self.skipTest("Could not initialize adaptive SLAM")
            
        # Test mode switching
        test_cases = [
            ("No sensors", None, None, SlamMode.NONE),
            ("RGB only", self.test_image, None, SlamMode.MONOCULAR),
            ("RGB + IMU", self.test_image, np.random.randn(6) * 0.1, SlamMode.VISUAL_INERTIAL),
            ("No sensors again", None, None, SlamMode.NONE),
        ]
        
        mode_history = []
        
        for name, rgb, imu, expected_mode in test_cases:
            slam_input = SlamInput(
                rgb_frame=rgb,
                head_imu_data=imu,
                timestamp=len(mode_history) * 0.033
            )
            
            output = adaptive.process_frame(slam_input)
            mode_history.append(output.mode_used)
            
            self.assertEqual(output.mode_used, expected_mode, 
                           f"{name} should use {expected_mode}")
            print(f"✅ {name} → {output.mode_used.value}")
        
        # Verify mode switches occurred
        unique_modes = set(mode_history)
        self.assertEqual(len(unique_modes), 3, "Should have used all 3 modes")
        print(f"✅ Successfully switched between {len(unique_modes)} modes")
        
        adaptive.shutdown()
        
    def test_04_ensemble_weight_dynamics(self):
        """Test ensemble weight calculation dynamics."""
        print("\n⚖️  Test 4: Ensemble Weight Dynamics")
        
        calc = EnsembleWeightCalculator()
        
        # Simulate different scenarios
        scenarios = [
            ("SLAM Lost", 0.8, 0.0, "lost", 1.0, 0.0),
            ("Both Good", 0.7, 0.85, "tracking", None, None),
            ("SLAM Better", 0.5, 0.9, "tracking", None, None),
            ("IMU Better", 0.9, 0.6, "tracking", None, None),
        ]
        
        for name, imu_conf, slam_conf, slam_state, expected_imu, expected_slam in scenarios:
            slam_output = type('SlamOutput', (), {
                'confidence': slam_conf,
                'tracking_state': slam_state,
                'pose': np.eye(4) if slam_state == "tracking" else None,
                'mode_used': SlamMode.VISUAL_INERTIAL,
                'scale_confidence': 0.8
            })()
            
            imu_weight, slam_weight = calc.calculate_weights(
                imu_conf, slam_output, imu_pose_available=True
            )
            
            # Verify weights sum to 1
            self.assertAlmostEqual(imu_weight + slam_weight, 1.0, places=6)
            
            # Check expected values if provided
            if expected_imu is not None:
                self.assertAlmostEqual(imu_weight, expected_imu, places=2)
            if expected_slam is not None:
                self.assertAlmostEqual(slam_weight, expected_slam, places=2)
                
            print(f"✅ {name}: IMU={imu_weight:.2f}, SLAM={slam_weight:.2f}")
        
    def test_05_slam_performance_metrics(self):
        """Test SLAM performance metrics."""
        print("\n📊 Test 5: SLAM Performance Metrics")
        
        slam = create_slam_interface("real")
        slam.initialize()
        
        # Warm up
        for _ in range(5):
            slam.process_frame(self.test_image, 0.0)
        
        # Measure performance
        num_frames = 20
        start_time = time.time()
        
        for i in range(num_frames):
            # Add slight variation to each frame
            img = self.test_image.copy()
            cv2.putText(img, f"{i}", (300, 300), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            slam.process_frame(img, i * 0.033)
            
        elapsed = time.time() - start_time
        fps = num_frames / elapsed
        
        print(f"✅ Performance Metrics:")
        print(f"   - FPS: {fps:.1f}")
        print(f"   - ms/frame: {(elapsed/num_frames)*1000:.1f}")
        print(f"   - Total time: {elapsed:.2f}s")
        
        # Real SLAM should achieve reasonable FPS
        self.assertGreater(fps, 10, "Should achieve at least 10 FPS")
        
        slam.shutdown()


def run_demo_tests():
    """Run the SLAM functionality demonstration tests."""
    
    print("\n" + "="*60)
    print("🚀 SLAM UNIT TEST DEMONSTRATION")
    print("="*60)
    print("Testing key SLAM functionality with real ORB-SLAM3")
    
    # Create and run test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSlamKeyFunctionality)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "="*60)
    print("📋 SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {result.failures and len(result.failures) or 0}")
    print(f"Errors: {result.errors and len(result.errors) or 0}")
    
    if result.wasSuccessful():
        print("\n✅ All SLAM unit tests passed!")
        print("\n🎯 Key Takeaways:")
        print("1. Real ORB-SLAM3 initializes and processes frames")
        print("2. Adaptive SLAM switches modes based on sensor availability")
        print("3. Ensemble weights adapt to confidence levels")
        print("4. Performance is suitable for real-time operation")
    else:
        print("\n❌ Some tests failed")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_demo_tests()