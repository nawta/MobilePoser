import numpy as np
import torch
import cv2
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import time
from threading import Thread, Lock
from queue import Queue, Empty
from dataclasses import dataclass
import logging

from mobileposer.models.net import MobilePoserNet
from mobileposer.models.slam import SlamInterface, create_slam_interface
from mobileposer.models.fusion import PoseFusionModule, PoseEstimate
from mobileposer.models.real_orbslam3 import RealOrbSlam3Interface
from mobileposer.config import *
import mobileposer.articulate as art


@dataclass
class HeadPoseData:
    """Container for head pose information."""
    position: np.ndarray         # 3D head position
    orientation: np.ndarray      # 3x3 rotation matrix or quaternion
    confidence: float            # Confidence score [0, 1]
    timestamp: float             # Timestamp
    source: str                  # "imu", "vi_slam", "fused"
    scale_factor: float = 1.0    # Scale factor from VI-SLAM
    

class VisualInertialSlamInterface(SlamInterface):
    """
    Visual-Inertial SLAM interface that uses both camera and IMU data
    to provide scaled pose estimates. Extends the base SLAM interface
    to handle IMU integration.
    """
    
    def __init__(self):
        super().__init__()
        self.imu_buffer = []
        self.max_imu_buffer_size = 100
        self.scale_estimate = 1.0
        self.scale_confidence = 0.0
        
    def process_frame_with_imu(self, image: np.ndarray, timestamp: float, 
                              imu_data: Optional[np.ndarray] = None) -> Optional[Dict[str, Any]]:
        """
        Process RGB frame with optional IMU data for Visual-Inertial SLAM.
        
        Args:
            image: RGB image (H, W, 3)
            timestamp: Frame timestamp
            imu_data: IMU measurements [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
            
        Returns:
            Dictionary with pose, scale, and confidence information
        """
        # Add IMU data to buffer
        if imu_data is not None:
            self.imu_buffer.append({
                'data': imu_data,
                'timestamp': timestamp
            })
            
            # Keep buffer size manageable
            if len(self.imu_buffer) > self.max_imu_buffer_size:
                self.imu_buffer = self.imu_buffer[-self.max_imu_buffer_size:]
        
        # Process with base SLAM functionality
        result = self.process_frame(image, timestamp)
        
        if result is not None:
            # Add scale information for VI-SLAM
            result['scale_factor'] = self.scale_estimate
            result['scale_confidence'] = self.scale_confidence
            
            # Update scale estimate using IMU data
            if imu_data is not None:
                self._update_scale_estimate(result, imu_data)
        
        return result
    
    def _update_scale_estimate(self, slam_result: Dict[str, Any], imu_data: np.ndarray):
        """
        Update scale estimate using IMU accelerometer data.
        This is a simplified implementation - in practice, you'd use
        more sophisticated VI-SLAM techniques.
        """
        # Extract accelerometer data (first 3 components)
        acc = imu_data[:3]
        
        # Estimate scale based on gravity alignment
        gravity_magnitude = np.linalg.norm(acc)
        expected_gravity = 9.81
        
        # Update scale estimate with exponential moving average
        new_scale = expected_gravity / max(gravity_magnitude, 0.1)
        self.scale_estimate = 0.9 * self.scale_estimate + 0.1 * new_scale
        
        # Update confidence based on how close to expected gravity
        gravity_error = abs(gravity_magnitude - expected_gravity) / expected_gravity
        confidence = max(0.0, 1.0 - gravity_error)
        self.scale_confidence = 0.9 * self.scale_confidence + 0.1 * confidence


class MockVisualInertialSlam(VisualInertialSlamInterface):
    """
    Mock implementation of Visual-Inertial SLAM for testing.
    Simulates realistic VI-SLAM behavior with scale estimation.
    """
    
    def __init__(self):
        super().__init__()
        self.true_scale = 1.0
        self.scale_noise = 0.05
        
    def initialize(self, config_path: str = None) -> bool:
        """Initialize mock VI-SLAM."""
        self.is_initialized = True
        self.scale_estimate = 1.0 + np.random.normal(0, 0.1)  # Initial scale estimate
        self.scale_confidence = 0.5
        return True
    
    def process_frame_with_imu(self, image: np.ndarray, timestamp: float, 
                              imu_data: Optional[np.ndarray] = None) -> Optional[Dict[str, Any]]:
        """Process frame with mock VI-SLAM."""
        if not self.is_initialized:
            return None
        
        # Generate synthetic camera motion with proper scale
        t = timestamp
        radius = 2.0 * self.scale_estimate  # Scale affects trajectory size
        height = 1.5 * self.scale_estimate
        
        # Circular motion in XZ plane
        x = radius * np.cos(t * 0.1) + np.random.normal(0, 0.02 * self.scale_estimate)
        z = radius * np.sin(t * 0.1) + np.random.normal(0, 0.02 * self.scale_estimate)  
        y = height + np.random.normal(0, 0.01 * self.scale_estimate)
        
        # Create transformation matrix
        pose = np.eye(4)
        pose[0, 3] = x
        pose[1, 3] = y
        pose[2, 3] = z
        
        # Add rotation
        angle = np.arctan2(z, x) + np.pi
        pose[0, 0] = np.cos(angle)
        pose[0, 2] = np.sin(angle)
        pose[2, 0] = -np.sin(angle)
        pose[2, 2] = np.cos(angle)
        
        # Update scale estimate using IMU if available
        if imu_data is not None:
            self._update_scale_estimate_mock(imu_data)
        
        # Mock confidence (higher when scale is well estimated)
        confidence = 0.7 + 0.2 * self.scale_confidence + 0.1 * np.random.random()
        
        return {
            'pose': pose,
            'confidence': confidence,
            'scale_factor': self.scale_estimate,
            'scale_confidence': self.scale_confidence,
            'timestamp': timestamp
        }
    
    def _update_scale_estimate_mock(self, imu_data: np.ndarray):
        """Mock scale estimation update."""
        # Simulate scale convergence over time
        scale_error = self.true_scale - self.scale_estimate
        correction = 0.01 * scale_error + np.random.normal(0, self.scale_noise * 0.01)
        self.scale_estimate += correction
        
        # Update confidence (higher when closer to true scale)
        error_ratio = abs(scale_error) / self.true_scale
        self.scale_confidence = max(0.1, 1.0 - error_ratio)
    
    def reset(self):
        """Reset mock VI-SLAM state."""
        self.scale_estimate = 1.0 + np.random.normal(0, 0.1)
        self.scale_confidence = 0.5
        
    def shutdown(self):
        """Shutdown mock VI-SLAM."""
        self.is_initialized = False


class HeadPoseEnsemble:
    """
    Ensemble system specifically for head pose estimation.
    Combines MobilePoser head IMU data with Visual-Inertial SLAM
    to improve translation accuracy while maintaining orientation quality.
    """
    
    def __init__(self,
                 mobileposer_weights: str,
                 slam_type: str = "mock_vi",
                 slam_config: str = None,
                 head_imu_index: int = 4,  # Head IMU sensor index in Nymeria
                 fusion_method: str = "weighted_average"):
        """
        Initialize head pose ensemble.
        
        Args:
            mobileposer_weights: Path to MobilePoser weights
            slam_type: Type of SLAM ("mock_vi", "orb_slam3_vi")
            slam_config: SLAM configuration file
            head_imu_index: Index of head IMU sensor
            fusion_method: Pose fusion strategy
        """
        self.head_imu_index = head_imu_index
        
        # Initialize MobilePoser
        self.mobileposer = MobilePoserNet.load_from_checkpoint(mobileposer_weights)
        self.mobileposer.eval()
        
        # Initialize VI-SLAM
        if slam_type == "mock_vi":
            self.vi_slam = MockVisualInertialSlam()
        elif slam_type == "orb_slam3_vi" or slam_type == "real_vi":
            # Initialize real ORB-SLAM3 in Visual-Inertial mode
            self.vi_slam = RealOrbSlam3Interface(
                mode="visual_inertial",
                settings_file=slam_config,
                enable_viewer=False
            )
        elif slam_type == "orb_slam3_mono" or slam_type == "real_mono":
            # Initialize real ORB-SLAM3 in Monocular mode
            self.vi_slam = RealOrbSlam3Interface(
                mode="monocular",
                settings_file=slam_config,
                enable_viewer=False
            )
        else:
            # Raise error for unknown SLAM types
            raise ValueError(f"Unknown SLAM type '{slam_type}'. Valid options: 'mock_vi', 'orb_slam3_vi', 'real_vi', 'orb_slam3_mono', 'real_mono'")
            
        if not self.vi_slam.initialize(slam_config):
            raise RuntimeError("Failed to initialize VI-SLAM")
        
        # Initialize fusion module specialized for head pose
        self.head_fusion = HeadPoseFusionModule(fusion_method=fusion_method)
        
        # State tracking
        self.last_head_pose_imu = None
        self.last_head_pose_slam = None
        self.processing_stats = {
            'frames_processed': 0,
            'fusion_success_rate': 0.0,
            'average_confidence': 0.0
        }
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def extract_head_imu_data(self, full_imu_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract head IMU data from full IMU sensor array.
        
        Args:
            full_imu_data: Full IMU data from all sensors
            
        Returns:
            Tuple of (head_acceleration, head_angular_velocity)
        """
        # Extract head IMU data (assuming 6 sensors with 12 values each)
        # Format: [acc_x, acc_y, acc_z, rot_00, rot_01, rot_02, rot_10, rot_11, rot_12, rot_20, rot_21, rot_22]
        
        if full_imu_data.shape[0] == 72:  # 6 sensors * 12 values
            head_data = full_imu_data[self.head_imu_index * 12:(self.head_imu_index + 1) * 12]
        elif full_imu_data.shape[0] == 60:  # Alternative format
            head_data = full_imu_data[self.head_imu_index * 10:(self.head_imu_index + 1) * 10]
        else:
            # Fallback: use first sensor data
            head_data = full_imu_data[:12] if len(full_imu_data) >= 12 else full_imu_data
        
        # Extract acceleration (first 3 values)
        head_acceleration = head_data[:3]
        
        # Extract rotation matrix and convert to angular velocity (simplified)
        if len(head_data) >= 12:
            rotation_matrix = head_data[3:12].reshape(3, 3)
            # Simplified angular velocity estimation (in practice, use gyroscope data)
            head_angular_velocity = np.array([0.1, 0.1, 0.1])  # Placeholder
        else:
            head_angular_velocity = np.array([0.0, 0.0, 0.0])
        
        return head_acceleration, head_angular_velocity
    
    def process_frame(self, rgb_frame: np.ndarray, full_imu_data: np.ndarray, 
                     timestamp: float) -> Optional[HeadPoseData]:
        """
        Process a single frame with both RGB and IMU data.
        
        Args:
            rgb_frame: RGB image (H, W, 3)
            full_imu_data: Full IMU sensor data
            timestamp: Frame timestamp
            
        Returns:
            Fused head pose estimate or None
        """
        # Extract head-specific IMU data
        head_acc, head_gyro = self.extract_head_imu_data(full_imu_data)
        head_imu_6dof = np.concatenate([head_acc, head_gyro])
        
        # Get head pose from MobilePoser
        head_pose_imu = self._estimate_head_pose_imu(full_imu_data, timestamp)
        
        # Get head pose from VI-SLAM
        head_pose_slam = self._estimate_head_pose_slam(rgb_frame, head_imu_6dof, timestamp)
        
        # Fuse poses
        fused_head_pose = self.head_fusion.fuse_head_poses(head_pose_imu, head_pose_slam)
        
        # Update statistics
        self._update_stats(head_pose_imu, head_pose_slam, fused_head_pose)
        
        return fused_head_pose
    
    def _estimate_head_pose_imu(self, full_imu_data: np.ndarray, 
                               timestamp: float) -> Optional[HeadPoseData]:
        """Estimate head pose using MobilePoser."""
        try:
            # Forward through MobilePoser
            imu_tensor = torch.from_numpy(full_imu_data).float()
            
            with torch.no_grad():
                pose, joints, translation, contact = self.mobileposer.forward_online(imu_tensor)
            
            # Extract head joint information (joint index 15 is typically head in SMPL)
            if isinstance(joints, torch.Tensor):
                joints_np = joints.cpu().numpy()
            else:
                joints_np = joints
                
            if isinstance(translation, torch.Tensor):
                translation_np = translation.cpu().numpy()
            else:
                translation_np = translation
            
            # Get head position (joint 15) relative to root
            if joints_np.shape[0] >= 16:  # Ensure we have enough joints
                head_position_rel = joints_np[15]  # Head joint
                head_position_global = translation_np + head_position_rel
            else:
                head_position_global = translation_np
            
            # Get head orientation from pose
            if isinstance(pose, torch.Tensor):
                pose_np = pose.cpu().numpy()
            else:
                pose_np = pose
                
            # Extract head rotation (joint 15)
            if pose_np.shape == (24, 9):
                head_rotation = pose_np[15].reshape(3, 3)
            elif pose_np.shape == (24, 3, 3):
                head_rotation = pose_np[15]
            else:
                head_rotation = np.eye(3)
            
            # Estimate confidence based on contact probabilities
            confidence = 0.8  # Default confidence for IMU
            if contact is not None:
                if isinstance(contact, torch.Tensor):
                    contact_np = contact.cpu().numpy()
                    confidence = 0.6 + 0.4 * float(np.mean(contact_np))
                    
            self.last_head_pose_imu = HeadPoseData(
                position=head_position_global,
                orientation=head_rotation,
                confidence=confidence,
                timestamp=timestamp,
                source="imu"
            )
            
            return self.last_head_pose_imu
            
        except Exception as e:
            self.logger.error(f"Error estimating head pose from IMU: {e}")
            return None
    
    def _estimate_head_pose_slam(self, rgb_frame: np.ndarray, head_imu_data: np.ndarray,
                                timestamp: float) -> Optional[HeadPoseData]:
        """Estimate head pose using Visual-Inertial SLAM."""
        try:
            # Process with VI-SLAM
            slam_result = self.vi_slam.process_frame_with_imu(rgb_frame, timestamp, head_imu_data)
            
            if slam_result is None:
                return None
            
            # Extract camera pose (which represents head pose for head-mounted camera)
            camera_pose = slam_result['pose']  # 4x4 transformation matrix
            confidence = slam_result['confidence']
            scale_factor = slam_result.get('scale_factor', 1.0)
            
            # Extract position and orientation
            head_position = camera_pose[:3, 3] * scale_factor  # Apply scale correction
            head_orientation = camera_pose[:3, :3]
            
            self.last_head_pose_slam = HeadPoseData(
                position=head_position,
                orientation=head_orientation,
                confidence=confidence,
                timestamp=timestamp,
                source="vi_slam",
                scale_factor=scale_factor
            )
            
            return self.last_head_pose_slam
            
        except Exception as e:
            self.logger.error(f"Error estimating head pose from VI-SLAM: {e}")
            return None
    
    def _update_stats(self, imu_pose: Optional[HeadPoseData], 
                     slam_pose: Optional[HeadPoseData],
                     fused_pose: Optional[HeadPoseData]):
        """Update processing statistics."""
        self.processing_stats['frames_processed'] += 1
        
        if fused_pose is not None:
            # Update fusion success rate
            success = 1.0
            self.processing_stats['fusion_success_rate'] = (
                0.95 * self.processing_stats['fusion_success_rate'] + 0.05 * success
            )
            
            # Update average confidence
            self.processing_stats['average_confidence'] = (
                0.95 * self.processing_stats['average_confidence'] + 0.05 * fused_pose.confidence
            )
        
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return self.processing_stats.copy()
    
    def reset(self):
        """Reset ensemble state."""
        self.mobileposer.reset()
        self.vi_slam.reset()
        self.head_fusion.reset()
        self.last_head_pose_imu = None
        self.last_head_pose_slam = None


class HeadPoseFusionModule:
    """
    Specialized fusion module for head pose estimation.
    Optimized for combining IMU orientation accuracy with SLAM translation accuracy.
    """
    
    def __init__(self, fusion_method: str = "weighted_average"):
        self.fusion_method = fusion_method
        # IMU is typically better for orientation, SLAM better for translation
        self.imu_orientation_weight = 0.7
        self.slam_translation_weight = 0.8
        
    def fuse_head_poses(self, imu_pose: Optional[HeadPoseData],
                       slam_pose: Optional[HeadPoseData]) -> Optional[HeadPoseData]:
        """
        Fuse head poses with bias towards IMU orientation and SLAM translation.
        """
        if imu_pose is None and slam_pose is None:
            return None
        elif imu_pose is None:
            return slam_pose
        elif slam_pose is None:
            return imu_pose
        
        if self.fusion_method == "weighted_average":
            return self._weighted_head_fusion(imu_pose, slam_pose)
        else:
            # For other methods, use the general fusion approach
            return self._confidence_based_head_fusion(imu_pose, slam_pose)
    
    def _weighted_head_fusion(self, imu_pose: HeadPoseData, 
                             slam_pose: HeadPoseData) -> HeadPoseData:
        """
        Weighted fusion biased towards IMU orientation and SLAM translation.
        """
        # Translation: favor SLAM (better scale and drift properties)
        translation_weight_slam = self.slam_translation_weight * slam_pose.confidence
        translation_weight_imu = (1 - self.slam_translation_weight) * imu_pose.confidence
        total_trans_weight = translation_weight_slam + translation_weight_imu
        
        if total_trans_weight > 0:
            fused_position = (
                (translation_weight_slam * slam_pose.position +
                 translation_weight_imu * imu_pose.position) / total_trans_weight
            )
        else:
            fused_position = imu_pose.position
        
        # Orientation: favor IMU (better short-term accuracy, less drift)
        orientation_weight_imu = self.imu_orientation_weight * imu_pose.confidence
        orientation_weight_slam = (1 - self.imu_orientation_weight) * slam_pose.confidence
        total_orient_weight = orientation_weight_imu + orientation_weight_slam
        
        if total_orient_weight > 0:
            # Use SLERP for rotation interpolation
            slerp_t = orientation_weight_slam / total_orient_weight
            fused_orientation = self._slerp_rotation_matrices(
                imu_pose.orientation, slam_pose.orientation, slerp_t
            )
        else:
            fused_orientation = imu_pose.orientation
        
        # Combined confidence
        fused_confidence = (imu_pose.confidence + slam_pose.confidence) / 2
        
        # Use average timestamp  
        fused_timestamp = (imu_pose.timestamp + slam_pose.timestamp) / 2
        
        return HeadPoseData(
            position=fused_position,
            orientation=fused_orientation,
            confidence=fused_confidence,
            timestamp=fused_timestamp,
            source="fused",
            scale_factor=slam_pose.scale_factor
        )
    
    def _confidence_based_head_fusion(self, imu_pose: HeadPoseData,
                                     slam_pose: HeadPoseData) -> HeadPoseData:
        """Simple confidence-based selection."""
        if slam_pose.confidence > imu_pose.confidence:
            return slam_pose
        else:
            return imu_pose
    
    def _slerp_rotation_matrices(self, R1: np.ndarray, R2: np.ndarray, t: float) -> np.ndarray:
        """Spherical linear interpolation between rotation matrices."""
        # Convert to quaternions
        q1 = self._rotation_matrix_to_quaternion(R1)
        q2 = self._rotation_matrix_to_quaternion(R2)
        
        # SLERP
        q_interp = self._slerp_quaternions(q1, q2, t)
        
        # Convert back to rotation matrix
        return self._quaternion_to_rotation_matrix(q_interp)
    
    def _rotation_matrix_to_quaternion(self, R: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to quaternion [w, x, y, z]."""
        trace = np.trace(R)
        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2
            w = 0.25 * s
            x = (R[2, 1] - R[1, 2]) / s
            y = (R[0, 2] - R[2, 0]) / s
            z = (R[1, 0] - R[0, 1]) / s
        else:
            if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
                w = (R[2, 1] - R[1, 2]) / s
                x = 0.25 * s
                y = (R[0, 1] + R[1, 0]) / s
                z = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:
                s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
                w = (R[0, 2] - R[2, 0]) / s
                x = (R[0, 1] + R[1, 0]) / s
                y = 0.25 * s
                z = (R[1, 2] + R[2, 1]) / s
            else:
                s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
                w = (R[1, 0] - R[0, 1]) / s
                x = (R[0, 2] + R[2, 0]) / s
                y = (R[1, 2] + R[2, 1]) / s
                z = 0.25 * s
        return np.array([w, x, y, z])
    
    def _quaternion_to_rotation_matrix(self, q: np.ndarray) -> np.ndarray:
        """Convert quaternion [w, x, y, z] to rotation matrix."""
        w, x, y, z = q
        return np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
            [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
        ])
    
    def _slerp_quaternions(self, q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
        """Spherical linear interpolation between quaternions."""
        dot = np.dot(q1, q2)
        
        if dot < 0.0:
            q2 = -q2
            dot = -dot
        
        if dot > 0.9995:
            result = q1 + t * (q2 - q1)
            return result / np.linalg.norm(result)
        
        theta_0 = np.arccos(abs(dot))
        sin_theta_0 = np.sin(theta_0)
        theta = theta_0 * t
        sin_theta = np.sin(theta)
        
        s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0
        
        return s0 * q1 + s1 * q2
    
    def reset(self):
        """Reset fusion module state."""
        pass