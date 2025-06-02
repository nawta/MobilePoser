import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass
import time

@dataclass
class PoseEstimate:
    """Container for pose estimates from different sources."""
    translation: np.ndarray  # 3D translation vector
    rotation: np.ndarray     # 3x3 rotation matrix or quaternion
    confidence: float        # Confidence score [0, 1]
    timestamp: float         # Timestamp in seconds
    source: str             # Source identifier ("imu", "visual", "fused")
    
    def to_transformation_matrix(self) -> np.ndarray:
        """Convert to 4x4 transformation matrix."""
        T = np.eye(4)
        
        if self.rotation.shape == (3, 3):
            T[:3, :3] = self.rotation
        elif self.rotation.shape == (4,):
            # Convert quaternion to rotation matrix
            q = self.rotation
            T[:3, :3] = self._quaternion_to_rotation_matrix(q)
        else:
            raise ValueError(f"Unsupported rotation format: {self.rotation.shape}")
            
        T[:3, 3] = self.translation
        return T
    
    @staticmethod
    def _quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
        """Convert quaternion [w, x, y, z] to rotation matrix."""
        w, x, y, z = q
        return np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
            [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
        ])


class PoseFusionModule:
    """
    Ensemble fusion module for combining IMU-based and visual SLAM pose estimates.
    Implements various fusion strategies including weighted averaging, Kalman filtering,
    and confidence-based selection.
    """
    
    def __init__(self, fusion_method: str = "weighted_average", 
                 confidence_threshold: float = 0.5,
                 temporal_window: float = 1.0):
        """
        Initialize pose fusion module.
        
        Args:
            fusion_method: Fusion strategy ("weighted_average", "kalman", "confidence_based")
            confidence_threshold: Minimum confidence to use estimate
            temporal_window: Time window for temporal consistency (seconds)
        """
        self.fusion_method = fusion_method
        self.confidence_threshold = confidence_threshold
        self.temporal_window = temporal_window
        
        # History for temporal consistency
        self.pose_history: List[PoseEstimate] = []
        self.max_history_size = 100
        
        # Kalman filter state (if using Kalman fusion)
        self.kalman_state = None
        self.kalman_covariance = None
        
        # Weights for different modalities
        self.imu_weight = 0.6    # IMU tends to be more reliable for orientation
        self.visual_weight = 0.4  # Visual tends to be better for translation
        
    def fuse_poses(self, imu_pose: Optional[PoseEstimate], 
                   visual_pose: Optional[PoseEstimate]) -> Optional[PoseEstimate]:
        """
        Fuse IMU and visual pose estimates.
        
        Args:
            imu_pose: Pose estimate from IMU-based system
            visual_pose: Pose estimate from visual SLAM
            
        Returns:
            Fused pose estimate or None if fusion fails
        """
        # Handle cases where one or both estimates are missing
        if imu_pose is None and visual_pose is None:
            return None
        elif imu_pose is None:
            return visual_pose if visual_pose.confidence > self.confidence_threshold else None
        elif visual_pose is None:
            return imu_pose if imu_pose.confidence > self.confidence_threshold else None
            
        # Apply confidence thresholding
        use_imu = imu_pose.confidence > self.confidence_threshold
        use_visual = visual_pose.confidence > self.confidence_threshold
        
        if not use_imu and not use_visual:
            return None
        elif not use_imu:
            return visual_pose
        elif not use_visual:
            return imu_pose
            
        # Both estimates are valid, apply fusion strategy
        if self.fusion_method == "weighted_average":
            return self._weighted_average_fusion(imu_pose, visual_pose)
        elif self.fusion_method == "kalman":
            return self._kalman_fusion(imu_pose, visual_pose)
        elif self.fusion_method == "confidence_based":
            return self._confidence_based_fusion(imu_pose, visual_pose)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
    
    def _weighted_average_fusion(self, imu_pose: PoseEstimate, 
                                visual_pose: PoseEstimate) -> PoseEstimate:
        """
        Weighted average fusion based on confidence and modality strengths.
        """
        # Normalize confidences
        total_conf = imu_pose.confidence + visual_pose.confidence
        imu_weight = (imu_pose.confidence / total_conf) * self.imu_weight
        visual_weight = (visual_pose.confidence / total_conf) * self.visual_weight
        
        # Normalize weights
        total_weight = imu_weight + visual_weight
        imu_weight /= total_weight
        visual_weight /= total_weight
        
        # Fuse translation (simple weighted average)
        fused_translation = (imu_weight * imu_pose.translation + 
                           visual_weight * visual_pose.translation)
        
        # Fuse rotation (using SLERP for rotation matrices)
        fused_rotation = self._slerp_rotation_matrices(
            imu_pose.rotation, visual_pose.rotation, visual_weight
        )
        
        # Combined confidence
        fused_confidence = (imu_weight * imu_pose.confidence + 
                          visual_weight * visual_pose.confidence)
        
        # Use average timestamp
        fused_timestamp = (imu_pose.timestamp + visual_pose.timestamp) / 2
        
        fused_pose = PoseEstimate(
            translation=fused_translation,
            rotation=fused_rotation,
            confidence=fused_confidence,
            timestamp=fused_timestamp,
            source="fused"
        )
        
        self._update_history(fused_pose)
        return fused_pose
    
    def _confidence_based_fusion(self, imu_pose: PoseEstimate, 
                               visual_pose: PoseEstimate) -> PoseEstimate:
        """
        Select pose based on confidence scores with smooth transitions.
        """
        # Determine which estimate to trust more
        conf_diff = abs(imu_pose.confidence - visual_pose.confidence)
        
        if conf_diff < 0.1:  # Similar confidence, use weighted average
            return self._weighted_average_fusion(imu_pose, visual_pose)
        elif imu_pose.confidence > visual_pose.confidence:
            # Use IMU translation but consider visual rotation if very confident
            if visual_pose.confidence > 0.8:
                # High visual confidence, blend rotations
                alpha = 0.3  # Slight bias towards visual rotation
                rotation = self._slerp_rotation_matrices(
                    imu_pose.rotation, visual_pose.rotation, alpha
                )
            else:
                rotation = imu_pose.rotation
                
            return PoseEstimate(
                translation=imu_pose.translation,
                rotation=rotation,
                confidence=imu_pose.confidence,
                timestamp=imu_pose.timestamp,
                source="fused"
            )
        else:
            # Use visual estimate but consider IMU for orientation refinement
            if imu_pose.confidence > 0.8:
                # High IMU confidence, blend rotations
                alpha = 0.7  # Bias towards IMU rotation
                rotation = self._slerp_rotation_matrices(
                    imu_pose.rotation, visual_pose.rotation, alpha
                )
            else:
                rotation = visual_pose.rotation
                
            return PoseEstimate(
                translation=visual_pose.translation,
                rotation=rotation,
                confidence=visual_pose.confidence,
                timestamp=visual_pose.timestamp,
                source="fused"
            )
    
    def _kalman_fusion(self, imu_pose: PoseEstimate, 
                      visual_pose: PoseEstimate) -> PoseEstimate:
        """
        Kalman filter-based fusion (simplified implementation).
        """
        # This is a simplified Kalman filter implementation
        # In practice, you would use a full 6DOF or 7DOF state representation
        
        if self.kalman_state is None:
            # Initialize state with first available estimate
            self.kalman_state = np.concatenate([imu_pose.translation, 
                                              self._rotation_matrix_to_euler(imu_pose.rotation)])
            self.kalman_covariance = np.eye(6) * 0.1
        
        # Prediction step (assuming constant velocity model)
        # In practice, you would use IMU data for prediction
        
        # Update step with measurements
        imu_measurement = np.concatenate([imu_pose.translation,
                                        self._rotation_matrix_to_euler(imu_pose.rotation)])
        visual_measurement = np.concatenate([visual_pose.translation,
                                           self._rotation_matrix_to_euler(visual_pose.rotation)])
        
        # Measurement noise based on confidence
        imu_noise = (1.0 - imu_pose.confidence) * 0.1
        visual_noise = (1.0 - visual_pose.confidence) * 0.1
        
        # Simple weighted update (simplified Kalman update)
        imu_weight = 1.0 / (imu_noise + 1e-6)
        visual_weight = 1.0 / (visual_noise + 1e-6)
        total_weight = imu_weight + visual_weight
        
        updated_state = (imu_weight * imu_measurement + visual_weight * visual_measurement) / total_weight
        self.kalman_state = updated_state
        
        # Extract fused pose
        fused_translation = updated_state[:3]
        fused_rotation = self._euler_to_rotation_matrix(updated_state[3:])
        fused_confidence = min(imu_pose.confidence, visual_pose.confidence) + 0.1
        
        return PoseEstimate(
            translation=fused_translation,
            rotation=fused_rotation,
            confidence=min(fused_confidence, 1.0),
            timestamp=(imu_pose.timestamp + visual_pose.timestamp) / 2,
            source="fused"
        )
    
    def _slerp_rotation_matrices(self, R1: np.ndarray, R2: np.ndarray, t: float) -> np.ndarray:
        """
        Spherical linear interpolation between rotation matrices.
        """
        # Convert to quaternions, interpolate, then back to rotation matrix
        q1 = self._rotation_matrix_to_quaternion(R1)
        q2 = self._rotation_matrix_to_quaternion(R2)
        q_interp = self._slerp_quaternions(q1, q2, t)
        return PoseEstimate._quaternion_to_rotation_matrix(q_interp)
    
    def _slerp_quaternions(self, q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
        """Spherical linear interpolation between quaternions."""
        dot = np.dot(q1, q2)
        
        # If dot product is negative, negate one quaternion to take shorter path
        if dot < 0.0:
            q2 = -q2
            dot = -dot
        
        # If quaternions are very close, use linear interpolation
        if dot > 0.9995:
            result = q1 + t * (q2 - q1)
            return result / np.linalg.norm(result)
        
        # Calculate angle between quaternions
        theta_0 = np.arccos(abs(dot))
        sin_theta_0 = np.sin(theta_0)
        theta = theta_0 * t
        sin_theta = np.sin(theta)
        
        s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0
        
        return s0 * q1 + s1 * q2
    
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
    
    def _rotation_matrix_to_euler(self, R: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to Euler angles (ZYX convention)."""
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6
        
        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0
        
        return np.array([x, y, z])
    
    def _euler_to_rotation_matrix(self, euler: np.ndarray) -> np.ndarray:
        """Convert Euler angles to rotation matrix (ZYX convention)."""
        x, y, z = euler
        
        # Rotation matrices for each axis
        Rx = np.array([[1, 0, 0],
                      [0, np.cos(x), -np.sin(x)],
                      [0, np.sin(x), np.cos(x)]])
        
        Ry = np.array([[np.cos(y), 0, np.sin(y)],
                      [0, 1, 0],
                      [-np.sin(y), 0, np.cos(y)]])
        
        Rz = np.array([[np.cos(z), -np.sin(z), 0],
                      [np.sin(z), np.cos(z), 0],
                      [0, 0, 1]])
        
        return Rz @ Ry @ Rx
    
    def _update_history(self, pose: PoseEstimate):
        """Update pose history for temporal consistency."""
        self.pose_history.append(pose)
        
        # Remove old poses outside temporal window
        current_time = pose.timestamp
        self.pose_history = [p for p in self.pose_history 
                           if current_time - p.timestamp <= self.temporal_window]
        
        # Limit history size
        if len(self.pose_history) > self.max_history_size:
            self.pose_history = self.pose_history[-self.max_history_size:]
    
    def get_temporal_consistency_score(self) -> float:
        """
        Calculate temporal consistency score based on pose history.
        Higher scores indicate more consistent motion.
        """
        if len(self.pose_history) < 3:
            return 0.5  # Neutral score for insufficient data
        
        # Calculate pose differences over time
        differences = []
        for i in range(1, len(self.pose_history)):
            prev_pose = self.pose_history[i-1]
            curr_pose = self.pose_history[i]
            
            # Translation difference
            trans_diff = np.linalg.norm(curr_pose.translation - prev_pose.translation)
            differences.append(trans_diff)
        
        # Lower variance in differences indicates higher consistency
        if len(differences) > 1:
            variance = np.var(differences)
            consistency = 1.0 / (1.0 + variance)  # Higher consistency for lower variance
            return min(consistency, 1.0)
        
        return 0.5
    
    def reset(self):
        """Reset fusion module state."""
        self.pose_history.clear()
        self.kalman_state = None
        self.kalman_covariance = None