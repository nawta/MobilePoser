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
from mobileposer.models.adaptive_slam import (
    AdaptiveSlamInterface, SlamInput, SlamOutput, SlamMode,
    EnsembleWeightCalculator
)
from mobileposer.models.fusion import PoseEstimate
from mobileposer.config import *
import mobileposer.articulate as art


@dataclass
class AdaptiveEnsembleInput:
    """Input data for adaptive ensemble system."""
    rgb_frame: Optional[np.ndarray] = None      # RGB image (H, W, 3)
    full_imu_data: Optional[np.ndarray] = None  # All IMU sensors data
    head_imu_data: Optional[np.ndarray] = None  # Head IMU only [acc, gyro]
    timestamp: float = 0.0
    frame_id: int = 0


@dataclass
class AdaptiveEnsembleOutput:
    """Output from adaptive ensemble system."""
    head_pose: Optional[PoseEstimate] = None     # Fused head pose
    imu_pose: Optional[PoseEstimate] = None      # IMU-only pose
    slam_output: Optional[SlamOutput] = None     # SLAM output
    ensemble_weights: Tuple[float, float] = (1.0, 0.0)  # (IMU, SLAM) weights
    mode_used: SlamMode = SlamMode.NONE
    processing_time: float = 0.0
    timestamp: float = 0.0


class AdaptiveHeadPoseEnsemble:
    """
    Adaptive head pose ensemble system that:
    1. Automatically selects SLAM mode based on available data
    2. Calculates dynamic ensemble weights
    3. Uses temporal feedback for next frame prediction
    4. Handles missing data gracefully
    """
    
    def __init__(self,
                 mobileposer_weights: str,
                 head_imu_index: int = 4,
                 orb_vocabulary_path: Optional[str] = None,
                 camera_config: Optional[Dict[str, Any]] = None,
                 enable_temporal_feedback: bool = True,
                 sync_tolerance: float = 0.05):
        """
        Initialize adaptive head pose ensemble.
        
        Args:
            mobileposer_weights: Path to MobilePoser weights
            head_imu_index: Index of head IMU sensor
            orb_vocabulary_path: Path to ORB vocabulary
            camera_config: Camera calibration parameters
            enable_temporal_feedback: Enable temporal feedback mechanism
            sync_tolerance: Time synchronization tolerance
        """
        self.head_imu_index = head_imu_index
        self.enable_temporal_feedback = enable_temporal_feedback
        self.sync_tolerance = sync_tolerance
        
        # Initialize MobilePoser
        self.mobileposer = MobilePoserNet.load_from_checkpoint(mobileposer_weights)
        self.mobileposer.eval()
        
        # Initialize adaptive SLAM
        self.adaptive_slam = AdaptiveSlamInterface(
            orb_vocabulary_path=orb_vocabulary_path,
            camera_config=camera_config
        )
        
        # Initialize ensemble weight calculator
        self.weight_calculator = EnsembleWeightCalculator()
        
        # Temporal feedback state
        self.previous_fused_pose = None
        self.pose_history = []
        self.max_history_size = 10
        
        # Performance tracking
        self.processing_stats = {
            'frames_processed': 0,
            'slam_frames': 0,
            'imu_only_frames': 0,
            'ensemble_frames': 0,
            'mode_distribution': {mode.value: 0 for mode in SlamMode},
            'average_weights': {'imu': 0.0, 'slam': 0.0},
            'processing_times': []
        }
        
        # State tracking
        self.frame_count = 0
        self.start_time = None
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def initialize(self) -> bool:
        """Initialize all components."""
        try:
            # Initialize SLAM system
            slam_init = self.adaptive_slam.initialize()
            
            if not slam_init:
                self.logger.warning("SLAM initialization failed, using IMU-only mode")
            
            self.start_time = time.time()
            self.logger.info("Adaptive head pose ensemble initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ensemble: {e}")
            return False
    
    def process_frame(self, ensemble_input: AdaptiveEnsembleInput) -> AdaptiveEnsembleOutput:
        """
        Process a single frame with adaptive ensemble approach.
        
        Args:
            ensemble_input: Input data container
            
        Returns:
            Ensemble output with fused pose and metadata
        """
        start_time = time.time()
        
        # Extract head IMU data if not provided
        if ensemble_input.head_imu_data is None and ensemble_input.full_imu_data is not None:
            ensemble_input.head_imu_data = self._extract_head_imu_data(
                ensemble_input.full_imu_data
            )
        
        # Process with MobilePoser (always available as baseline)
        imu_pose = self._process_with_mobileposer(
            ensemble_input.full_imu_data,
            ensemble_input.timestamp
        )
        
        # Process with adaptive SLAM (if RGB available)
        slam_output = None
        if ensemble_input.rgb_frame is not None:
            slam_input = SlamInput(
                rgb_frame=ensemble_input.rgb_frame,
                head_imu_data=ensemble_input.head_imu_data,
                timestamp=ensemble_input.timestamp,
                frame_id=ensemble_input.frame_id
            )
            slam_output = self.adaptive_slam.process_frame(slam_input)
        
        # Calculate ensemble weights
        ensemble_weights = self._calculate_ensemble_weights(imu_pose, slam_output)
        
        # Fuse poses with temporal feedback
        fused_pose = self._fuse_poses_with_feedback(
            imu_pose, slam_output, ensemble_weights, ensemble_input.timestamp
        )
        
        # Update statistics
        processing_time = time.time() - start_time
        self._update_stats(slam_output, ensemble_weights, processing_time)
        
        # Create output
        output = AdaptiveEnsembleOutput(
            head_pose=fused_pose,
            imu_pose=imu_pose,
            slam_output=slam_output,
            ensemble_weights=ensemble_weights,
            mode_used=slam_output.mode_used if slam_output else SlamMode.NONE,
            processing_time=processing_time,
            timestamp=ensemble_input.timestamp
        )
        
        self.frame_count += 1
        return output
    
    def _extract_head_imu_data(self, full_imu_data: np.ndarray) -> np.ndarray:
        """Extract head-specific IMU data from full sensor array."""
        try:
            if full_imu_data.shape[0] == 72:  # 6 sensors * 12 values
                head_data = full_imu_data[self.head_imu_index * 12:(self.head_imu_index + 1) * 12]
            elif full_imu_data.shape[0] == 60:  # Alternative format
                head_data = full_imu_data[self.head_imu_index * 10:(self.head_imu_index + 1) * 10]
            else:
                # Fallback: use first 12 values
                head_data = full_imu_data[:12] if len(full_imu_data) >= 12 else full_imu_data
            
            # Extract acceleration and angular velocity
            head_acceleration = head_data[:3]
            
            # For head angular velocity, we need gyroscope data
            # If not available, estimate from rotation matrix changes
            if len(head_data) >= 6:
                head_angular_velocity = head_data[3:6]  # Assuming gyro data available
            else:
                head_angular_velocity = np.array([0.0, 0.0, 0.0])  # Placeholder
            
            return np.concatenate([head_acceleration, head_angular_velocity])
            
        except Exception as e:
            self.logger.error(f"Error extracting head IMU data: {e}")
            return np.zeros(6)  # Return zero IMU data as fallback
    
    def _process_with_mobileposer(self, full_imu_data: np.ndarray, 
                                 timestamp: float) -> Optional[PoseEstimate]:
        """Process IMU data with MobilePoser to get head pose."""
        if full_imu_data is None:
            return None
            
        try:
            # Apply temporal feedback to MobilePoser if enabled
            if self.enable_temporal_feedback and self.previous_fused_pose is not None:
                # Modify MobilePoser state with previous fused pose
                self._apply_temporal_feedback_to_mobileposer()
            
            # Forward through MobilePoser
            imu_tensor = torch.from_numpy(full_imu_data).float()
            
            with torch.no_grad():
                pose, joints, translation, contact = self.mobileposer.forward_online(imu_tensor)
            
            # Extract head pose
            head_position, head_orientation = self._extract_head_pose_from_mobileposer(
                pose, joints, translation
            )
            
            # Estimate confidence
            confidence = self._estimate_imu_confidence(contact, joints)
            
            return PoseEstimate(
                translation=head_position,
                rotation=head_orientation,
                confidence=confidence,
                timestamp=timestamp,
                source="imu"
            )
            
        except Exception as e:
            self.logger.error(f"Error processing with MobilePoser: {e}")
            return None
    
    def _extract_head_pose_from_mobileposer(self, pose, joints, translation):
        """Extract head position and orientation from MobilePoser output."""
        # Convert tensors to numpy
        if isinstance(joints, torch.Tensor):
            joints_np = joints.cpu().numpy()
        else:
            joints_np = joints
            
        if isinstance(translation, torch.Tensor):
            translation_np = translation.cpu().numpy()
        else:
            translation_np = translation
        
        if isinstance(pose, torch.Tensor):
            pose_np = pose.cpu().numpy()
        else:
            pose_np = pose
        
        # Get head position (joint 15 is head in SMPL)
        if joints_np.shape[0] >= 16:
            head_position_rel = joints_np[15]  # Head joint relative to root
            head_position_global = translation_np + head_position_rel
        else:
            head_position_global = translation_np
        
        # Get head orientation
        if pose_np.shape == (24, 9):
            head_rotation = pose_np[15].reshape(3, 3)
        elif pose_np.shape == (24, 3, 3):
            head_rotation = pose_np[15]
        else:
            head_rotation = np.eye(3)
        
        return head_position_global, head_rotation
    
    def _estimate_imu_confidence(self, contact, joints) -> float:
        """Estimate confidence of IMU-based pose estimate."""
        base_confidence = 0.8  # Base confidence for IMU
        
        # Adjust based on foot contact probabilities
        if contact is not None:
            if isinstance(contact, torch.Tensor):
                contact_np = contact.cpu().numpy()
                contact_confidence = float(np.mean(contact_np))
                base_confidence = 0.6 + 0.4 * contact_confidence
        
        # Adjust based on joint consistency (simplified)
        if joints is not None:
            # Could add more sophisticated joint consistency checks
            pass
        
        return base_confidence
    
    def _apply_temporal_feedback_to_mobileposer(self):
        """Apply temporal feedback from previous fused pose to MobilePoser."""
        if self.previous_fused_pose is None:
            return
        
        try:
            # This would require modifying MobilePoser's internal state
            # For now, we implement a simplified version
            
            # Update MobilePoser's last known pose
            prev_pos = self.previous_fused_pose.translation
            prev_rot = self.previous_fused_pose.rotation
            
            # Convert to MobilePoser's expected format and update internal state
            # This is a placeholder for actual implementation
            
            self.logger.debug("Applied temporal feedback to MobilePoser")
            
        except Exception as e:
            self.logger.error(f"Error applying temporal feedback: {e}")
    
    def _calculate_ensemble_weights(self, imu_pose: Optional[PoseEstimate],
                                   slam_output: Optional[SlamOutput]) -> Tuple[float, float]:
        """Calculate dynamic ensemble weights."""
        if imu_pose is None:
            return (0.0, 1.0) if slam_output and slam_output.pose is not None else (1.0, 0.0)
        
        if slam_output is None or slam_output.pose is None:
            return (1.0, 0.0)
        
        # Use weight calculator
        imu_weight, slam_weight = self.weight_calculator.calculate_weights(
            imu_pose.confidence,
            slam_output,
            imu_pose_available=True
        )
        
        return (imu_weight, slam_weight)
    
    def _fuse_poses_with_feedback(self, 
                                 imu_pose: Optional[PoseEstimate],
                                 slam_output: Optional[SlamOutput],
                                 weights: Tuple[float, float],
                                 timestamp: float) -> Optional[PoseEstimate]:
        """
        Fuse IMU and SLAM poses with temporal feedback.
        
        Args:
            imu_pose: IMU-based pose estimate
            slam_output: SLAM output
            weights: (IMU weight, SLAM weight)
            timestamp: Current timestamp
            
        Returns:
            Fused pose estimate
        """
        imu_weight, slam_weight = weights
        
        # Handle cases where one estimate is missing
        if imu_pose is None and slam_output is None:
            return None
        elif imu_pose is None:
            fused_pose = self._slam_output_to_pose_estimate(slam_output, timestamp)
        elif slam_output is None or slam_output.pose is None:
            fused_pose = imu_pose
        else:
            # Fuse both estimates
            slam_pose = self._slam_output_to_pose_estimate(slam_output, timestamp)
            fused_pose = self._weighted_pose_fusion(imu_pose, slam_pose, weights)
        
        # Apply temporal smoothing
        if self.enable_temporal_feedback and fused_pose is not None:
            fused_pose = self._apply_temporal_smoothing(fused_pose)
        
        # Update history
        if fused_pose is not None:
            self.previous_fused_pose = fused_pose
            self._update_pose_history(fused_pose)
        
        return fused_pose
    
    def _slam_output_to_pose_estimate(self, slam_output: SlamOutput, 
                                     timestamp: float) -> PoseEstimate:
        """Convert SLAM output to PoseEstimate format."""
        pose_matrix = slam_output.pose
        position = pose_matrix[:3, 3]
        orientation = pose_matrix[:3, :3]
        
        return PoseEstimate(
            translation=position,
            rotation=orientation,
            confidence=slam_output.confidence,
            timestamp=timestamp,
            source="slam"
        )
    
    def _weighted_pose_fusion(self, imu_pose: PoseEstimate, slam_pose: PoseEstimate,
                             weights: Tuple[float, float]) -> PoseEstimate:
        """Fuse poses using weighted combination."""
        imu_weight, slam_weight = weights
        
        # Weighted translation (favor SLAM for translation)
        translation_imu_weight = imu_weight * 0.3  # Reduce IMU weight for translation
        translation_slam_weight = slam_weight * 1.7  # Increase SLAM weight for translation
        
        # Normalize translation weights
        total_trans_weight = translation_imu_weight + translation_slam_weight
        if total_trans_weight > 0:
            translation_imu_weight /= total_trans_weight
            translation_slam_weight /= total_trans_weight
        
        fused_translation = (
            translation_imu_weight * imu_pose.translation +
            translation_slam_weight * slam_pose.translation
        )
        
        # Weighted orientation (favor IMU for orientation)
        orientation_imu_weight = imu_weight * 1.5  # Increase IMU weight for orientation
        orientation_slam_weight = slam_weight * 0.5  # Reduce SLAM weight for orientation
        
        # Normalize orientation weights
        total_orient_weight = orientation_imu_weight + orientation_slam_weight
        if total_orient_weight > 0:
            orientation_imu_weight /= total_orient_weight
            orientation_slam_weight /= total_orient_weight
        
        # SLERP for rotation
        slerp_t = orientation_slam_weight
        fused_orientation = self._slerp_rotation_matrices(
            imu_pose.rotation, slam_pose.rotation, slerp_t
        )
        
        # Combined confidence
        fused_confidence = imu_weight * imu_pose.confidence + slam_weight * slam_pose.confidence
        
        return PoseEstimate(
            translation=fused_translation,
            rotation=fused_orientation,
            confidence=fused_confidence,
            timestamp=imu_pose.timestamp,
            source="fused"
        )
    
    def _apply_temporal_smoothing(self, current_pose: PoseEstimate) -> PoseEstimate:
        """Apply temporal smoothing to reduce jitter."""
        if self.previous_fused_pose is None or len(self.pose_history) < 2:
            return current_pose
        
        # Simple exponential smoothing
        smoothing_factor = 0.7  # Adjust based on desired smoothness vs. responsiveness
        
        smoothed_translation = (
            smoothing_factor * current_pose.translation +
            (1 - smoothing_factor) * self.previous_fused_pose.translation
        )
        
        # For rotation, use SLERP
        smoothed_orientation = self._slerp_rotation_matrices(
            self.previous_fused_pose.rotation,
            current_pose.rotation,
            smoothing_factor
        )
        
        return PoseEstimate(
            translation=smoothed_translation,
            rotation=smoothed_orientation,
            confidence=current_pose.confidence,
            timestamp=current_pose.timestamp,
            source=current_pose.source
        )
    
    def _slerp_rotation_matrices(self, R1: np.ndarray, R2: np.ndarray, t: float) -> np.ndarray:
        """Spherical linear interpolation between rotation matrices."""
        # Convert to quaternions, SLERP, then back to matrices
        q1 = self._rotation_matrix_to_quaternion(R1)
        q2 = self._rotation_matrix_to_quaternion(R2)
        q_interp = self._slerp_quaternions(q1, q2, t)
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
    
    def _update_pose_history(self, pose: PoseEstimate):
        """Update pose history for temporal analysis."""
        self.pose_history.append(pose)
        
        if len(self.pose_history) > self.max_history_size:
            self.pose_history = self.pose_history[-self.max_history_size:]
    
    def _update_stats(self, slam_output: Optional[SlamOutput], 
                     weights: Tuple[float, float], processing_time: float):
        """Update performance statistics."""
        self.processing_stats['frames_processed'] += 1
        self.processing_stats['processing_times'].append(processing_time)
        
        if slam_output and slam_output.pose is not None:
            self.processing_stats['slam_frames'] += 1
            mode_used = slam_output.mode_used.value
            self.processing_stats['mode_distribution'][mode_used] += 1
            
            if weights[0] > 0 and weights[1] > 0:
                self.processing_stats['ensemble_frames'] += 1
        else:
            self.processing_stats['imu_only_frames'] += 1
        
        # Update average weights
        imu_weight, slam_weight = weights
        n = self.processing_stats['frames_processed']
        avg_imu = self.processing_stats['average_weights']['imu']
        avg_slam = self.processing_stats['average_weights']['slam']
        
        self.processing_stats['average_weights']['imu'] = (avg_imu * (n-1) + imu_weight) / n
        self.processing_stats['average_weights']['slam'] = (avg_slam * (n-1) + slam_weight) / n
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        base_stats = self.processing_stats.copy()
        
        if base_stats['processing_times']:
            base_stats['average_processing_time'] = np.mean(base_stats['processing_times'])
            base_stats['max_processing_time'] = np.max(base_stats['processing_times'])
            base_stats['fps'] = 1.0 / base_stats['average_processing_time']
        
        # Add SLAM-specific stats
        slam_stats = self.adaptive_slam.get_performance_stats()
        base_stats['slam_stats'] = slam_stats
        
        # Add weight calculator stats
        weight_breakdown = self.weight_calculator.get_weight_breakdown(
            0.8,  # dummy values for latest breakdown
            SlamOutput(confidence=0.7, tracking_state="tracking")
        )
        base_stats['latest_weight_breakdown'] = weight_breakdown
        
        return base_stats
    
    def reset(self):
        """Reset ensemble state."""
        self.mobileposer.reset()
        self.adaptive_slam.reset()
        self.weight_calculator.reset()
        
        self.previous_fused_pose = None
        self.pose_history.clear()
        self.frame_count = 0
        
        # Reset stats
        self.processing_stats = {
            'frames_processed': 0,
            'slam_frames': 0,
            'imu_only_frames': 0,
            'ensemble_frames': 0,
            'mode_distribution': {mode.value: 0 for mode in SlamMode},
            'average_weights': {'imu': 0.0, 'slam': 0.0},
            'processing_times': []
        }
    
    def shutdown(self):
        """Shutdown ensemble system."""
        self.adaptive_slam.shutdown()
        self.logger.info("Adaptive head pose ensemble shutdown")