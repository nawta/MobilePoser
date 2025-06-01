"""
SLAM-integrated MobilePoser network that fuses IMU predictions with SLAM head pose estimates.

This network extends MobilePoserNet to:
1. Accept SLAM head pose estimates as additional input
2. Use adaptive fusion to combine IMU and SLAM predictions
3. Improve overall pose accuracy, especially for head translation
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Dict, Any

from mobileposer.models.net import MobilePoserNet
from mobileposer.models.fusion import PoseFusionModule
from mobileposer.models.adaptive_slam import EnsembleWeightCalculator
import mobileposer.articulate as art


class SlamIntegratedMobilePoserNet(MobilePoserNet):
    """
    MobilePoser network with SLAM integration for improved head pose estimation.
    
    During training: Uses SLAM head poses as supervision signal for head joints
    During inference: Fuses IMU predictions with real-time SLAM estimates
    """
    
    def __init__(self, *args, use_slam_fusion: bool = True, **kwargs):
        """
        Initialize SLAM-integrated network.
        
        Args:
            use_slam_fusion: Whether to use SLAM fusion (can be disabled for baseline)
            *args, **kwargs: Arguments for base MobilePoserNet
        """
        super().__init__(*args, **kwargs)
        
        self.use_slam_fusion = use_slam_fusion
        
        # Head joint index in SMPL model
        self.head_joint_idx = 15
        
        # SLAM fusion components
        if self.use_slam_fusion:
            # Adaptive weight calculator for dynamic fusion
            self.weight_calculator = EnsembleWeightCalculator()
            
            # Learned fusion network for combining IMU and SLAM features
            self.head_fusion_net = nn.Sequential(
                nn.Linear(256 + 12, 128),  # IMU features + SLAM head pose
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 12)  # Output: refined head pose (3 pos + 9 rot)
            )
            
            # Confidence estimation network
            self.confidence_net = nn.Sequential(
                nn.Linear(256 + 12, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 2)  # Output: [IMU confidence, SLAM confidence]
            )
    
    def forward(self, acc: torch.Tensor, ori: torch.Tensor, 
                slam_head_pose: Optional[torch.Tensor] = None,
                return_intermediates: bool = False) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass with optional SLAM head pose integration.
        
        Args:
            acc: Acceleration data
            ori: Orientation data  
            slam_head_pose: SLAM-estimated head pose [B, T, 12] (3 pos + 9 rot)
            return_intermediates: Return intermediate features
            
        Returns:
            Tuple of (pose, joints, translation, contact, [optional intermediates])
        """
        # Get base predictions from IMU
        base_outputs = super().forward(acc, ori)
        pose, joints, translation, contact = base_outputs[:4]
        
        if not self.use_slam_fusion or slam_head_pose is None:
            # No SLAM fusion, return base predictions
            return base_outputs
        
        # Apply SLAM fusion for head pose refinement
        refined_outputs = self._apply_slam_fusion(
            pose, joints, translation, acc, ori, slam_head_pose
        )
        
        if return_intermediates:
            # Include fusion weights and confidences
            intermediates = {
                'base_pose': pose,
                'base_joints': joints,
                'fusion_weights': refined_outputs['weights'],
                'confidences': refined_outputs['confidences']
            }
            return (refined_outputs['pose'], refined_outputs['joints'], 
                   refined_outputs['translation'], contact, intermediates)
        else:
            return (refined_outputs['pose'], refined_outputs['joints'],
                   refined_outputs['translation'], contact)
    
    def _apply_slam_fusion(self, pose: torch.Tensor, joints: torch.Tensor,
                          translation: torch.Tensor, acc: torch.Tensor,
                          ori: torch.Tensor, slam_head_pose: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Apply SLAM fusion to refine head pose estimates.
        
        Args:
            pose: IMU-predicted pose [B, T, 24, 3, 3]
            joints: IMU-predicted joints [B, T, 24, 3]  
            translation: IMU-predicted translation [B, T, 3]
            acc: Acceleration data
            ori: Orientation data
            slam_head_pose: SLAM head pose [B, T, 12]
            
        Returns:
            Dictionary with refined pose, joints, translation, and metadata
        """
        B, T = pose.shape[:2]
        device = pose.device
        
        # Extract IMU features for fusion
        imu_features = self._extract_fusion_features(acc, ori)  # [B, T, 256]
        
        # Prepare SLAM head pose
        if slam_head_pose.dim() == 2:
            slam_head_pose = slam_head_pose.unsqueeze(0)  # Add batch dim
        if slam_head_pose.shape[0] == 1 and B > 1:
            slam_head_pose = slam_head_pose.expand(B, -1, -1)
        
        # Ensure temporal alignment
        if slam_head_pose.shape[1] != T:
            # Interpolate or pad to match temporal dimension
            slam_head_pose = self._align_temporal_dimension(slam_head_pose, T)
        
        # Concatenate features for fusion
        fusion_input = torch.cat([imu_features, slam_head_pose], dim=-1)
        
        # Predict fusion weights
        confidences = torch.sigmoid(self.confidence_net(fusion_input))  # [B, T, 2]
        imu_conf = confidences[..., 0:1]  # [B, T, 1]
        slam_conf = confidences[..., 1:2]  # [B, T, 1]
        
        # Calculate adaptive weights
        weights = self._calculate_adaptive_weights(imu_conf, slam_conf)
        
        # Refine head pose through fusion network
        refined_head_features = self.head_fusion_net(fusion_input)  # [B, T, 12]
        refined_head_pos = refined_head_features[..., :3]  # [B, T, 3]
        refined_head_rot = refined_head_features[..., 3:].reshape(B, T, 3, 3)  # [B, T, 3, 3]
        
        # Update pose and joints with refined head
        refined_pose = pose.clone()
        refined_joints = joints.clone()
        
        # Update head rotation in pose
        refined_pose[:, :, self.head_joint_idx] = refined_head_rot
        
        # Update head position in joints
        head_pos_local = joints[:, :, self.head_joint_idx] - translation
        head_pos_refined = refined_head_pos
        refined_joints[:, :, self.head_joint_idx] = head_pos_refined
        
        # Update global translation based on refined head position
        # This is a weighted update to maintain body consistency
        translation_update = (head_pos_refined - joints[:, :, self.head_joint_idx]).mean(dim=1, keepdim=True)
        refined_translation = translation + 0.3 * translation_update  # Conservative update
        
        # Propagate head refinement to connected joints (neck, etc.)
        refined_pose, refined_joints = self._propagate_head_refinement(
            refined_pose, refined_joints, refined_head_rot, refined_head_pos
        )
        
        return {
            'pose': refined_pose,
            'joints': refined_joints,
            'translation': refined_translation,
            'weights': weights,
            'confidences': confidences
        }
    
    def _extract_fusion_features(self, acc: torch.Tensor, ori: torch.Tensor) -> torch.Tensor:
        """
        Extract features from IMU data for fusion.
        
        Args:
            acc: Acceleration data
            ori: Orientation data
            
        Returns:
            Feature tensor [B, T, 256]
        """
        # Get features from joints module (reuse existing feature extraction)
        B, T = acc.shape[:2]
        
        # Flatten temporal dimension for processing
        acc_flat = acc.reshape(B * T, -1)
        ori_flat = ori.reshape(B * T, -1)
        
        # Process through first layers of joints module
        x = torch.cat([acc_flat, ori_flat], dim=-1)
        x = self.joints.fc1(x)
        x = self.joints.relu(x)
        x = self.joints.dropout(x)
        x = self.joints.fc2(x)
        features = self.joints.relu(x)
        
        # Reshape back to temporal
        features = features.reshape(B, T, -1)
        
        return features
    
    def _align_temporal_dimension(self, slam_pose: torch.Tensor, target_T: int) -> torch.Tensor:
        """
        Align SLAM pose temporal dimension with target.
        
        Args:
            slam_pose: SLAM pose tensor [B, T_slam, 12]
            target_T: Target temporal dimension
            
        Returns:
            Aligned tensor [B, target_T, 12]
        """
        B, T_slam, D = slam_pose.shape
        
        if T_slam == target_T:
            return slam_pose
        elif T_slam > target_T:
            # Downsample
            indices = torch.linspace(0, T_slam - 1, target_T).long()
            return slam_pose[:, indices]
        else:
            # Upsample with interpolation
            slam_pose_t = slam_pose.transpose(1, 2)  # [B, 12, T_slam]
            upsampled = torch.nn.functional.interpolate(
                slam_pose_t, size=target_T, mode='linear', align_corners=True
            )
            return upsampled.transpose(1, 2)  # [B, target_T, 12]
    
    def _calculate_adaptive_weights(self, imu_conf: torch.Tensor, 
                                   slam_conf: torch.Tensor) -> torch.Tensor:
        """
        Calculate adaptive fusion weights based on confidences.
        
        Args:
            imu_conf: IMU confidence [B, T, 1]
            slam_conf: SLAM confidence [B, T, 1]
            
        Returns:
            Fusion weights [B, T, 2] for [IMU, SLAM]
        """
        # Normalize confidences
        total_conf = imu_conf + slam_conf + 1e-6
        weights = torch.cat([imu_conf / total_conf, slam_conf / total_conf], dim=-1)
        
        # Apply bias based on modality strengths
        # IMU is typically better for orientation, SLAM for translation
        orientation_bias = 0.7  # Favor IMU for orientation
        translation_bias = 0.8  # Favor SLAM for translation
        
        # This is simplified - in practice, would separate orientation/translation weights
        weights = weights * 0.5 + 0.5  # Smooth weights
        
        return weights
    
    def _propagate_head_refinement(self, pose: torch.Tensor, joints: torch.Tensor,
                                   head_rot: torch.Tensor, head_pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Propagate head refinement to connected joints.
        
        Args:
            pose: Full pose tensor
            joints: Full joints tensor  
            head_rot: Refined head rotation
            head_pos: Refined head position
            
        Returns:
            Updated (pose, joints) tuple
        """
        # For now, just return as-is
        # In a full implementation, would update neck and upper spine
        return pose, joints
    
    def forward_online(self, imu_data: torch.Tensor,
                      slam_head_pose: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, ...]:
        """
        Online forward pass for real-time inference with SLAM.
        
        Args:
            imu_data: IMU sensor data
            slam_head_pose: Real-time SLAM head pose estimate
            
        Returns:
            Tuple of (pose, joints, translation, contact)
        """
        # Prepare IMU data
        if imu_data.dim() == 1:
            imu_data = imu_data.unsqueeze(0).unsqueeze(0)  # Add batch and time dims
        elif imu_data.dim() == 2:
            imu_data = imu_data.unsqueeze(0)  # Add batch dim
        
        # Split into acc and ori
        acc = imu_data[..., :self.joints.input_dim_acc].reshape(1, 1, 6, 3)
        ori = imu_data[..., self.joints.input_dim_acc:].reshape(1, 1, 6, 9)
        
        # Forward with SLAM fusion
        return self.forward(acc, ori, slam_head_pose)
    
    def training_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> torch.Tensor:
        """
        Training step with SLAM supervision.
        
        Args:
            batch: Training batch including optional SLAM head pose
            batch_idx: Batch index
            
        Returns:
            Loss value
        """
        # Unpack batch
        if len(batch) > 10:  # Extended batch with SLAM data
            acc, ori, pose, tran, joint, contact, shape, gender, _, _, slam_head_pose = batch
        else:
            slam_head_pose = None
            acc, ori, pose, tran, joint, contact, shape, gender, _, _ = batch
        
        # Forward pass with SLAM
        outputs = self.forward(acc, ori, slam_head_pose, return_intermediates=True)
        
        if len(outputs) > 4:  # With intermediates
            pred_pose, pred_joint, pred_tran, pred_contact, intermediates = outputs
            
            # Add SLAM supervision loss if available
            if slam_head_pose is not None and self.use_slam_fusion:
                slam_loss = self._compute_slam_supervision_loss(
                    pred_joint, slam_head_pose, intermediates
                )
            else:
                slam_loss = 0.0
        else:
            pred_pose, pred_joint, pred_tran, pred_contact = outputs
            slam_loss = 0.0
        
        # Compute standard losses
        loss_dict = self._compute_losses(
            (pred_pose, pred_joint, pred_tran, pred_contact),
            (pose, joint, tran, contact)
        )
        
        # Add SLAM supervision loss
        if slam_loss > 0:
            loss_dict['slam_supervision'] = slam_loss
            loss_dict['total'] = loss_dict['total'] + 0.1 * slam_loss  # Weight SLAM loss
        
        # Log losses
        for name, value in loss_dict.items():
            self.log(f'train_{name}_loss', value, on_step=True, on_epoch=True)
        
        return loss_dict['total']
    
    def _compute_slam_supervision_loss(self, pred_joints: torch.Tensor,
                                      slam_head_pose: torch.Tensor,
                                      intermediates: Dict[str, Any]) -> torch.Tensor:
        """
        Compute supervision loss between predicted and SLAM head poses.
        
        Args:
            pred_joints: Predicted joints [B, T, 24, 3]
            slam_head_pose: SLAM head pose [B, T, 12]
            intermediates: Intermediate outputs
            
        Returns:
            SLAM supervision loss
        """
        B, T = pred_joints.shape[:2]
        
        # Extract predicted head position
        pred_head_pos = pred_joints[:, :, self.head_joint_idx]  # [B, T, 3]
        
        # Extract SLAM head position
        slam_head_pos = slam_head_pose[..., :3]  # [B, T, 3]
        
        # Position loss
        pos_loss = torch.nn.functional.mse_loss(pred_head_pos, slam_head_pos)
        
        # Orientation loss (if using full pose)
        if intermediates is not None and 'base_pose' in intermediates:
            pred_head_rot = intermediates['base_pose'][:, :, self.head_joint_idx]  # [B, T, 3, 3]
            slam_head_rot = slam_head_pose[..., 3:].reshape(B, T, 3, 3)
            
            # Rotation matrix loss
            rot_loss = torch.nn.functional.mse_loss(pred_head_rot, slam_head_rot)
        else:
            rot_loss = 0.0
        
        # Combine losses
        total_loss = pos_loss + 0.5 * rot_loss
        
        return total_loss
    
    def configure_optimizers(self):
        """Configure optimizer for training."""
        return torch.optim.AdamW(self.parameters(), lr=self.hypers.learning_rate)