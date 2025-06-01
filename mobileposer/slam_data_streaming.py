"""
SLAM-integrated data module with streaming support for training MobilePoser with Visual-Inertial SLAM head pose estimates.

This module extends the standard data loading to include:
1. RGB video loading from Nymeria dataset with streaming
2. Real-time SLAM processing during training  
3. Adaptive fusion of IMU and SLAM head poses
4. Memory-efficient streaming for large datasets
"""

import os
import cv2
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Iterator
from torch.utils.data import Dataset, DataLoader, IterableDataset
import lightning as L
from threading import Thread, Lock
from queue import Queue
import time
import logging
import random

from mobileposer.data import PoseDataset, PoseIterableDataset, pad_seq
from mobileposer.models.real_orbslam3 import RealOrbSlam3Interface
from mobileposer.models.adaptive_slam import AdaptiveSlamInterface, SlamInput, SlamMode
from mobileposer.config import paths, train_hypers, finetune_hypers, datasets, amass
from mobileposer.slam_selector import slam_selector
import mobileposer.articulate as art


class StreamingSlamPoseDataset(IterableDataset):
    """
    Streaming dataset that includes SLAM processing for head pose estimation.
    Loads RGB videos alongside IMU data and processes them through SLAM on-the-fly.
    """
    
    def __init__(self, fold: str = 'train', 
                 finetune: str = None,
                 slam_enabled: bool = True,
                 slam_type: str = "adaptive",
                 stream_buffer_size: int = 10,
                 cache_slam_results: bool = True):
        """
        Initialize streaming SLAM dataset.
        
        Args:
            fold: 'train' or 'test'
            finetune: Dataset name for finetuning
            slam_enabled: Whether to enable SLAM processing
            slam_type: Type of SLAM ("adaptive", "vi", "mono", "mock")
            stream_buffer_size: Number of sequences to buffer
            cache_slam_results: Cache SLAM results for efficiency
        """
        super().__init__()
        
        self.fold = fold
        self.finetune = finetune
        self.slam_enabled = slam_enabled
        self.slam_type = slam_type
        self.stream_buffer_size = stream_buffer_size
        self.cache_slam_results = cache_slam_results
        
        # Initialize components
        self.bodymodel = art.model.ParametricModel(paths.smpl_file)
        self.combos = list(amass.combos.items())
        
        # SLAM components
        self.slam_interface = None
        self.slam_cache = {} if cache_slam_results else None
        self.video_captures = {}
        self.video_capture_lock = Lock()
        
        # RGB video root path
        self.rgb_root = Path("/mnt/nas2/naoto/nymeria_dataset/data_video_main_rgb")
        
        # Dataset configuration
        # For Nymeria training data, files are in main processed_datasets folder
        if finetune and 'nymeria' in finetune and 'train' in finetune:
            self.data_folder = paths.processed_datasets
        else:
            self.data_folder = paths.processed_datasets / ('eval' if finetune else '')
        self.data_files = self._get_data_files()
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        if self.slam_enabled:
            self._initialize_slam()
    
    def _get_data_files(self) -> List[str]:
        """Get list of data files to process."""
        if self.fold == 'train':
            if self.finetune:
                # For Nymeria, we have multiple training files
                if 'nymeria' in self.finetune:
                    # Look for nymeria training files
                    files = []
                    if self.data_folder.exists():
                        for f in os.listdir(self.data_folder):
                            if 'nymeria' in f and 'train' in f and f.endswith('.pt'):
                                # Filter to match the specific type (aria/xsens)
                                if 'aria' in self.finetune and 'aria' in f:
                                    files.append(f)
                                elif 'xsens' in self.finetune and 'xsens' in f:
                                    files.append(f)
                    return sorted(files)  # Sort for consistent ordering
                else:
                    return [datasets.finetune_datasets.get(self.finetune, '')]
            else:
                return [x.name for x in self.data_folder.iterdir() if x.is_file() and x.suffix == '.pt']
        else:  # test
            return [datasets.test_datasets.get(self.finetune, 'nymeria_aria_xdata_test.pt')]
    
    def _initialize_slam(self):
        """Initialize SLAM system."""
        try:
            # Use slam_selector for proper SLAM initialization
            self.slam_interface = slam_selector.create_slam(
                slam_type=self.slam_type,
                mode="visual_inertial" if self.slam_type == "vi" else "monocular",
                allow_mock=True  # Allow mock for testing
            )
            self.logger.info(f"Initialized {self.slam_type} SLAM system")
        except Exception as e:
            self.logger.error(f"Failed to initialize SLAM: {e}")
            self.slam_enabled = False
    
    def _get_rgb_video_path(self, sequence_name: str) -> Optional[Path]:
        """Get RGB video path for a sequence."""
        video_dir = self.rgb_root / sequence_name
        video_path = video_dir / "video_main_rgb.mp4"
        
        if video_path.exists():
            return video_path
        else:
            self.logger.warning(f"RGB video not found: {video_path}")
            return None
    
    def _load_rgb_frame(self, video_path: Path, frame_idx: int) -> Optional[np.ndarray]:
        """Load RGB frame from video file with caching."""
        with self.video_capture_lock:
            video_key = str(video_path)
            
            # Get or create video capture
            if video_key not in self.video_captures:
                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened():
                    self.logger.error(f"Failed to open video: {video_path}")
                    return None
                self.video_captures[video_key] = cap
            
            cap = self.video_captures[video_key]
            
            # Seek to frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                return None
            
            # Convert BGR to RGB
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    def _process_slam(self, rgb_frame: np.ndarray, imu_data: torch.Tensor, 
                      timestamp: float) -> Optional[torch.Tensor]:
        """Process frame through SLAM and return head pose."""
        if not self.slam_enabled or rgb_frame is None:
            return None
        
        # Check cache first
        cache_key = f"{timestamp:.6f}"
        if self.slam_cache is not None and cache_key in self.slam_cache:
            return self.slam_cache[cache_key]
        
        # Extract head IMU data (sensor index 4 for Nymeria)
        head_imu_idx = 4
        if imu_data.shape[0] >= 60:  # 5 sensors * 12 values
            # Extract head sensor data
            acc_start = head_imu_idx * 3
            ori_start = 15 + head_imu_idx * 9  # 5*3 for acc, then orientations
            
            head_acc = imu_data[acc_start:acc_start+3].numpy()
            # For simplicity, assume zero angular velocity
            head_gyro = np.zeros(3)
            head_imu_6dof = np.concatenate([head_acc, head_gyro])
        else:
            head_imu_6dof = None
        
        # Process with SLAM
        try:
            if self.slam_type == "adaptive":
                slam_input = SlamInput(
                    rgb_frame=rgb_frame,
                    head_imu_data=head_imu_6dof,
                    timestamp=timestamp
                )
                slam_output = self.slam_interface.process_frame(slam_input)
                
                if slam_output.pose is not None:
                    # Extract head pose
                    position = slam_output.pose[:3, 3] * slam_output.scale_factor
                    orientation = slam_output.pose[:3, :3].reshape(-1)
                    head_pose = torch.FloatTensor(np.concatenate([position, orientation]))
                    
                    # Cache result
                    if self.slam_cache is not None:
                        self.slam_cache[cache_key] = head_pose
                    
                    return head_pose
            else:
                # Standard SLAM processing
                result = self.slam_interface.process_frame(rgb_frame, timestamp)
                if result and result.get('pose') is not None:
                    pose_matrix = result['pose']
                    position = pose_matrix[:3, 3]
                    orientation = pose_matrix[:3, :3].reshape(-1)
                    head_pose = torch.FloatTensor(np.concatenate([position, orientation]))
                    
                    if self.slam_cache is not None:
                        self.slam_cache[cache_key] = head_pose
                    
                    return head_pose
        except Exception as e:
            self.logger.error(f"SLAM processing error: {e}")
        
        return None
    
    def _process_sequence(self, acc: torch.Tensor, ori: torch.Tensor, 
                         pose: torch.Tensor, tran: torch.Tensor,
                         joint: Optional[torch.Tensor], foot: Optional[torch.Tensor],
                         sequence_name: str) -> Iterator[Tuple]:
        """Process a single sequence and yield windows."""
        # Normalize IMU data
        acc = acc[:, :5] / amass.acc_scale
        ori = ori[:, :5]
        
        # Get RGB video path
        video_path = self._get_rgb_video_path(sequence_name) if self.slam_enabled else None
        
        # Forward kinematics
        try:
            pose_global, joint_glb = self.bodymodel.forward_kinematics(pose=pose.view(-1, 216))
            pose_use = pose if self.fold == 'test' else pose_global.view(-1, 24, 3, 3)
            joint_use = joint_glb.view(-1, 24, 3)
        except Exception as e:
            self.logger.error(f"FK error: {e}")
            pose_use = pose
            joint_use = joint
        
        # Process with each IMU combination
        for _, combo in self.combos:
            combo_acc = torch.zeros_like(acc)
            combo_ori = torch.zeros_like(ori)
            combo_acc[:, combo] = acc[:, combo]
            combo_ori[:, combo] = ori[:, combo]
            imu_input = torch.cat([combo_acc.flatten(1), combo_ori.flatten(1)], dim=1)
            
            win_len = len(imu_input) if self.fold == 'test' else datasets.window_length
            
            for start in range(0, len(imu_input), win_len):
                end = min(start + win_len, len(imu_input))
                
                # Extract window data
                imu_chunk = imu_input[start:end].float()
                pose_chunk = pose_use[start:end] if pose_use is not None else None
                joint_chunk = joint_use[start:end] if joint_use is not None else None
                tran_chunk = tran[start:end] if tran is not None else None
                
                # Process SLAM for this window
                slam_head_poses = []
                if self.slam_enabled and video_path is not None:
                    for i in range(len(imu_chunk)):
                        frame_idx = start + i
                        # Convert IMU frame index (30 FPS) to RGB frame index (15 FPS)
                        # RGB videos are 15 FPS while IMU is 30 FPS, so we need every 2nd frame
                        rgb_frame_idx = frame_idx // 2
                        rgb_frame = self._load_rgb_frame(video_path, rgb_frame_idx)
                        if rgb_frame is not None:
                            # Timestamp is based on IMU timing (30 FPS)
                            timestamp = frame_idx / 30.0
                            slam_pose = self._process_slam(rgb_frame, imu_chunk[i], timestamp)
                            slam_head_poses.append(slam_pose)
                        else:
                            slam_head_poses.append(None)
                
                # Convert slam_head_poses to tensor (None if no valid poses)
                if slam_head_poses and any(p is not None for p in slam_head_poses):
                    # Fill None values with zeros
                    slam_tensor = torch.stack([
                        p if p is not None else torch.zeros(12) 
                        for p in slam_head_poses
                    ])
                else:
                    slam_tensor = None
                
                # Convert pose to r6d
                if pose_chunk is not None:
                    num_pred = len(amass.pred_joints_set)
                    r6d = art.math.rotation_matrix_to_r6d(pose_chunk).reshape(-1, 24, 6)
                    r6d = r6d[:, amass.pred_joints_set].reshape(-1, 6 * num_pred)
                else:
                    r6d = None
                
                # Yield data based on fold
                if self.fold == 'test' or self.finetune:
                    yield (imu_chunk, r6d, joint_chunk, tran_chunk, slam_tensor)
                else:
                    # Training: include velocity and contact
                    # Velocity calculation
                    if tran_chunk is not None and len(tran_chunk) > 1:
                        root_vel = torch.cat((torch.zeros(1, 3), tran_chunk[1:] - tran_chunk[:-1]))
                    else:
                        root_vel = torch.zeros(len(tran_chunk) if tran_chunk is not None else 1, 3)
                    
                    if joint_chunk is not None and len(joint_chunk) > 1:
                        vel_all = torch.cat((torch.zeros(1, 24, 3), torch.diff(joint_chunk, dim=0)))
                        vel_all[:, 0] = root_vel
                        vel_out = vel_all * (datasets.fps / amass.vel_scale)
                    else:
                        vel_out = torch.zeros(len(joint_chunk) if joint_chunk is not None else 1, 24, 3)
                    
                    # Foot contact
                    contact_chunk = None
                    if foot is not None and len(foot) > start:
                        contact_chunk = foot[start:end]
                        if contact_chunk.shape[1] == 4:
                            contact_chunk = torch.stack([contact_chunk[:, 1], contact_chunk[:, 3]], dim=1)
                    
                    yield (imu_chunk, r6d, joint_chunk, tran_chunk, vel_out, contact_chunk, slam_tensor)
    
    def __iter__(self) -> Iterator[Tuple]:
        """Iterate through sequences with streaming."""
        # Shuffle files for each epoch
        file_list = self.data_files.copy()
        random.shuffle(file_list)
        
        # Buffer for sequences
        sequence_buffer = []
        
        for data_file in file_list:
            try:
                # Load data file
                file_path = self.data_folder / data_file
                file_data = torch.load(file_path)
                
                # Extract data
                accs = file_data.get('acc', [])
                oris = file_data.get('ori', [])
                poses = file_data.get('pose', [])
                trans = file_data.get('tran', [])
                joints = file_data.get('joint', [None] * len(poses))
                foots = file_data.get('contact', [None] * len(poses))
                seq_names = file_data.get('sequence_name', [''] * len(poses))
                
                # Convert quaternion to rotation matrix if needed
                if len(oris) > 0 and oris[0].dim() == 3 and oris[0].shape[-1] == 4:
                    converted_oris = []
                    for q in oris:
                        T, N, _ = q.shape
                        rotm = art.math.quaternion_to_rotation_matrix(q.view(-1, 4)).view(T, N, 3, 3)
                        converted_oris.append(rotm)
                    oris = converted_oris
                
                # Add sequences to buffer
                for i, (acc, ori, pose, tran, joint, foot) in enumerate(
                    zip(accs, oris, poses, trans, joints, foots)
                ):
                    seq_name = seq_names[i] if i < len(seq_names) else f"unknown_{i}"
                    sequence_buffer.append((acc, ori, pose, tran, joint, foot, seq_name))
                    
                    # Process buffer when it reaches desired size
                    if len(sequence_buffer) >= self.stream_buffer_size:
                        # Process sequences in random order
                        random.shuffle(sequence_buffer)
                        for seq_data in sequence_buffer:
                            yield from self._process_sequence(*seq_data)
                        sequence_buffer = []
                
            except Exception as e:
                self.logger.error(f"Error loading {data_file}: {e}")
                continue
        
        # Process remaining sequences
        if sequence_buffer:
            random.shuffle(sequence_buffer)
            for seq_data in sequence_buffer:
                yield from self._process_sequence(*seq_data)
    
    def cleanup(self):
        """Clean up resources."""
        # Close video captures
        with self.video_capture_lock:
            for cap in self.video_captures.values():
                cap.release()
            self.video_captures.clear()
        
        # Shutdown SLAM
        if self.slam_interface is not None:
            try:
                self.slam_interface.shutdown()
            except:
                pass


class SlamPoseDataModule(L.LightningDataModule):
    """
    Lightning data module with SLAM integration and streaming support.
    """
    
    def __init__(self, 
                 finetune: str = None,
                 max_sequences: int = None,
                 streaming: bool = True,
                 slam_enabled: bool = True,
                 slam_type: str = "adaptive",
                 cache_slam_results: bool = True):
        super().__init__()
        
        self.finetune = finetune
        self.max_sequences = max_sequences
        self.streaming = streaming
        self.slam_enabled = slam_enabled
        self.slam_type = slam_type
        self.cache_slam_results = cache_slam_results
        
        self.hypers = finetune_hypers if finetune else train_hypers
        
        # Adjust batch size for streaming with SLAM
        if self.streaming:
            original_bs = self.hypers.batch_size
            # SLAM processing is expensive, reduce batch size more
            self.batch_size = max(16, original_bs // 16)
            self.num_workers = min(4, self.hypers.num_workers)
        else:
            self.batch_size = self.hypers.batch_size
            self.num_workers = self.hypers.num_workers
            
        self.logger = logging.getLogger(__name__)
    
    def setup(self, stage: str):
        """Set up datasets."""
        if stage == 'fit':
            if self.streaming:
                # Use streaming dataset for training
                stream_buffer_size = getattr(self.hypers, 'stream_buffer_size', 10)
                self.train_dataset = StreamingSlamPoseDataset(
                    fold='train',
                    finetune=self.finetune,
                    slam_enabled=self.slam_enabled,
                    slam_type=self.slam_type,
                    stream_buffer_size=stream_buffer_size,
                    cache_slam_results=self.cache_slam_results
                )
                
                # For validation, use streaming with smaller buffer
                self.val_dataset = StreamingSlamPoseDataset(
                    fold='train',
                    finetune=self.finetune,
                    slam_enabled=self.slam_enabled,
                    slam_type=self.slam_type,
                    stream_buffer_size=2,
                    cache_slam_results=True
                )
            else:
                # Non-streaming mode not implemented for SLAM
                raise NotImplementedError("Non-streaming mode not implemented for SLAM training")
    
    def _collate_slam(self, batch):
        """Custom collate function for SLAM data."""
        # Separate regular data and SLAM poses
        if len(batch[0]) == 5:  # Test/finetune mode
            inputs, poses, joints, trans, slam_poses = zip(*batch)
            has_velocity = False
        else:  # Training mode
            inputs, poses, joints, trans, vels, foots, slam_poses = zip(*batch)
            has_velocity = True
        
        # Pad sequences
        def _pad(sequence):
            seq_list = list(sequence)
            # Handle None values
            seq_list = [s if s is not None else torch.zeros(1, s.shape[-1] if len(s.shape) > 1 else 1) 
                       for s in seq_list]
            padded = torch.nn.utils.rnn.pad_sequence(seq_list, batch_first=True)
            lengths = [s.shape[0] for s in sequence]
            return padded, lengths
        
        # Pad regular data
        inputs_padded, input_lengths = _pad(inputs)
        poses_padded, pose_lengths = _pad(poses)
        joints_padded, joint_lengths = _pad(joints)
        trans_padded, trans_lengths = _pad(trans)
        
        # Pad SLAM poses
        slam_valid = [s for s in slam_poses if s is not None]
        if slam_valid:
            slam_padded, slam_lengths = _pad(slam_poses)
        else:
            slam_padded = None
            slam_lengths = None
        
        outputs = {
            'poses': poses_padded, 
            'joints': joints_padded, 
            'trans': trans_padded,
            'slam_head_poses': slam_padded
        }
        output_lengths = {
            'poses': pose_lengths,
            'joints': joint_lengths,
            'trans': trans_lengths,
            'slam_head_poses': slam_lengths
        }
        
        if has_velocity:
            vels_padded, vel_lengths = _pad(vels)
            foots_padded, foot_lengths = _pad(foots)
            outputs['vels'] = vels_padded
            outputs['foot_contacts'] = foots_padded
            output_lengths['vels'] = vel_lengths
            output_lengths['foot_contacts'] = foot_lengths
        
        return (inputs_padded, input_lengths), (outputs, output_lengths)
    
    def train_dataloader(self):
        """Get training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=self._collate_slam,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def val_dataloader(self):
        """Get validation dataloader."""
        val_batch_size = max(8, self.batch_size // 2)
        return DataLoader(
            self.val_dataset,
            batch_size=val_batch_size,
            collate_fn=self._collate_slam,
            num_workers=min(2, self.num_workers),
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def teardown(self, stage: str):
        """Clean up resources."""
        if hasattr(self, 'train_dataset') and hasattr(self.train_dataset, 'cleanup'):
            self.train_dataset.cleanup()
        if hasattr(self, 'val_dataset') and hasattr(self.val_dataset, 'cleanup'):
            self.val_dataset.cleanup()