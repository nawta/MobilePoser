"""
SLAM-integrated data module for training MobilePoser with Visual-Inertial SLAM head pose estimates.

This module extends the standard PoseDataModule to include:
1. RGB video loading from Nymeria dataset
2. Real-time SLAM processing during training
3. Adaptive fusion of IMU and SLAM head poses
"""

import os
import cv2
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from torch.utils.data import Dataset, DataLoader
import lightning as L
from threading import Thread
from queue import Queue
import time
import logging

from mobileposer.data import PoseDataset, PoseIterableDataset
from mobileposer.models.real_orbslam3 import RealOrbSlam3Interface
from mobileposer.models.adaptive_slam import AdaptiveSlamInterface, SlamInput, SlamMode
from mobileposer.config import paths, train_hypers, finetune_hypers


class SlamPoseDataset(PoseDataset):
    """
    Extended dataset that includes SLAM processing for head pose estimation.
    Loads RGB videos alongside IMU data and processes them through SLAM.
    """
    
    def __init__(self, data: Dict[str, torch.Tensor], hypers, 
                 rgb_video_paths: Dict[str, str] = None,
                 slam_enabled: bool = True,
                 slam_type: str = "adaptive"):
        """
        Initialize SLAM-integrated dataset.
        
        Args:
            data: Standard pose dataset dictionary
            hypers: Hyperparameters
            rgb_video_paths: Mapping from sequence index to RGB video paths
            slam_enabled: Whether to enable SLAM processing
            slam_type: Type of SLAM ("adaptive", "vi", "mono", "mock")
        """
        super().__init__(data, hypers)
        
        self.rgb_video_paths = rgb_video_paths or {}
        self.slam_enabled = slam_enabled
        self.slam_type = slam_type
        
        # Initialize SLAM systems
        self.slam_systems = {}
        self.slam_cache = {}  # Cache SLAM results for efficiency
        
        # Video capture objects
        self.video_captures = {}
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        if self.slam_enabled:
            self._initialize_slam_systems()
    
    def _initialize_slam_systems(self):
        """Initialize SLAM systems for processing."""
        if self.slam_type == "adaptive":
            self.slam_interface = AdaptiveSlamInterface()
            if not self.slam_interface.initialize():
                self.logger.warning("Failed to initialize adaptive SLAM, disabling SLAM")
                self.slam_enabled = False
        elif self.slam_type == "vi":
            self.slam_interface = RealOrbSlam3Interface(
                mode="visual_inertial",
                enable_viewer=False
            )
            if not self.slam_interface.initialize():
                self.logger.warning("Failed to initialize VI-SLAM, disabling SLAM")
                self.slam_enabled = False
        elif self.slam_type == "mono":
            self.slam_interface = RealOrbSlam3Interface(
                mode="monocular",
                enable_viewer=False
            )
            if not self.slam_interface.initialize():
                self.logger.warning("Failed to initialize monocular SLAM, disabling SLAM")
                self.slam_enabled = False
        elif self.slam_type == "mock":
            from mobileposer.models.slam import MockSlamInterface
            self.slam_interface = MockSlamInterface()
            self.slam_interface.initialize()
        else:
            raise ValueError(f"Unknown SLAM type: {self.slam_type}")
    
    def _load_rgb_frame(self, seq_idx: int, frame_idx: int) -> Optional[np.ndarray]:
        """
        Load RGB frame from video file.
        
        Args:
            seq_idx: Sequence index
            frame_idx: Frame index within sequence
            
        Returns:
            RGB frame as numpy array or None if unavailable
        """
        if seq_idx not in self.rgb_video_paths:
            return None
        
        video_path = self.rgb_video_paths[seq_idx]
        
        # Open video if not already opened
        if seq_idx not in self.video_captures:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                self.logger.error(f"Failed to open video: {video_path}")
                return None
            self.video_captures[seq_idx] = cap
        
        cap = self.video_captures[seq_idx]
        
        # Seek to frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            return None
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame_rgb
    
    def _process_slam(self, rgb_frame: np.ndarray, imu_data: torch.Tensor, 
                      timestamp: float) -> Optional[Dict[str, Any]]:
        """
        Process frame through SLAM system.
        
        Args:
            rgb_frame: RGB image
            imu_data: IMU sensor data
            timestamp: Frame timestamp
            
        Returns:
            SLAM result dictionary or None
        """
        if not self.slam_enabled or rgb_frame is None:
            return None
        
        # Extract head IMU data (sensor index 4 for Nymeria)
        head_imu_idx = 4
        if imu_data.shape[0] >= (head_imu_idx + 1) * 12:
            head_data = imu_data[head_imu_idx * 12:(head_imu_idx + 1) * 12]
            head_acc = head_data[:3].numpy()
            # Simplified: assume zero angular velocity
            head_gyro = np.zeros(3)
            head_imu_6dof = np.concatenate([head_acc, head_gyro])
        else:
            head_imu_6dof = None
        
        # Process with SLAM
        if self.slam_type == "adaptive":
            slam_input = SlamInput(
                rgb_frame=rgb_frame,
                head_imu_data=head_imu_6dof,
                timestamp=timestamp
            )
            slam_output = self.slam_interface.process_frame(slam_input)
            
            if slam_output.pose is not None:
                return {
                    'pose': slam_output.pose,
                    'confidence': slam_output.confidence,
                    'scale_factor': slam_output.scale_factor,
                    'mode': slam_output.mode_used.value
                }
        else:
            # Standard SLAM processing
            if hasattr(self.slam_interface, 'process_frame_with_imu') and head_imu_6dof is not None:
                result = self.slam_interface.process_frame_with_imu(
                    rgb_frame, timestamp, head_imu_6dof
                )
            else:
                result = self.slam_interface.process_frame(rgb_frame, timestamp)
            
            if result is not None and result.get('pose') is not None:
                return {
                    'pose': result['pose'],
                    'confidence': result.get('confidence', 0.0),
                    'scale_factor': result.get('scale_factor', 1.0),
                    'mode': self.slam_type
                }
        
        return None
    
    def _get_slam_head_pose(self, seq_idx: int, frame_idx: int, 
                           imu_data: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Get SLAM-estimated head pose for given frame.
        
        Args:
            seq_idx: Sequence index
            frame_idx: Frame index
            imu_data: IMU data for the frame
            
        Returns:
            Head pose tensor [position(3), orientation(9)] or None
        """
        # Check cache first
        cache_key = (seq_idx, frame_idx)
        if cache_key in self.slam_cache:
            return self.slam_cache[cache_key]
        
        # Load RGB frame
        rgb_frame = self._load_rgb_frame(seq_idx, frame_idx)
        if rgb_frame is None:
            return None
        
        # Process through SLAM
        timestamp = frame_idx / 30.0  # Assume 30 FPS
        slam_result = self._process_slam(rgb_frame, imu_data, timestamp)
        
        if slam_result is None:
            return None
        
        # Extract head pose from SLAM result
        pose_matrix = slam_result['pose']  # 4x4 transformation
        
        # Extract position and orientation
        position = pose_matrix[:3, 3] * slam_result['scale_factor']
        orientation = pose_matrix[:3, :3].reshape(-1)  # Flatten to 9 values
        
        # Create head pose tensor
        head_pose = torch.FloatTensor(np.concatenate([position, orientation]))
        
        # Cache result
        self.slam_cache[cache_key] = head_pose
        
        return head_pose
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """
        Get item with SLAM-integrated head pose.
        
        Returns extended tuple with SLAM head pose if available.
        """
        # Get base data
        base_data = super().__getitem__(idx)
        
        if not self.slam_enabled:
            # Return base data with None for SLAM head pose
            return base_data + (None,)
        
        # Extract sequence and frame indices
        seq_idx = idx // self.data["acc"].shape[1]
        frame_idx = idx % self.data["acc"].shape[1]
        
        # Get IMU data
        acc, ori = base_data[0], base_data[1]
        imu_data = torch.cat([acc.reshape(-1), ori.reshape(-1)])
        
        # Get SLAM head pose
        slam_head_pose = self._get_slam_head_pose(seq_idx, frame_idx, imu_data)
        
        # Return extended data
        return base_data + (slam_head_pose,)
    
    def cleanup(self):
        """Clean up video captures and SLAM systems."""
        # Close all video captures
        for cap in self.video_captures.values():
            cap.release()
        self.video_captures.clear()
        
        # Shutdown SLAM
        if hasattr(self, 'slam_interface') and self.slam_interface is not None:
            self.slam_interface.shutdown()


class SlamPoseDataModule(L.LightningDataModule):
    """
    Lightning data module with SLAM integration.
    Extends standard data loading to include RGB videos and SLAM processing.
    """
    
    def __init__(self, 
                 finetune: str = None,
                 max_sequences: int = None,
                 streaming: bool = False,
                 slam_enabled: bool = True,
                 slam_type: str = "adaptive",
                 cache_slam_results: bool = True):
        """
        Initialize SLAM-integrated data module.
        
        Args:
            finetune: Dataset to finetune on
            max_sequences: Maximum sequences to load
            streaming: Use streaming dataset
            slam_enabled: Enable SLAM processing
            slam_type: Type of SLAM to use
            cache_slam_results: Cache SLAM results for efficiency
        """
        super().__init__()
        
        self.finetune = finetune
        self.max_sequences = max_sequences
        self.streaming = streaming
        self.slam_enabled = slam_enabled
        self.slam_type = slam_type
        self.cache_slam_results = cache_slam_results
        
        self.hypers = finetune_hypers if finetune else train_hypers
        self.batch_size = self.hypers.batch_size
        self.num_workers = self.hypers.num_workers
        
        # RGB video paths will be loaded in setup
        self.rgb_video_paths = {}
        
        self.logger = logging.getLogger(__name__)
    
    def _load_rgb_video_paths(self, dataset_name: str) -> Dict[int, str]:
        """
        Load RGB video paths for the dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Mapping from sequence index to video path
        """
        video_paths = {}
        
        if "nymeria" in dataset_name.lower():
            # Load Nymeria RGB videos
            rgb_root = Path("/mnt/nas2/naoto/nymeria_dataset/data_video_main_rgb")
            
            # Get sequence directories
            seq_dirs = [d for d in rgb_root.iterdir() 
                       if d.is_dir() and d.name not in ["data_summary.json", "download_summary.json"]]
            
            # Map sequence indices to video paths
            for idx, seq_dir in enumerate(sorted(seq_dirs)):
                video_path = seq_dir / "video_main_rgb.mp4"
                if video_path.exists():
                    video_paths[idx] = str(video_path)
                    
            self.logger.info(f"Found {len(video_paths)} RGB videos for {dataset_name}")
        
        return video_paths
    
    def setup(self, stage: str):
        """Set up datasets with SLAM integration."""
        # Load base datasets
        if self.finetune:
            train_data = torch.load(paths.processed_datasets / f"{self.finetune}.pt")
            val_data = train_data  # Use same data for validation in finetuning
            
            # Load RGB video paths
            self.rgb_video_paths = self._load_rgb_video_paths(self.finetune)
        else:
            # Load all training datasets
            train_datasets = []
            for dataset_name in ["ACCAD", "BMLhandball", "BMLmovi", "CMU", "MPI_HDM05"]:
                data_path = paths.processed_datasets / f"{dataset_name}.pt"
                if data_path.exists():
                    data = torch.load(data_path)
                    train_datasets.append(data)
                    
                    # Load RGB paths if available
                    if self.slam_enabled:
                        video_paths = self._load_rgb_video_paths(dataset_name)
                        # Offset indices for concatenated datasets
                        offset = sum(d["acc"].shape[0] for d in train_datasets[:-1])
                        for idx, path in video_paths.items():
                            self.rgb_video_paths[offset + idx] = path
            
            # Concatenate datasets
            train_data = self._concatenate_datasets(train_datasets)
            val_data = train_data
        
        # Limit sequences if specified
        if self.max_sequences:
            for key in train_data:
                train_data[key] = train_data[key][:self.max_sequences]
            for key in val_data:
                val_data[key] = val_data[key][:self.max_sequences]
        
        # Create datasets
        if self.streaming:
            self.train_dataset = PoseIterableDataset(train_data, self.hypers)
            self.val_dataset = PoseIterableDataset(val_data, self.hypers)
        else:
            self.train_dataset = SlamPoseDataset(
                train_data, self.hypers,
                rgb_video_paths=self.rgb_video_paths,
                slam_enabled=self.slam_enabled,
                slam_type=self.slam_type
            )
            self.val_dataset = SlamPoseDataset(
                val_data, self.hypers,
                rgb_video_paths=self.rgb_video_paths,
                slam_enabled=self.slam_enabled,
                slam_type=self.slam_type
            )
    
    def _concatenate_datasets(self, datasets: List[Dict]) -> Dict:
        """Concatenate multiple datasets."""
        if not datasets:
            return {}
        
        concatenated = {}
        for key in datasets[0].keys():
            concatenated[key] = torch.cat([d[key] for d in datasets], dim=0)
        
        return concatenated
    
    def train_dataloader(self) -> DataLoader:
        """Get training dataloader with SLAM support."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def val_dataloader(self) -> DataLoader:
        """Get validation dataloader with SLAM support."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def teardown(self, stage: str):
        """Clean up resources."""
        if hasattr(self, 'train_dataset') and hasattr(self.train_dataset, 'cleanup'):
            self.train_dataset.cleanup()
        if hasattr(self, 'val_dataset') and hasattr(self.val_dataset, 'cleanup'):
            self.val_dataset.cleanup()