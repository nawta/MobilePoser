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
from mobileposer.config import *
import mobileposer.articulate as art


@dataclass
class SensorData:
    """Container for synchronized sensor data."""
    imu_data: Optional[np.ndarray] = None      # IMU acceleration + orientation
    rgb_frame: Optional[np.ndarray] = None     # RGB image frame
    timestamp: float = 0.0                     # Synchronized timestamp
    frame_id: int = 0                          # Frame sequence number


class EnsemblePoseEstimator:
    """
    Ensemble pose estimation system combining MobilePoser (IMU-based) 
    with ORB-SLAM3 (visual-based) for improved accuracy and robustness.
    """
    
    def __init__(self, 
                 mobileposer_weights: str,
                 slam_type: str = "mock",
                 slam_config: str = None,
                 fusion_method: str = "weighted_average",
                 sync_tolerance: float = 0.05,
                 max_queue_size: int = 100):
        """
        Initialize ensemble pose estimator.
        
        Args:
            mobileposer_weights: Path to MobilePoser model weights
            slam_type: Type of SLAM system ("mock", "orb_slam3")
            slam_config: Path to SLAM configuration file
            fusion_method: Pose fusion strategy
            sync_tolerance: Maximum time difference for data synchronization (seconds)
            max_queue_size: Maximum size of data queues
        """
        self.sync_tolerance = sync_tolerance
        self.max_queue_size = max_queue_size
        
        # Initialize MobilePoser
        self.mobileposer = MobilePoserNet.load_from_checkpoint(mobileposer_weights)
        self.mobileposer.eval()
        
        # Initialize SLAM system
        self.slam_interface = create_slam_interface(slam_type, slam_config)
        if not self.slam_interface.initialize(slam_config):
            logging.warning(f"Failed to initialize {slam_type} SLAM, falling back to mock")
            self.slam_interface = create_slam_interface("mock")
            self.slam_interface.initialize()
        
        # Initialize fusion module
        self.fusion_module = PoseFusionModule(fusion_method=fusion_method)
        
        # Data queues for synchronization
        self.imu_queue = Queue(maxsize=max_queue_size)
        self.rgb_queue = Queue(maxsize=max_queue_size)
        self.result_queue = Queue(maxsize=max_queue_size)
        
        # Threading control
        self.processing_thread = None
        self.is_running = False
        self.data_lock = Lock()
        
        # State tracking
        self.last_imu_pose = None
        self.last_visual_pose = None
        self.frame_count = 0
        self.start_time = None
        
        # Performance metrics
        self.processing_times = []
        self.fusion_success_rate = 0.0
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def start_processing(self):
        """Start the ensemble processing pipeline."""
        if self.is_running:
            self.logger.warning("Processing already running")
            return
            
        self.is_running = True
        self.start_time = time.time()
        self.frame_count = 0
        
        # Start processing thread
        self.processing_thread = Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        
        self.logger.info("Ensemble pose estimator started")
        
    def stop_processing(self):
        """Stop the ensemble processing pipeline."""
        self.is_running = False
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5.0)
            
        # Clear queues
        self._clear_queues()
        
        # Reset systems
        self.mobileposer.reset()
        self.slam_interface.reset()
        self.fusion_module.reset()
        
        self.logger.info("Ensemble pose estimator stopped")
        
    def process_imu_data(self, imu_data: np.ndarray, timestamp: float):
        """
        Add IMU data to processing queue.
        
        Args:
            imu_data: IMU sensor data [acceleration + orientation matrices]
            timestamp: Data timestamp in seconds
        """
        try:
            self.imu_queue.put_nowait({
                'data': imu_data,
                'timestamp': timestamp
            })
        except:
            # Queue full, remove oldest item and add new one
            try:
                self.imu_queue.get_nowait()
                self.imu_queue.put_nowait({
                    'data': imu_data, 
                    'timestamp': timestamp
                })
            except:
                pass
                
    def process_rgb_frame(self, rgb_frame: np.ndarray, timestamp: float):
        """
        Add RGB frame to processing queue.
        
        Args:
            rgb_frame: RGB image (H, W, 3)
            timestamp: Frame timestamp in seconds
        """
        try:
            self.rgb_queue.put_nowait({
                'frame': rgb_frame,
                'timestamp': timestamp
            })
        except:
            # Queue full, remove oldest item and add new one
            try:
                self.rgb_queue.get_nowait()
                self.rgb_queue.put_nowait({
                    'frame': rgb_frame,
                    'timestamp': timestamp
                })
            except:
                pass
                
    def get_latest_pose(self) -> Optional[PoseEstimate]:
        """Get the latest fused pose estimate."""
        try:
            return self.result_queue.get_nowait()
        except Empty:
            return None
            
    def get_all_poses(self) -> List[PoseEstimate]:
        """Get all available pose estimates."""
        poses = []
        while True:
            try:
                poses.append(self.result_queue.get_nowait())
            except Empty:
                break
        return poses
        
    def _processing_loop(self):
        """Main processing loop running in separate thread."""
        while self.is_running:
            try:
                # Synchronize IMU and RGB data
                synchronized_data = self._synchronize_data()
                
                if synchronized_data is not None:
                    # Process synchronized data
                    fused_pose = self._process_synchronized_data(synchronized_data)
                    
                    if fused_pose is not None:
                        # Add to result queue
                        try:
                            self.result_queue.put_nowait(fused_pose)
                        except:
                            # Remove oldest result if queue full
                            try:
                                self.result_queue.get_nowait()
                                self.result_queue.put_nowait(fused_pose)
                            except:
                                pass
                        
                        self.frame_count += 1
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.001)
                
            except Exception as e:
                self.logger.error(f"Error in processing loop: {e}")
                time.sleep(0.01)
                
    def _synchronize_data(self) -> Optional[SensorData]:
        """
        Synchronize IMU and RGB data based on timestamps.
        Returns synchronized data or None if synchronization fails.
        """
        try:
            # Get latest data from both queues without blocking
            imu_data = None
            rgb_data = None
            
            # Try to get IMU data
            try:
                imu_data = self.imu_queue.get_nowait()
            except Empty:
                pass
                
            # Try to get RGB data  
            try:
                rgb_data = self.rgb_queue.get_nowait()
            except Empty:
                pass
                
            # Check if we have both types of data
            if imu_data is None and rgb_data is None:
                return None
                
            # If we only have one type, create partial synchronized data
            if imu_data is not None and rgb_data is None:
                return SensorData(
                    imu_data=imu_data['data'],
                    rgb_frame=None,
                    timestamp=imu_data['timestamp'],
                    frame_id=self.frame_count
                )
            elif rgb_data is not None and imu_data is None:
                return SensorData(
                    imu_data=None,
                    rgb_frame=rgb_data['frame'],
                    timestamp=rgb_data['timestamp'],
                    frame_id=self.frame_count
                )
            else:
                # We have both, check synchronization
                time_diff = abs(imu_data['timestamp'] - rgb_data['timestamp'])
                
                if time_diff <= self.sync_tolerance:
                    # Data is synchronized
                    avg_timestamp = (imu_data['timestamp'] + rgb_data['timestamp']) / 2
                    return SensorData(
                        imu_data=imu_data['data'],
                        rgb_frame=rgb_data['frame'],
                        timestamp=avg_timestamp,
                        frame_id=self.frame_count
                    )
                else:
                    # Use the more recent data
                    if imu_data['timestamp'] > rgb_data['timestamp']:
                        return SensorData(
                            imu_data=imu_data['data'],
                            rgb_frame=None,
                            timestamp=imu_data['timestamp'],
                            frame_id=self.frame_count
                        )
                    else:
                        return SensorData(
                            imu_data=None,
                            rgb_frame=rgb_data['frame'],
                            timestamp=rgb_data['timestamp'],
                            frame_id=self.frame_count
                        )
                        
        except Exception as e:
            self.logger.error(f"Error synchronizing data: {e}")
            return None
            
    def _process_synchronized_data(self, data: SensorData) -> Optional[PoseEstimate]:
        """
        Process synchronized sensor data and return fused pose estimate.
        """
        start_time = time.time()
        
        imu_pose = None
        visual_pose = None
        
        # Process IMU data with MobilePoser
        if data.imu_data is not None:
            imu_pose = self._process_imu_data(data.imu_data, data.timestamp)
            
        # Process RGB frame with SLAM
        if data.rgb_frame is not None:
            visual_pose = self._process_rgb_frame(data.rgb_frame, data.timestamp)
            
        # Fuse poses
        fused_pose = self.fusion_module.fuse_poses(imu_pose, visual_pose)
        
        # Track performance
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        if len(self.processing_times) > 1000:
            self.processing_times = self.processing_times[-1000:]
            
        # Update fusion success rate
        if fused_pose is not None:
            self.fusion_success_rate = 0.95 * self.fusion_success_rate + 0.05 * 1.0
        else:
            self.fusion_success_rate = 0.95 * self.fusion_success_rate + 0.05 * 0.0
            
        return fused_pose
        
    def _process_imu_data(self, imu_data: np.ndarray, timestamp: float) -> Optional[PoseEstimate]:
        """Process IMU data with MobilePoser."""
        try:
            # Convert to torch tensor
            imu_tensor = torch.from_numpy(imu_data).float().unsqueeze(0)
            
            # Forward pass through MobilePoser
            with torch.no_grad():
                pose, joints, translation, contact = self.mobileposer.forward_online(imu_tensor.squeeze(0))
                
            # Convert to numpy
            if isinstance(pose, torch.Tensor):
                pose_np = pose.cpu().numpy()
            else:
                pose_np = pose
                
            if isinstance(translation, torch.Tensor):
                translation_np = translation.cpu().numpy()
            else:
                translation_np = translation
                
            # Extract rotation matrix from pose (assuming pose contains rotation matrices)
            if pose_np.shape == (24, 9):  # 24 joints, 9 values per rotation matrix
                # Use root joint rotation (first joint)
                root_rotation = pose_np[0].reshape(3, 3)
            elif pose_np.shape == (24, 3, 3):
                root_rotation = pose_np[0]
            else:
                # Default to identity if format is unexpected
                root_rotation = np.eye(3)
                
            # Estimate confidence based on contact probabilities and joint consistency
            if isinstance(contact, torch.Tensor):
                contact_np = contact.cpu().numpy()
                confidence = float(np.mean(contact_np))
            else:
                confidence = 0.8  # Default confidence
                
            self.last_imu_pose = PoseEstimate(
                translation=translation_np,
                rotation=root_rotation,
                confidence=confidence,
                timestamp=timestamp,
                source="imu"
            )
            
            return self.last_imu_pose
            
        except Exception as e:
            self.logger.error(f"Error processing IMU data: {e}")
            return None
            
    def _process_rgb_frame(self, rgb_frame: np.ndarray, timestamp: float) -> Optional[PoseEstimate]:
        """Process RGB frame with SLAM."""
        try:
            # Process frame with SLAM interface
            slam_result = self.slam_interface.process_frame(rgb_frame, timestamp)
            
            if slam_result is None:
                return None
                
            # Extract pose information
            pose_matrix = slam_result['pose']  # 4x4 transformation matrix
            confidence = slam_result['confidence']
            
            # Extract translation and rotation
            translation = pose_matrix[:3, 3]
            rotation = pose_matrix[:3, :3]
            
            self.last_visual_pose = PoseEstimate(
                translation=translation,
                rotation=rotation,
                confidence=confidence,
                timestamp=timestamp,
                source="visual"
            )
            
            return self.last_visual_pose
            
        except Exception as e:
            self.logger.error(f"Error processing RGB frame: {e}")
            return None
            
    def _clear_queues(self):
        """Clear all data queues."""
        queues = [self.imu_queue, self.rgb_queue, self.result_queue]
        for queue in queues:
            while not queue.empty():
                try:
                    queue.get_nowait()
                except Empty:
                    break
                    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self.processing_times:
            return {}
            
        return {
            'average_processing_time': np.mean(self.processing_times),
            'max_processing_time': np.max(self.processing_times),
            'min_processing_time': np.min(self.processing_times),
            'fusion_success_rate': self.fusion_success_rate,
            'total_frames_processed': self.frame_count,
            'fps': self.frame_count / (time.time() - self.start_time) if self.start_time else 0.0,
            'queue_sizes': {
                'imu': self.imu_queue.qsize(),
                'rgb': self.rgb_queue.qsize(),
                'results': self.result_queue.qsize()
            }
        }
        
    def __del__(self):
        """Cleanup on destruction."""
        if self.is_running:
            self.stop_processing()
        if hasattr(self, 'slam_interface'):
            self.slam_interface.shutdown()