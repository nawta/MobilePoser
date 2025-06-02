import numpy as np
import torch
import cv2
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

from mobileposer.models.slam import SlamInterface, MockSlamInterface


class SlamMode(Enum):
    """Enumeration of different SLAM modes."""
    NONE = "none"                    # No SLAM, IMU only
    MONOCULAR = "monocular"          # Monocular visual SLAM
    VISUAL_INERTIAL = "visual_inertial"  # Visual-Inertial SLAM


@dataclass
class SlamInput:
    """Container for SLAM input data."""
    rgb_frame: Optional[np.ndarray] = None      # RGB image
    head_imu_data: Optional[np.ndarray] = None  # Head IMU data [acc, gyro]
    timestamp: float = 0.0
    frame_id: int = 0


@dataclass 
class SlamOutput:
    """Container for SLAM output data."""
    pose: Optional[np.ndarray] = None           # 4x4 transformation matrix
    confidence: float = 0.0                     # Confidence score [0, 1]
    scale_factor: float = 1.0                   # Scale factor (for VI-SLAM)
    scale_confidence: float = 0.0               # Scale estimation confidence
    tracking_state: str = "lost"                # "tracking", "lost", "initializing"
    keypoints: Optional[np.ndarray] = None      # Detected keypoints
    map_points: Optional[np.ndarray] = None     # Map points
    timestamp: float = 0.0
    mode_used: SlamMode = SlamMode.NONE


class AdaptiveSlamInterface:
    """
    Adaptive SLAM interface that automatically selects the appropriate
    SLAM mode based on available input data:
    - RGB + Head IMU -> Visual-Inertial SLAM
    - RGB only -> Monocular SLAM  
    - No RGB -> No SLAM (IMU-only mode)
    """
    
    def __init__(self, 
                 orb_vocabulary_path: Optional[str] = None,
                 camera_config: Optional[Dict[str, Any]] = None,
                 enable_viewer: bool = False):
        """
        Initialize adaptive SLAM interface.
        
        Args:
            orb_vocabulary_path: Path to ORB vocabulary file
            camera_config: Camera calibration parameters
            enable_viewer: Enable SLAM viewer window
        """
        self.orb_vocabulary_path = orb_vocabulary_path
        self.camera_config = camera_config or self._default_camera_config()
        self.enable_viewer = enable_viewer
        
        # SLAM system instances
        self.monocular_slam = None
        self.visual_inertial_slam = None
        
        # Current state
        self.current_mode = SlamMode.NONE
        self.is_initialized = False
        self.frame_count = 0
        
        # Performance tracking
        self.mode_switches = 0
        self.processing_times = []
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def initialize(self) -> bool:
        """Initialize SLAM systems."""
        try:
            # Initialize both SLAM modes
            self.monocular_slam = self._create_monocular_slam()
            self.visual_inertial_slam = self._create_visual_inertial_slam()
            
            if self.monocular_slam is None or self.visual_inertial_slam is None:
                self.logger.error("Failed to initialize ORB-SLAM3")
                raise RuntimeError("Failed to initialize SLAM systems. Please ensure pyOrbSlam3 is installed.")
                
            # Initialize both systems
            mono_init = self.monocular_slam.initialize()
            vi_init = self.visual_inertial_slam.initialize()
            
            self.is_initialized = mono_init and vi_init
            return self.is_initialized
            
        except Exception as e:
            self.logger.error(f"Failed to initialize adaptive SLAM: {e}")
            return False
    
    def process_frame(self, slam_input: SlamInput) -> SlamOutput:
        """
        Process frame with automatic mode selection.
        
        Args:
            slam_input: Input data container
            
        Returns:
            SLAM output with pose estimate and metadata
        """
        if not self.is_initialized:
            return SlamOutput(timestamp=slam_input.timestamp)
        
        # Determine appropriate SLAM mode
        selected_mode = self._select_slam_mode(slam_input)
        
        # Handle mode switches
        if selected_mode != self.current_mode:
            self._handle_mode_switch(self.current_mode, selected_mode)
            self.current_mode = selected_mode
            self.mode_switches += 1
        
        # Process with selected mode
        slam_output = self._process_with_mode(slam_input, selected_mode)
        slam_output.mode_used = selected_mode
        slam_output.timestamp = slam_input.timestamp
        
        self.frame_count += 1
        return slam_output
    
    def _select_slam_mode(self, slam_input: SlamInput) -> SlamMode:
        """
        Select appropriate SLAM mode based on available input data.
        
        Args:
            slam_input: Input data container
            
        Returns:
            Selected SLAM mode
        """
        has_rgb = slam_input.rgb_frame is not None
        has_head_imu = slam_input.head_imu_data is not None
        
        if not has_rgb:
            return SlamMode.NONE
        elif has_rgb and has_head_imu:
            return SlamMode.VISUAL_INERTIAL
        else:  # has_rgb and not has_head_imu
            return SlamMode.MONOCULAR
    
    def _process_with_mode(self, slam_input: SlamInput, mode: SlamMode) -> SlamOutput:
        """
        Process input with specified SLAM mode.
        
        Args:
            slam_input: Input data
            mode: SLAM mode to use
            
        Returns:
            SLAM output
        """
        if mode == SlamMode.NONE:
            return SlamOutput(
                tracking_state="none",
                confidence=0.0,
                timestamp=slam_input.timestamp
            )
        
        elif mode == SlamMode.MONOCULAR:
            return self._process_monocular(slam_input)
        
        elif mode == SlamMode.VISUAL_INERTIAL:
            return self._process_visual_inertial(slam_input)
        
        else:
            self.logger.error(f"Unknown SLAM mode: {mode}")
            return SlamOutput(timestamp=slam_input.timestamp)
    
    def _process_monocular(self, slam_input: SlamInput) -> SlamOutput:
        """Process with monocular SLAM."""
        try:
            result = self.monocular_slam.process_frame(
                slam_input.rgb_frame, 
                slam_input.timestamp
            )
            
            if result is None:
                return SlamOutput(
                    tracking_state="lost",
                    timestamp=slam_input.timestamp
                )
            
            return SlamOutput(
                pose=result['pose'],
                confidence=result['confidence'],
                scale_factor=1.0,  # Monocular SLAM has scale ambiguity
                scale_confidence=0.0,
                tracking_state="tracking" if result['confidence'] > 0.5 else "lost",
                keypoints=result.get('keypoints'),
                map_points=result.get('map_points'),
                timestamp=slam_input.timestamp
            )
            
        except Exception as e:
            self.logger.error(f"Error in monocular SLAM: {e}")
            return SlamOutput(
                tracking_state="lost",
                timestamp=slam_input.timestamp
            )
    
    def _process_visual_inertial(self, slam_input: SlamInput) -> SlamOutput:
        """Process with visual-inertial SLAM."""
        try:
            # For VI-SLAM, we need both RGB and IMU data
            if hasattr(self.visual_inertial_slam, 'process_frame_with_imu'):
                result = self.visual_inertial_slam.process_frame_with_imu(
                    slam_input.rgb_frame,
                    slam_input.timestamp,
                    slam_input.head_imu_data
                )
            else:
                # Fallback to standard process_frame
                result = self.visual_inertial_slam.process_frame(
                    slam_input.rgb_frame,
                    slam_input.timestamp
                )
            
            if result is None:
                return SlamOutput(
                    tracking_state="lost",
                    timestamp=slam_input.timestamp
                )
            
            return SlamOutput(
                pose=result['pose'],
                confidence=result['confidence'],
                scale_factor=result.get('scale_factor', 1.0),
                scale_confidence=result.get('scale_confidence', 0.0),
                tracking_state="tracking" if result['confidence'] > 0.5 else "lost",
                keypoints=result.get('keypoints'),
                map_points=result.get('map_points'),
                timestamp=slam_input.timestamp
            )
            
        except Exception as e:
            self.logger.error(f"Error in visual-inertial SLAM: {e}")
            return SlamOutput(
                tracking_state="lost",
                timestamp=slam_input.timestamp
            )
    
    def _handle_mode_switch(self, old_mode: SlamMode, new_mode: SlamMode):
        """
        Handle transitions between different SLAM modes.
        
        Args:
            old_mode: Previous SLAM mode
            new_mode: New SLAM mode
        """
        self.logger.info(f"SLAM mode switch: {old_mode.value} -> {new_mode.value}")
        
        # For mode switches, we might want to:
        # 1. Reset tracking state
        # 2. Transfer map information if possible
        # 3. Adjust confidence estimates
        
        if old_mode == SlamMode.NONE and new_mode != SlamMode.NONE:
            self.logger.info("Enabling SLAM tracking")
            
        elif old_mode != SlamMode.NONE and new_mode == SlamMode.NONE:
            self.logger.info("Disabling SLAM tracking")
            
        elif old_mode == SlamMode.MONOCULAR and new_mode == SlamMode.VISUAL_INERTIAL:
            self.logger.info("Upgrading to Visual-Inertial SLAM")
            
        elif old_mode == SlamMode.VISUAL_INERTIAL and new_mode == SlamMode.MONOCULAR:
            self.logger.info("Downgrading to Monocular SLAM")
    
    def _create_monocular_slam(self) -> Optional[SlamInterface]:
        """Create monocular SLAM instance."""
        try:
            # Try to create actual ORB-SLAM3 monocular instance
            from mobileposer.models.real_orbslam3 import create_real_orbslam3_interface
            
            slam_interface = create_real_orbslam3_interface(
                mode="monocular",
                vocabulary_path=self.orb_vocabulary_path,
                enable_viewer=self.enable_viewer
            )
            
            if slam_interface.initialize():
                self.logger.info("Real ORB-SLAM3 monocular system created successfully")
                return slam_interface
            else:
                self.logger.error("Failed to initialize real ORB-SLAM3 monocular system")
                raise RuntimeError("Failed to initialize ORB-SLAM3 monocular system")
            
        except Exception as e:
            self.logger.error(f"Failed to create monocular SLAM: {e}")
            raise RuntimeError(f"Failed to create monocular SLAM system: {e}")
    
    def _create_visual_inertial_slam(self) -> Optional[SlamInterface]:
        """Create visual-inertial SLAM instance."""
        try:
            # Try to create actual ORB-SLAM3 VI instance
            from mobileposer.models.real_orbslam3 import create_real_orbslam3_interface
            
            slam_interface = create_real_orbslam3_interface(
                mode="visual_inertial",
                vocabulary_path=self.orb_vocabulary_path,
                enable_viewer=self.enable_viewer
            )
            
            if slam_interface.initialize():
                self.logger.info("Real ORB-SLAM3 visual-inertial system created successfully")
                return slam_interface
            else:
                self.logger.error("Failed to initialize real ORB-SLAM3 visual-inertial system")
                raise RuntimeError("Failed to initialize ORB-SLAM3 visual-inertial system")
            
        except Exception as e:
            self.logger.error(f"Failed to create visual-inertial SLAM: {e}")
            raise RuntimeError(f"Failed to create visual-inertial SLAM system: {e}")
    
    def _default_camera_config(self) -> Dict[str, Any]:
        """Default camera configuration for Nymeria dataset."""
        return {
            "Camera.type": "PinHole",
            "Camera.fx": 525.0,
            "Camera.fy": 525.0,
            "Camera.cx": 319.5,
            "Camera.cy": 239.5,
            "Camera.k1": 0.0,
            "Camera.k2": 0.0,
            "Camera.p1": 0.0,
            "Camera.p2": 0.0,
            "Camera.fps": 30.0,
            "Camera.RGB": 1,
            
            "ORBextractor.nFeatures": 1000,
            "ORBextractor.scaleFactor": 1.2,
            "ORBextractor.nLevels": 8,
            "ORBextractor.iniThFAST": 20,
            "ORBextractor.minThFAST": 7,
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            'current_mode': self.current_mode.value,
            'mode_switches': self.mode_switches,
            'frames_processed': self.frame_count,
            'average_processing_time': np.mean(self.processing_times) if self.processing_times else 0.0,
            'slam_initialized': self.is_initialized
        }
    
    def reset(self):
        """Reset SLAM state."""
        if self.monocular_slam:
            self.monocular_slam.reset()
        if self.visual_inertial_slam:
            self.visual_inertial_slam.reset()
            
        self.current_mode = SlamMode.NONE
        self.frame_count = 0
        self.mode_switches = 0
        self.processing_times.clear()
    
    def shutdown(self):
        """Shutdown SLAM systems."""
        if self.monocular_slam:
            self.monocular_slam.shutdown()
        if self.visual_inertial_slam:
            self.visual_inertial_slam.shutdown()
            
        self.is_initialized = False


class EnsembleWeightCalculator:
    """
    Module to calculate optimal ensemble weight ratios between
    IMU and SLAM pose estimates based on various factors.
    """
    
    def __init__(self, 
                 confidence_weight: float = 0.4,
                 tracking_weight: float = 0.3,
                 temporal_weight: float = 0.2,
                 scale_weight: float = 0.1):
        """
        Initialize weight calculator.
        
        Args:
            confidence_weight: Weight for confidence-based calculation
            tracking_weight: Weight for tracking state
            temporal_weight: Weight for temporal consistency
            scale_weight: Weight for scale estimation quality
        """
        self.confidence_weight = confidence_weight
        self.tracking_weight = tracking_weight  
        self.temporal_weight = temporal_weight
        self.scale_weight = scale_weight
        
        # History for temporal analysis
        self.slam_confidence_history = []
        self.imu_confidence_history = []
        self.max_history_size = 30
        
    def calculate_weights(self,
                         imu_confidence: float,
                         slam_output: SlamOutput,
                         imu_pose_available: bool = True) -> Tuple[float, float]:
        """
        Calculate ensemble weights for IMU and SLAM pose estimates.
        
        Args:
            imu_confidence: Confidence of IMU-based pose estimate
            slam_output: SLAM output with pose and metadata
            imu_pose_available: Whether IMU pose is available
            
        Returns:
            Tuple of (imu_weight, slam_weight) summing to 1.0
        """
        if not imu_pose_available:
            return (0.0, 1.0)
        
        if slam_output.tracking_state == "lost" or slam_output.pose is None:
            return (1.0, 0.0)
        
        # Update confidence history
        self._update_confidence_history(imu_confidence, slam_output.confidence)
        
        # Calculate individual weight components
        confidence_component = self._calculate_confidence_component(
            imu_confidence, slam_output.confidence
        )
        
        tracking_component = self._calculate_tracking_component(slam_output)
        
        temporal_component = self._calculate_temporal_component()
        
        scale_component = self._calculate_scale_component(slam_output)
        
        # Weighted combination
        slam_score = (
            self.confidence_weight * confidence_component +
            self.tracking_weight * tracking_component +
            self.temporal_weight * temporal_component +
            self.scale_weight * scale_component
        )
        
        # Convert to weights (bias towards IMU for orientation, SLAM for translation)
        # This is a base score that gets modified by modality strengths
        base_slam_weight = np.clip(slam_score, 0.0, 1.0)
        base_imu_weight = 1.0 - base_slam_weight
        
        # Apply modality-specific biases
        # For translation: favor SLAM when available and confident
        translation_slam_weight = min(base_slam_weight * 1.5, 1.0)
        translation_imu_weight = 1.0 - translation_slam_weight
        
        # For orientation: favor IMU (better short-term accuracy)
        orientation_imu_weight = min(base_imu_weight * 1.3, 1.0)
        orientation_slam_weight = 1.0 - orientation_imu_weight
        
        # Return average weights (can be specialized per component in future)
        final_slam_weight = (translation_slam_weight + orientation_slam_weight) / 2
        final_imu_weight = 1.0 - final_slam_weight
        
        return (final_imu_weight, final_slam_weight)
    
    def _calculate_confidence_component(self, imu_conf: float, slam_conf: float) -> float:
        """Calculate weight component based on confidence scores."""
        if slam_conf <= 0.1:
            return 0.0
        
        # Relative confidence difference
        conf_ratio = slam_conf / max(imu_conf, 0.1)
        return np.clip(conf_ratio - 1.0, 0.0, 1.0)
    
    def _calculate_tracking_component(self, slam_output: SlamOutput) -> float:
        """Calculate weight component based on tracking state."""
        if slam_output.tracking_state == "tracking":
            return 1.0
        elif slam_output.tracking_state == "initializing":
            return 0.3
        else:  # lost
            return 0.0
    
    def _calculate_temporal_component(self) -> float:
        """Calculate weight component based on temporal consistency."""
        if len(self.slam_confidence_history) < 5:
            return 0.5  # Neutral when insufficient history
        
        # Analyze SLAM confidence stability
        recent_slam_confs = self.slam_confidence_history[-10:]
        slam_std = np.std(recent_slam_confs)
        
        # Lower standard deviation indicates more consistent tracking
        consistency_score = max(0.0, 1.0 - slam_std * 2.0)
        return consistency_score
    
    def _calculate_scale_component(self, slam_output: SlamOutput) -> float:
        """Calculate weight component based on scale estimation quality."""
        if slam_output.mode_used == SlamMode.VISUAL_INERTIAL:
            # VI-SLAM provides scale, weight it by scale confidence
            return slam_output.scale_confidence
        elif slam_output.mode_used == SlamMode.MONOCULAR:
            # Monocular SLAM has scale ambiguity, reduce weight
            return 0.3
        else:
            return 0.0
    
    def _update_confidence_history(self, imu_conf: float, slam_conf: float):
        """Update confidence history for temporal analysis."""
        self.imu_confidence_history.append(imu_conf)
        self.slam_confidence_history.append(slam_conf)
        
        # Maintain history size
        if len(self.imu_confidence_history) > self.max_history_size:
            self.imu_confidence_history = self.imu_confidence_history[-self.max_history_size:]
            
        if len(self.slam_confidence_history) > self.max_history_size:
            self.slam_confidence_history = self.slam_confidence_history[-self.max_history_size:]
    
    def reset(self):
        """Reset weight calculator state."""
        self.imu_confidence_history.clear()
        self.slam_confidence_history.clear()
    
    def get_weight_breakdown(self, 
                           imu_confidence: float,
                           slam_output: SlamOutput) -> Dict[str, float]:
        """Get detailed breakdown of weight calculation components."""
        if slam_output.tracking_state == "lost":
            return {
                'confidence_component': 0.0,
                'tracking_component': 0.0,
                'temporal_component': 0.0,
                'scale_component': 0.0,
                'final_slam_weight': 0.0,
                'final_imu_weight': 1.0
            }
        
        confidence_comp = self._calculate_confidence_component(imu_confidence, slam_output.confidence)
        tracking_comp = self._calculate_tracking_component(slam_output)
        temporal_comp = self._calculate_temporal_component()
        scale_comp = self._calculate_scale_component(slam_output)
        
        imu_weight, slam_weight = self.calculate_weights(imu_confidence, slam_output)
        
        return {
            'confidence_component': confidence_comp,
            'tracking_component': tracking_comp,
            'temporal_component': temporal_comp,
            'scale_component': scale_comp,
            'final_slam_weight': slam_weight,
            'final_imu_weight': imu_weight
        }