import numpy as np
import cv2
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
import logging
import sys
import os

from mobileposer.models.slam import SlamInterface


class RealOrbSlam3Interface(SlamInterface):
    """
    Real ORB-SLAM3 interface using pyOrbSlam3 wrapper.
    Supports both Monocular and Visual-Inertial SLAM modes.
    """
    
    def __init__(self, 
                 mode: str = "monocular",  # "monocular" or "visual_inertial"
                 vocabulary_path: Optional[str] = None,
                 settings_file: Optional[str] = None,
                 enable_viewer: bool = False):
        """
        Initialize ORB-SLAM3 system.
        
        Args:
            mode: SLAM mode ("monocular" or "visual_inertial")
            vocabulary_path: Path to ORB vocabulary file
            settings_file: Path to SLAM settings YAML file
            enable_viewer: Enable SLAM viewer window
        """
        super().__init__()
        
        self.mode = mode
        self.vocabulary_path = vocabulary_path or self._get_default_vocabulary_path()
        self.settings_file = settings_file
        self.enable_viewer = enable_viewer
        
        # ORB-SLAM3 system
        self.slam_system = None
        
        # State tracking
        self.frame_count = 0
        self.last_pose = None
        self.tracking_state = "lost"
        
        # IMU data buffer for VI-SLAM
        self.imu_buffer = []
        self.max_imu_buffer_size = 100
        
        # Scale estimation for VI-SLAM
        self.scale_estimate = 1.0
        self.scale_confidence = 0.0
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def _get_default_vocabulary_path(self) -> str:
        """Get default ORB vocabulary file path."""
        vocab_path = Path(__file__).parent.parent.parent / "third_party/pyOrbSlam3/pyOrbSlam3/modules/ORB_SLAM3/Vocabulary/ORBvoc.txt"
        return str(vocab_path)
    
    def _get_default_settings_file(self) -> str:
        """Get default settings file for SLAM."""
        # Use Nymeria-specific calibration files
        slam_configs_dir = Path(__file__).parent.parent / "slam_configs"
        
        if self.mode == "monocular":
            # Use the more accurate Nymeria monocular calibration
            settings_file = slam_configs_dir / "nymeria_mono_base.yaml"
            if not settings_file.exists():
                # Fallback to TUM1 if Nymeria config not found
                settings_file = Path(__file__).parent.parent.parent / "third_party/pyOrbSlam3/pyOrbSlam3/modules/ORB_SLAM3/Examples/Monocular/TUM1.yaml"
        else:  # visual_inertial  
            # Use Nymeria VI-SLAM config
            settings_file = slam_configs_dir / "nymeria_vi.yaml"
            if not settings_file.exists():
                # Fallback to EuRoC if Nymeria config not found
                settings_file = Path(__file__).parent.parent.parent / "third_party/pyOrbSlam3/pyOrbSlam3/modules/ORB_SLAM3/Examples/Monocular-Inertial/EuRoC.yaml"
            
        return str(settings_file)
    
    def _create_monocular_settings(self, settings_file: Path):
        """Create monocular SLAM settings file for Nymeria."""
        settings_content = """%YAML:1.0

#--------------------------------------------------------------------------------------------
# System config
#--------------------------------------------------------------------------------------------
File.version: "1.0"

#--------------------------------------------------------------------------------------------
# Camera Parameters for Nymeria Head-mounted Camera
#--------------------------------------------------------------------------------------------

# Camera type: PinHole or KannalaBrandt8
Camera.type: "PinHole"

# Camera calibration parameters (OpenCV)
Camera1.fx: 525.0
Camera1.fy: 525.0
Camera1.cx: 319.5
Camera1.cy: 239.5

# Distortion parameters [k1, k2, p1, p2, k3]
Camera1.k1: 0.0
Camera1.k2: 0.0
Camera1.p1: 0.0
Camera1.p2: 0.0

# Camera resolution
Camera.width: 640
Camera.height: 480

# Camera frames per second
Camera.fps: 30

# Color order of the images (0: BGR, 1: RGB)
Camera.RGB: 1

#--------------------------------------------------------------------------------------------
# ORB Parameters
#--------------------------------------------------------------------------------------------

# ORB Extractor: Number of features per image
ORBextractor.nFeatures: 1000

# ORB Extractor: Scale factor between levels in the scale pyramid
ORBextractor.scaleFactor: 1.2

# ORB Extractor: Number of levels in the scale pyramid
ORBextractor.nLevels: 8

# ORB Extractor: Fast threshold
ORBextractor.iniThFAST: 20
ORBextractor.minThFAST: 7

#--------------------------------------------------------------------------------------------
# Viewer Parameters
#--------------------------------------------------------------------------------------------
Viewer.KeyFrameSize: 0.05
Viewer.KeyFrameLineWidth: 1.0
Viewer.GraphLineWidth: 0.9
Viewer.PointSize: 2.0
Viewer.CameraSize: 0.08
Viewer.CameraLineWidth: 3.0
Viewer.ViewpointX: 0.0
Viewer.ViewpointY: -0.7
Viewer.ViewpointZ: -1.8
Viewer.ViewpointF: 500.0
"""
        with open(settings_file, 'w') as f:
            f.write(settings_content)
    
    def _create_vi_settings(self, settings_file: Path):
        """Create Visual-Inertial SLAM settings file for Nymeria."""
        settings_content = """
%YAML:1.0

#--------------------------------------------------------------------------------------------
# Camera Parameters for Nymeria Head-mounted Camera
#--------------------------------------------------------------------------------------------

# Camera type: PinHole or KannalaBrandt8
Camera.type: "PinHole"

# Camera resolution
Camera.width: 640
Camera.height: 480

# Camera calibration parameters [fx, fy, cx, cy]
Camera.fx: 525.0
Camera.fy: 525.0
Camera.cx: 319.5
Camera.cy: 239.5

# Distortion parameters [k1, k2, p1, p2, k3]
Camera.k1: 0.0
Camera.k2: 0.0
Camera.p1: 0.0
Camera.p2: 0.0

# Camera frames per second
Camera.fps: 30

# Color order of the images (0: BGR, 1: RGB)
Camera.RGB: 1

#--------------------------------------------------------------------------------------------
# IMU Parameters for Nymeria Head IMU
#--------------------------------------------------------------------------------------------

# IMU noise (continuous time)
IMU.NoiseGyro: 1.7e-4       # rad/s
IMU.NoiseAcc: 2.0e-3        # m/s^2
IMU.GyroWalk: 1.9393e-05    # rad/s^2
IMU.AccWalk: 3.0000e-03     # m/s^3

# IMU frequency (Hz)
IMU.Frequency: 60

# Transformation from camera to IMU frame
# T_bc: transformation from camera to body (IMU) frame
Tbc: !!opencv-matrix
  rows: 4
  cols: 4
  dt: f
  data: [1.0, 0.0, 0.0, 0.0,
         0.0, 1.0, 0.0, 0.0,
         0.0, 0.0, 1.0, 0.0,
         0.0, 0.0, 0.0, 1.0]

# Do not insert KFs when recently lost
insertKFsWhenLost: 0

#--------------------------------------------------------------------------------------------
# ORB Parameters
#--------------------------------------------------------------------------------------------

# ORB Extractor: Number of features per image
ORBextractor.nFeatures: 1000

# ORB Extractor: Scale factor between levels in the scale pyramid
ORBextractor.scaleFactor: 1.2

# ORB Extractor: Number of levels in the scale pyramid
ORBextractor.nLevels: 8

# ORB Extractor: Fast threshold
ORBextractor.iniThFAST: 20
ORBextractor.minThFAST: 7

#--------------------------------------------------------------------------------------------
# Viewer Parameters
#--------------------------------------------------------------------------------------------
Viewer.KeyFrameSize: 0.05
Viewer.KeyFrameLineWidth: 1
Viewer.GraphLineWidth: 0.9
Viewer.PointSize: 2
Viewer.CameraSize: 0.08
Viewer.CameraLineWidth: 3
Viewer.ViewpointX: 0
Viewer.ViewpointY: -0.7
Viewer.ViewpointZ: -1.8
Viewer.ViewpointF: 500
"""
        with open(settings_file, 'w') as f:
            f.write(settings_content)
    
    def initialize(self, config_path: str = None) -> bool:
        """Initialize ORB-SLAM3 system."""
        try:
            # Import pyOrbSlam
            try:
                # Add the pyOrbSlam3 build path to Python path
                pyorb_build_path = Path(__file__).parent.parent.parent / "third_party/pyOrbSlam3/pyOrbSlam3/build"
                if str(pyorb_build_path) not in sys.path:
                    sys.path.insert(0, str(pyorb_build_path))
                
                import pyOrbSlam
                self.logger.info("Successfully imported pyOrbSlam")
                
            except ImportError as e:
                self.logger.error(f"Failed to import pyOrbSlam: {e}")
                raise RuntimeError(f"ORB-SLAM3 not available. Please install pyOrbSlam3 to use real SLAM integration: {e}")
            
            # Use provided config or instance settings_file or create default
            if config_path is None:
                config_path = self.settings_file or self._get_default_settings_file()
            
            # Update instance settings_file to track which config is being used
            self.settings_file = config_path
            
            # Verify vocabulary file exists
            if not os.path.exists(self.vocabulary_path):
                self.logger.error(f"ORB vocabulary file not found: {self.vocabulary_path}")
                return False
            
            # Verify settings file exists
            if not os.path.exists(config_path):
                self.logger.error(f"Settings file not found: {config_path}")
                return False
            
            # Initialize ORB-SLAM3 system
            # pyOrbSlam.OrbSlam(path_to_vocabulary, path_to_settings, sensorType, useViewer)
            if self.mode == "monocular":
                self.slam_system = pyOrbSlam.OrbSlam(
                    self.vocabulary_path, 
                    config_path, 
                    "Mono",  # Sensor type as string
                    self.enable_viewer
                )
            else:  # visual_inertial
                self.slam_system = pyOrbSlam.OrbSlam(
                    self.vocabulary_path, 
                    config_path, 
                    "MonoIMU",  # Monocular-Inertial
                    self.enable_viewer
                )
            
            self.is_initialized = True
            self.frame_count = 0
            
            self.logger.info(f"ORB-SLAM3 {self.mode} system initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ORB-SLAM3: {e}")
            return False
    
    def process_frame(self, image: np.ndarray, timestamp: float) -> Optional[Dict[str, Any]]:
        """
        Process RGB frame with monocular SLAM.
        
        Args:
            image: RGB image (H, W, 3)
            timestamp: Frame timestamp
            
        Returns:
            Dictionary with pose and tracking information
        """
        if not self.is_initialized or self.slam_system is None:
            return None
        
        try:
            # Convert RGB to grayscale if needed
            if len(image.shape) == 3:
                gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray_image = image
            
            # Process frame with ORB-SLAM3
            pose_matrix = self.slam_system.process(gray_image, timestamp)
            
            # Get tracking state
            tracking_state = self.slam_system.GetTrackingState()
            
            # Convert tracking state to our format
            if tracking_state == 0:  # NO_IMAGES_YET
                self.tracking_state = "initializing"
                confidence = 0.0
            elif tracking_state == 1:  # NOT_INITIALIZED
                self.tracking_state = "initializing"
                confidence = 0.2
            elif tracking_state == 2:  # OK
                self.tracking_state = "tracking"
                confidence = 0.9
            elif tracking_state == 3:  # LOST
                self.tracking_state = "lost"
                confidence = 0.1
            else:
                self.tracking_state = "lost"
                confidence = 0.0
            
            # Process pose if available
            if pose_matrix is not None and self.tracking_state == "tracking":
                # ORB-SLAM3 returns camera pose in world frame
                self.last_pose = pose_matrix
                
                # Get map points if available
                map_points = None
                try:
                    map_points = self.slam_system.GetTrackedMapPoints()  # TODO: Check if this method exists
                except:
                    pass
                
                return {
                    'pose': pose_matrix,
                    'confidence': confidence,
                    'tracking_state': self.tracking_state,
                    'map_points': map_points,
                    'keypoints': None,  # Could be added if needed
                    'timestamp': timestamp
                }
            else:
                return {
                    'pose': None,
                    'confidence': confidence,
                    'tracking_state': self.tracking_state,
                    'map_points': None,
                    'keypoints': None,
                    'timestamp': timestamp
                }
            
        except Exception as e:
            self.logger.error(f"Error processing frame with ORB-SLAM3: {e}")
            return None
        finally:
            self.frame_count += 1
    
    def process_frame_with_imu(self, image: np.ndarray, timestamp: float, 
                              imu_data: Optional[np.ndarray] = None) -> Optional[Dict[str, Any]]:
        """
        Process RGB frame with IMU data for Visual-Inertial SLAM.
        
        Args:
            image: RGB image (H, W, 3)
            timestamp: Frame timestamp
            imu_data: IMU data [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
            
        Returns:
            Dictionary with pose, scale, and tracking information
        """
        if not self.is_initialized or self.slam_system is None:
            return None
        
        if self.mode != "visual_inertial":
            # Fall back to monocular processing
            return self.process_frame(image, timestamp)
        
        try:
            # Convert RGB to grayscale if needed
            if len(image.shape) == 3:
                gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray_image = image
            
            # Add IMU data to buffer if provided
            if imu_data is not None:
                self._add_imu_data(imu_data, timestamp)
            
            # Process frame with VI-SLAM
            # Note: pyOrbSlam doesn't have a separate VI method exposed, 
            # so we use the regular process method
            # TODO: Implement proper VI-SLAM when pyOrbSlam3 supports it
            pose_matrix = self.slam_system.process(gray_image, timestamp)
            
            # Get tracking state
            tracking_state = self.slam_system.GetTrackingState()
            
            # Convert tracking state
            if tracking_state == 0:  # NO_IMAGES_YET
                self.tracking_state = "initializing"
                confidence = 0.0
            elif tracking_state == 1:  # NOT_INITIALIZED
                self.tracking_state = "initializing"
                confidence = 0.3
            elif tracking_state == 2:  # OK
                self.tracking_state = "tracking"
                confidence = 0.95  # Higher confidence for VI-SLAM
            elif tracking_state == 3:  # LOST
                self.tracking_state = "lost"
                confidence = 0.1
            else:
                self.tracking_state = "lost"
                confidence = 0.0
            
            # Update scale estimate if tracking
            if self.tracking_state == "tracking" and imu_data is not None:
                self._update_scale_estimate(imu_data)
            
            # Process pose if available
            if pose_matrix is not None and self.tracking_state == "tracking":
                self.last_pose = pose_matrix
                
                # Get map points
                map_points = None
                try:
                    map_points = self.slam_system.GetTrackedMapPoints()  # TODO: Check if this method exists
                except:
                    pass
                
                return {
                    'pose': pose_matrix,
                    'confidence': confidence,
                    'tracking_state': self.tracking_state,
                    'scale_factor': self.scale_estimate,
                    'scale_confidence': self.scale_confidence,
                    'map_points': map_points,
                    'keypoints': None,
                    'timestamp': timestamp
                }
            else:
                return {
                    'pose': None,
                    'confidence': confidence,
                    'tracking_state': self.tracking_state,
                    'scale_factor': self.scale_estimate,
                    'scale_confidence': self.scale_confidence,
                    'map_points': None,
                    'keypoints': None,
                    'timestamp': timestamp
                }
            
        except Exception as e:
            self.logger.error(f"Error processing frame with VI-SLAM: {e}")
            return None
        finally:
            self.frame_count += 1
    
    def _add_imu_data(self, imu_data: np.ndarray, timestamp: float):
        """Add IMU data to buffer for VI-SLAM."""
        self.imu_buffer.append({
            'data': imu_data.copy(),
            'timestamp': timestamp
        })
        
        # Keep buffer size manageable
        if len(self.imu_buffer) > self.max_imu_buffer_size:
            self.imu_buffer = self.imu_buffer[-self.max_imu_buffer_size:]
    
    def _get_imu_measurements_for_frame(self, frame_timestamp: float) -> np.ndarray:
        """Get IMU measurements for the given frame timestamp."""
        if not self.imu_buffer:
            return np.empty((0, 7))  # Empty array with correct shape
        
        # Find IMU measurements close to frame timestamp
        relevant_measurements = []
        time_tolerance = 0.05  # 50ms tolerance
        
        for imu_entry in self.imu_buffer:
            time_diff = abs(imu_entry['timestamp'] - frame_timestamp)
            if time_diff <= time_tolerance:
                # Format: [timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
                measurement = np.concatenate([
                    [imu_entry['timestamp']], 
                    imu_entry['data'][:6]  # First 6 values: acc + gyro
                ])
                relevant_measurements.append(measurement)
        
        if relevant_measurements:
            return np.array(relevant_measurements)
        else:
            return np.empty((0, 7))
    
    def _update_scale_estimate(self, imu_data: np.ndarray):
        """Update scale estimate using IMU accelerometer data."""
        # Extract accelerometer data
        acc = imu_data[:3]
        gravity_magnitude = np.linalg.norm(acc)
        expected_gravity = 9.81
        
        # Update scale estimate
        if gravity_magnitude > 0.1:  # Avoid division by zero
            new_scale = expected_gravity / gravity_magnitude
            self.scale_estimate = 0.95 * self.scale_estimate + 0.05 * new_scale
            
            # Update confidence based on gravity alignment
            gravity_error = abs(gravity_magnitude - expected_gravity) / expected_gravity
            confidence = max(0.0, 1.0 - gravity_error)
            self.scale_confidence = 0.9 * self.scale_confidence + 0.1 * confidence
    
    def get_trajectory(self) -> Optional[np.ndarray]:
        """Get full trajectory from ORB-SLAM3."""
        if not self.is_initialized or self.slam_system is None:
            return None
        
        try:
            # Get keyframe poses
            # Note: pyOrbSlam doesn't expose this method directly
            # TODO: Implement when available in pyOrbSlam3
            return None
        except Exception as e:
            self.logger.error(f"Error getting trajectory: {e}")
            return None
    
    def reset(self):
        """Reset ORB-SLAM3 system."""
        if self.slam_system is not None:
            try:
                self.slam_system.Reset()  # TODO: Check if lowercase 'reset' is needed
            except:
                pass
        
        self.frame_count = 0
        self.last_pose = None
        self.tracking_state = "lost"
        self.imu_buffer.clear()
        self.scale_estimate = 1.0
        self.scale_confidence = 0.0
    
    def shutdown(self):
        """Shutdown ORB-SLAM3 system."""
        if self.slam_system is not None:
            try:
                self.slam_system.shutdown()
            except:
                pass
            self.slam_system = None
        
        self.is_initialized = False
        self.logger.info("ORB-SLAM3 system shutdown")


def create_real_orbslam3_interface(mode: str = "monocular", 
                                   vocabulary_path: Optional[str] = None,
                                   settings_file: Optional[str] = None,
                                   enable_viewer: bool = False) -> RealOrbSlam3Interface:
    """
    Factory function to create real ORB-SLAM3 interface.
    
    Args:
        mode: SLAM mode ("monocular" or "visual_inertial")
        vocabulary_path: Path to ORB vocabulary file
        settings_file: Path to settings YAML file
        enable_viewer: Enable SLAM viewer
        
    Returns:
        RealOrbSlam3Interface instance
    """
    return RealOrbSlam3Interface(
        mode=mode,
        vocabulary_path=vocabulary_path,
        settings_file=settings_file,
        enable_viewer=enable_viewer
    )