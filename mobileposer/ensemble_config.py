"""
Configuration settings for the ensemble pose estimation system.
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class EnsembleConfig:
    """Configuration for ensemble pose estimation system."""
    
    # Model paths
    mobileposer_weights: str = "mobileposer/checkpoints/47/model_finetuned.pth"
    slam_config_path: Optional[str] = None  # Path to ORB-SLAM3 config file
    
    # SLAM settings
    slam_type: str = "mock"  # "mock", "orb_slam3"
    slam_vocabulary_path: Optional[str] = None  # Path to ORB vocabulary
    
    # Fusion settings
    fusion_method: str = "weighted_average"  # "weighted_average", "kalman", "confidence_based"
    fusion_confidence_threshold: float = 0.5
    fusion_temporal_window: float = 1.0  # seconds
    
    # Synchronization settings
    sync_tolerance: float = 0.05  # seconds
    max_queue_size: int = 100
    
    # Processing settings
    max_processing_threads: int = 2
    enable_visualization: bool = False
    save_trajectory: bool = True
    
    # Performance settings
    target_fps: float = 30.0
    max_processing_latency: float = 0.1  # seconds
    
    # Data paths
    nymeria_rgb_data_path: str = "/mnt/nas2/naoto/nymeria_dataset/data_video_main_rgb/"
    output_directory: str = "outputs/ensemble_results/"
    
    # IMU configuration
    imu_sampling_rate: float = 60.0  # Hz
    num_imu_sensors: int = 6
    imu_data_format: str = "acc_rotmat"  # "acc_rotmat", "acc_quat"
    
    # Visual processing
    image_downscale_factor: float = 1.0  # Downscale images for faster processing
    enable_feature_tracking: bool = True
    min_feature_matches: int = 50
    
    # Debugging and logging
    enable_debug_output: bool = False
    log_level: str = "INFO"  # "DEBUG", "INFO", "WARNING", "ERROR"
    save_debug_images: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'EnsembleConfig':
        """Create config from dictionary."""
        return cls(**config_dict)
    
    def save_to_file(self, file_path: str):
        """Save configuration to file."""
        import json
        config_dict = self.to_dict()
        
        # Convert Path objects to strings for JSON serialization
        for key, value in config_dict.items():
            if isinstance(value, Path):
                config_dict[key] = str(value)
        
        with open(file_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load_from_file(cls, file_path: str) -> 'EnsembleConfig':
        """Load configuration from file."""
        import json
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


# Default configurations for different use cases
class DefaultConfigs:
    """Default configurations for common use cases."""
    
    @staticmethod
    def development_config() -> EnsembleConfig:
        """Configuration for development and testing."""
        return EnsembleConfig(
            slam_type="mock",
            fusion_method="weighted_average",
            enable_debug_output=True,
            save_debug_images=True,
            max_queue_size=50,
            target_fps=10.0  # Lower FPS for easier debugging
        )
    
    @staticmethod
    def production_config() -> EnsembleConfig:
        """Configuration for production use."""
        return EnsembleConfig(
            slam_type="orb_slam3",
            fusion_method="kalman",
            enable_debug_output=False,
            save_debug_images=False,
            max_queue_size=200,
            target_fps=30.0
        )
    
    @staticmethod
    def high_accuracy_config() -> EnsembleConfig:
        """Configuration optimized for accuracy."""
        return EnsembleConfig(
            slam_type="orb_slam3",
            fusion_method="kalman",
            fusion_confidence_threshold=0.7,
            fusion_temporal_window=2.0,
            sync_tolerance=0.02,  # Tighter synchronization
            min_feature_matches=100,
            target_fps=20.0  # Lower FPS for more accurate processing
        )
    
    @staticmethod
    def real_time_config() -> EnsembleConfig:
        """Configuration optimized for real-time performance."""
        return EnsembleConfig(
            slam_type="mock",  # Use mock for fastest processing
            fusion_method="confidence_based",
            fusion_confidence_threshold=0.3,
            sync_tolerance=0.1,  # More relaxed synchronization
            image_downscale_factor=0.5,  # Downscale images for speed
            target_fps=60.0,
            max_processing_latency=0.05
        )


# ORB-SLAM3 specific configurations
class OrbSlamConfig:
    """ORB-SLAM3 specific configuration templates."""
    
    @staticmethod
    def monocular_config() -> Dict[str, Any]:
        """Configuration for monocular ORB-SLAM3."""
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
            
            "Viewer.KeyFrameSize": 0.05,
            "Viewer.KeyFrameLineWidth": 1,
            "Viewer.GraphLineWidth": 0.9,
            "Viewer.PointSize": 2,
            "Viewer.CameraSize": 0.08,
            "Viewer.CameraLineWidth": 3,
            "Viewer.ViewpointX": 0,
            "Viewer.ViewpointY": -0.7,
            "Viewer.ViewpointZ": -1.8,
            "Viewer.ViewpointF": 500
        }
    
    @staticmethod
    def save_config_file(config: Dict[str, Any], file_path: str):
        """Save ORB-SLAM3 configuration to YAML file."""
        import yaml
        with open(file_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)


# Sensor calibration data
class SensorCalibration:
    """Sensor calibration parameters."""
    
    @staticmethod
    def nymeria_aria_imu_config() -> Dict[str, Any]:
        """IMU configuration for Nymeria Aria dataset."""
        return {
            "imu_sensors": {
                "head": {"index": 4, "position": [0.0, 0.0, 0.0]},
                "left_wrist": {"index": 0, "position": [-0.3, 0.0, 0.0]},
                "right_wrist": {"index": 1, "position": [0.3, 0.0, 0.0]}
            },
            "sampling_rate": 60.0,
            "gravity": [0.0, -9.81, 0.0],
            "noise_std": {
                "accelerometer": 0.1,
                "gyroscope": 0.01
            }
        }
    
    @staticmethod
    def nymeria_xsens_imu_config() -> Dict[str, Any]:
        """IMU configuration for Nymeria XSens dataset."""
        return {
            "imu_sensors": {
                "head": {"index": 4, "position": [0.0, 0.0, 0.0]},
                "left_wrist": {"index": 0, "position": [-0.3, 0.0, 0.0]},
                "right_wrist": {"index": 1, "position": [0.3, 0.0, 0.0]},
                "left_foot": {"index": 2, "position": [-0.15, -1.0, 0.0]},
                "right_foot": {"index": 3, "position": [0.15, -1.0, 0.0]}
            },
            "sampling_rate": 60.0,
            "gravity": [0.0, -9.81, 0.0],
            "noise_std": {
                "accelerometer": 0.05,
                "gyroscope": 0.005
            }
        }