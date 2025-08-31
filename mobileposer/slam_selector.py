"""
SLAM selector module that provides graceful fallback between real and mock SLAM implementations.

This module allows explicit control over SLAM type selection while providing
clear error messages when real SLAM is requested but not available.
"""

import logging
from typing import Optional, Union
from pathlib import Path

from mobileposer.models.slam import SlamInterface, MockSlamInterface
from mobileposer.models.real_orbslam3 import RealOrbSlam3Interface
from mobileposer.models.adaptive_slam import AdaptiveSlamInterface


class SlamSelector:
    """
    Utility class for selecting appropriate SLAM implementation based on
    availability and user preference.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._check_orbslam3_availability()
    
    def _check_orbslam3_availability(self):
        """Check if ORB-SLAM3 is available."""
        self.orbslam3_available = False
        try:
            import sys
            pyorb_path = Path(__file__).parent.parent / "third_party/pyOrbSlam3/pyOrbSlam3"
            if str(pyorb_path) not in sys.path:
                sys.path.insert(0, str(pyorb_path))
            
            import pyOrbSlam
            self.orbslam3_available = True
            self.logger.info("ORB-SLAM3 is available")
        except ImportError:
            self.logger.warning("ORB-SLAM3 not available. Install pyOrbSlam3 to use real SLAM.")
    
    def create_slam(self, 
                   slam_type: str,
                   mode: str = "monocular",
                   allow_mock: bool = False,
                   **kwargs) -> SlamInterface:
        """
        Create SLAM interface with explicit control over mock fallback.
        
        Args:
            slam_type: Type of SLAM ("real", "orb_slam3", "adaptive", "mock")
            mode: SLAM mode ("monocular", "visual_inertial")
            allow_mock: Whether to allow fallback to mock if real SLAM unavailable
            **kwargs: Additional arguments for SLAM initialization
            
        Returns:
            SLAM interface instance
            
        Raises:
            RuntimeError: If real SLAM requested but not available and mock not allowed
        """
        # Handle mock request
        if slam_type == "mock":
            self.logger.info("Creating mock SLAM interface as requested")
            return self._create_mock_slam(mode)
        
        # Handle real SLAM request
        if slam_type in ["real", "orb_slam3"]:
            if self.orbslam3_available:
                self.logger.info(f"Creating real ORB-SLAM3 interface in {mode} mode")
                return self._create_real_slam(mode, **kwargs)
            elif allow_mock:
                self.logger.warning(f"ORB-SLAM3 not available, falling back to mock {mode} SLAM")
                return self._create_mock_slam(mode)
            else:
                raise RuntimeError(
                    "Real ORB-SLAM3 requested but not available. "
                    "Please install pyOrbSlam3 following docs/orbslam3_setup.md"
                )
        
        # Handle adaptive SLAM request
        if slam_type == "adaptive":
            if self.orbslam3_available:
                self.logger.info("Creating adaptive SLAM interface with real ORB-SLAM3")
                return self._create_adaptive_slam(**kwargs)
            elif allow_mock:
                self.logger.warning("ORB-SLAM3 not available, adaptive SLAM will use mock implementations")
                return self._create_adaptive_slam_mock(**kwargs)
            else:
                raise RuntimeError(
                    "Adaptive SLAM requested but ORB-SLAM3 not available. "
                    "Please install pyOrbSlam3 following docs/orbslam3_setup.md"
                )
        
        raise ValueError(f"Unknown SLAM type: {slam_type}")
    
    def _create_mock_slam(self, mode: str) -> SlamInterface:
        """Create mock SLAM interface."""
        if mode == "visual_inertial":
            from mobileposer.head_pose_ensemble import MockVisualInertialSlam
            slam = MockVisualInertialSlam()
        else:
            slam = MockSlamInterface()
        
        slam.initialize()
        return slam
    
    def _create_real_slam(self, mode: str, **kwargs) -> RealOrbSlam3Interface:
        """Create real ORB-SLAM3 interface."""
        slam = RealOrbSlam3Interface(mode=mode, **kwargs)
        
        if not slam.initialize():
            raise RuntimeError(f"Failed to initialize ORB-SLAM3 in {mode} mode")
        
        return slam
    
    def _create_adaptive_slam(self, **kwargs) -> AdaptiveSlamInterface:
        """Create adaptive SLAM interface with real implementations."""
        slam = AdaptiveSlamInterface(**kwargs)
        
        if not slam.initialize():
            raise RuntimeError("Failed to initialize adaptive SLAM")
        
        return slam
    
    def _create_adaptive_slam_mock(self, **kwargs) -> AdaptiveSlamInterface:
        """Create adaptive SLAM with mock implementations (for testing)."""
        # This would need custom implementation to use mock backends
        # For now, raise error
        raise NotImplementedError(
            "Adaptive SLAM with mock backends not yet implemented. "
            "Use slam_type='mock' for testing without ORB-SLAM3."
        )
    
    @property
    def available_types(self) -> list:
        """Get list of available SLAM types."""
        types = ["mock"]
        if self.orbslam3_available:
            types.extend(["real", "orb_slam3", "adaptive"])
        return types
    
    def print_status(self):
        """Print SLAM availability status."""
        print("=" * 60)
        print("SLAM Integration Status")
        print("=" * 60)
        print(f"ORB-SLAM3 Available: {self.orbslam3_available}")
        print(f"Available SLAM Types: {', '.join(self.available_types)}")
        
        if not self.orbslam3_available:
            print("\nTo enable real SLAM:")
            print("1. Follow instructions in docs/orbslam3_setup.md")
            print("2. Install pyOrbSlam3 Python wrapper")
            print("3. Restart MobilePoser")
        print("=" * 60)


# Global instance for convenience
slam_selector = SlamSelector()


def create_slam_with_fallback(slam_type: str, **kwargs) -> SlamInterface:
    """
    Convenience function to create SLAM with automatic mock fallback.
    
    This is useful for demos and testing where you want to run even
    without ORB-SLAM3 installed.
    """
    return slam_selector.create_slam(slam_type, allow_mock=True, **kwargs)


def create_slam_strict(slam_type: str, **kwargs) -> SlamInterface:
    """
    Convenience function to create SLAM without mock fallback.
    
    This is useful for production where you want to ensure real SLAM
    is being used.
    """
    return slam_selector.create_slam(slam_type, allow_mock=False, **kwargs)