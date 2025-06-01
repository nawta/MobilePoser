#!/usr/bin/env python3
"""
Comprehensive test script for ORB-SLAM3 with Nymeria sequence.
Includes proper video preprocessing and trajectory comparison.
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

# Setup environment for pyOrbSlam3
def setup_environment():
    """Set up environment for pyOrbSlam3."""
    lib_paths = [
        '/usr/local/lib',
        '/usr/lib/x86_64-linux-gnu',
        '/home/naoto/docker_workspace/MobilePoser/third_party/pyOrbSlam3/pyOrbSlam3/modules/ORB_SLAM3/lib',
        '/home/naoto/docker_workspace/MobilePoser/third_party/pyOrbSlam3/pyOrbSlam3/modules/ORB_SLAM3/Thirdparty/DBoW2/lib',
        '/home/naoto/docker_workspace/MobilePoser/third_party/pyOrbSlam3/pyOrbSlam3/modules/ORB_SLAM3/Thirdparty/g2o/lib',
    ]
    
    os.environ['LD_LIBRARY_PATH'] = ':'.join(lib_paths) + ':' + os.environ.get('LD_LIBRARY_PATH', '')
    
    pyorb_build_path = '/home/naoto/docker_workspace/MobilePoser/third_party/pyOrbSlam3/pyOrbSlam3/build'
    if pyorb_build_path not in sys.path:
        sys.path.insert(0, pyorb_build_path)
    
    print("Environment setup complete")

# Setup environment before imports
setup_environment()

# Try importing with ctypes preloading
import ctypes

# Pre-load Pangolin libraries
pangolin_libs = [
    'libpango_core.so.0',
    'libpango_opengl.so.0', 
    'libpango_windowing.so.0',
    'libpango_image.so.0',
    'libpango_vars.so.0',
    'libpango_display.so.0',
]

for lib in pangolin_libs:
    try:
        ctypes.CDLL(f'/usr/local/lib/{lib}', ctypes.RTLD_GLOBAL)
        print(f"✓ Loaded {lib}")
    except Exception as e:
        print(f"✗ Failed to load {lib}: {e}")

# Now import pyOrbSlam
try:
    import pyOrbSlam
    print("✅ pyOrbSlam imported successfully!")
except Exception as e:
    print(f"❌ Failed to import pyOrbSlam: {e}")
    sys.exit(1)


class NymeriaVideoPreprocessor:
    """Preprocessor for Nymeria RGB videos for SLAM."""
    
    def __init__(self):
        # Camera calibration parameters for 640x480 (from nymeria_mono_base.yaml)
        self.fx = 517.306408
        self.fy = 516.469215
        self.cx = 318.643040
        self.cy = 255.313989
        
        # Distortion coefficients
        self.k1 = 0.262383
        self.k2 = -0.953104
        self.p1 = -0.005358
        self.p2 = 0.002628
        self.k3 = 1.163314
        
        # Target resolution
        self.target_width = 640
        self.target_height = 480
        
        # Camera matrix
        self.camera_matrix = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Distortion coefficients
        self.dist_coeffs = np.array([self.k1, self.k2, self.p1, self.p2, self.k3], dtype=np.float32)
        
        # Compute undistortion maps
        self.new_camera_matrix, self.roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, 
            (self.target_width, self.target_height), 1, 
            (self.target_width, self.target_height)
        )
        
        self.mapx, self.mapy = cv2.initUndistortRectifyMap(
            self.camera_matrix, self.dist_coeffs, None, 
            self.new_camera_matrix, 
            (self.target_width, self.target_height), 
            cv2.CV_32FC1
        )
    
    def preprocess_frame(self, frame):
        """Preprocess a frame for SLAM."""
        # Original frame is 1408x1408, need to crop and resize
        h, w = frame.shape[:2]
        
        if h == 1408 and w == 1408:
            # Center crop to maintain aspect ratio
            # Target aspect ratio is 640/480 = 4/3
            crop_width = 1408
            crop_height = int(1408 * 480 / 640)  # 1056
            
            # Center crop
            y_start = (h - crop_height) // 2
            y_end = y_start + crop_height
            frame = frame[y_start:y_end, :, :]
        
        # Resize to target resolution
        frame = cv2.resize(frame, (self.target_width, self.target_height))
        
        # Undistort
        frame = cv2.remap(frame, self.mapx, self.mapy, cv2.INTER_LINEAR)
        
        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        return gray


def run_orbslam3_on_video(video_path, max_frames=500):
    """Run ORB-SLAM3 on a video and return trajectory."""
    
    # Initialize preprocessor
    preprocessor = NymeriaVideoPreprocessor()
    
    # ORB-SLAM3 paths
    vocab_path = "/home/naoto/docker_workspace/MobilePoser/third_party/pyOrbSlam3/pyOrbSlam3/modules/ORB_SLAM3/Vocabulary/ORBvoc.txt"
    config_path = "/home/naoto/docker_workspace/MobilePoser/mobileposer/slam_configs/nymeria_mono_base.yaml"
    
    # Check files exist
    if not os.path.exists(vocab_path):
        print(f"❌ Vocabulary file not found: {vocab_path}")
        return None
    
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        return None
    
    # Initialize ORB-SLAM3
    print("\nInitializing ORB-SLAM3...")
    slam = pyOrbSlam.OrbSlam(vocab_path, config_path, "Mono", False)  # No viewer
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Failed to open video: {video_path}")
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video FPS: {fps}")
    
    # Process frames
    trajectory = []
    frame_count = 0
    lost_count = 0
    
    print(f"\nProcessing video (max {max_frames} frames)...")
    
    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preprocess frame
        gray = preprocessor.preprocess_frame(frame)
        
        # Timestamp (assuming 30 FPS)
        timestamp = frame_count / 30.0
        
        # Process with SLAM
        try:
            pose = slam.process(gray, timestamp)
            
            # Get tracking state
            state = slam.GetTrackingState()
            state_str = ["NO_IMAGES_YET", "NOT_INITIALIZED", "OK", "LOST"][state] if state < 4 else "UNKNOWN"
            
            if pose is not None:
                # Extract translation from pose matrix
                if isinstance(pose, np.ndarray) and pose.shape == (4, 4):
                    x, y, z = pose[0, 3], pose[1, 3], pose[2, 3]
                    trajectory.append([timestamp, x, y, z])
                    
                    if frame_count % 100 == 0:
                        print(f"  Frame {frame_count}: State={state_str}, Pose=[{x:.3f}, {y:.3f}, {z:.3f}]")
            else:
                if state == 3:  # LOST
                    lost_count += 1
                
                if frame_count % 100 == 0:
                    print(f"  Frame {frame_count}: State={state_str}, No pose")
        
        except Exception as e:
            print(f"  Error at frame {frame_count}: {e}")
        
        frame_count += 1
    
    cap.release()
    
    print(f"\nProcessed {frame_count} frames")
    print(f"Tracked poses: {len(trajectory)}")
    print(f"Lost frames: {lost_count}")
    
    # Shutdown SLAM
    slam.shutdown()
    
    return np.array(trajectory) if trajectory else None


def load_reference_trajectory(csv_path):
    """Load reference trajectory from Nymeria SLAM CSV."""
    df = pd.read_csv(csv_path)
    
    # Extract relevant columns
    # The CSV has: tracking_timestamp_us, tx_world_device, ty_world_device, tz_world_device
    timestamps = df['tracking_timestamp_us'].values / 1e6  # Convert to seconds
    tx = df['tx_world_device'].values
    ty = df['ty_world_device'].values
    tz = df['tz_world_device'].values
    
    # Make timestamps relative to first frame
    timestamps -= timestamps[0]
    
    trajectory = np.column_stack([timestamps, tx, ty, tz])
    return trajectory


def visualize_trajectories(slam_traj, ref_traj, output_path):
    """Visualize and compare two trajectories."""
    fig = plt.figure(figsize=(15, 10))
    
    # 3D plot
    ax1 = fig.add_subplot(221, projection='3d')
    if slam_traj is not None and len(slam_traj) > 0:
        ax1.plot(slam_traj[:, 1], slam_traj[:, 2], slam_traj[:, 3], 'b-', label='ORB-SLAM3', linewidth=2)
    ax1.plot(ref_traj[:, 1], ref_traj[:, 2], ref_traj[:, 3], 'r--', label='Reference', linewidth=2)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Trajectory Comparison')
    ax1.legend()
    ax1.grid(True)
    
    # XY plane
    ax2 = fig.add_subplot(222)
    if slam_traj is not None and len(slam_traj) > 0:
        ax2.plot(slam_traj[:, 1], slam_traj[:, 2], 'b-', label='ORB-SLAM3', linewidth=2)
    ax2.plot(ref_traj[:, 1], ref_traj[:, 2], 'r--', label='Reference', linewidth=2)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('XY Plane (Top View)')
    ax2.legend()
    ax2.grid(True)
    ax2.axis('equal')
    
    # XZ plane
    ax3 = fig.add_subplot(223)
    if slam_traj is not None and len(slam_traj) > 0:
        ax3.plot(slam_traj[:, 1], slam_traj[:, 3], 'b-', label='ORB-SLAM3', linewidth=2)
    ax3.plot(ref_traj[:, 1], ref_traj[:, 3], 'r--', label='Reference', linewidth=2)
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Z (m)')
    ax3.set_title('XZ Plane (Side View)')
    ax3.legend()
    ax3.grid(True)
    ax3.axis('equal')
    
    # YZ plane
    ax4 = fig.add_subplot(224)
    if slam_traj is not None and len(slam_traj) > 0:
        ax4.plot(slam_traj[:, 2], slam_traj[:, 3], 'b-', label='ORB-SLAM3', linewidth=2)
    ax4.plot(ref_traj[:, 2], ref_traj[:, 3], 'r--', label='Reference', linewidth=2)
    ax4.set_xlabel('Y (m)')
    ax4.set_ylabel('Z (m)')
    ax4.set_title('YZ Plane (Front View)')
    ax4.legend()
    ax4.grid(True)
    ax4.axis('equal')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Trajectory comparison saved to: {output_path}")
    plt.close()


def main():
    """Main test function."""
    print("=" * 60)
    print("ORB-SLAM3 Test with Nymeria Sequence")
    print("Sequence: 20231031_s0_jason_brown_act0_zb36n0")
    print("=" * 60)
    
    # Video path
    video_path = "/mnt/nas2/naoto/nymeria_dataset/data_video_main_rgb/20231031_s0_jason_brown_act0_zb36n0/video_main_rgb.mp4"
    
    # Check if video exists
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return
    
    # Run ORB-SLAM3
    print("\n1. Running ORB-SLAM3 on video...")
    slam_trajectory = run_orbslam3_on_video(video_path, max_frames=500)
    
    # Save SLAM trajectory
    if slam_trajectory is not None and len(slam_trajectory) > 0:
        slam_traj_path = "/home/naoto/docker_workspace/MobilePoser/tmp/orbslam3_trajectory_nymeria.txt"
        np.savetxt(slam_traj_path, slam_trajectory, fmt='%.6f', 
                   header='timestamp x y z', comments='')
        print(f"✅ SLAM trajectory saved to: {slam_traj_path}")
    
    # Load reference trajectory
    print("\n2. Loading reference trajectory...")
    ref_traj_path = "/home/naoto/docker_workspace/MobilePoser/tmp/recording_head/mps/slam/closed_loop_trajectory.csv"
    
    if not os.path.exists(ref_traj_path):
        print(f"❌ Reference trajectory not found: {ref_traj_path}")
        return
    
    ref_trajectory = load_reference_trajectory(ref_traj_path)
    print(f"✅ Loaded reference trajectory: {len(ref_trajectory)} poses")
    
    # Visualize comparison
    print("\n3. Visualizing trajectory comparison...")
    output_path = "/home/naoto/docker_workspace/MobilePoser/tmp/slam_trajectory_comparison_nymeria.png"
    visualize_trajectories(slam_trajectory, ref_trajectory[:500], output_path)
    
    print("\n✅ Test completed!")


if __name__ == "__main__":
    main()