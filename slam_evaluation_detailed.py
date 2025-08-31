#!/usr/bin/env python3
"""
Detailed SLAM evaluation with comprehensive visualization similar to the reference image.
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import pandas as pd
from scipy.spatial.transform import Rotation
from matplotlib.patches import Rectangle

# Setup environment for pyOrbSlam3
def setup_environment():
    """Set up environment for pyOrbSlam3."""
    lib_paths = [
        '/usr/local/lib',
        '/usr/lib/x86_64-linux-gnu',
        '/home/naoto/docker_workspace/MobilePoser/third_party/pyOrbSlam3/pyOrbSlam3/modules/ORB_SLAM3/lib',
    ]
    
    os.environ['LD_LIBRARY_PATH'] = ':'.join(lib_paths) + ':' + os.environ.get('LD_LIBRARY_PATH', '')
    
    pyorb_build_path = '/home/naoto/docker_workspace/MobilePoser/third_party/pyOrbSlam3/pyOrbSlam3/build'
    if pyorb_build_path not in sys.path:
        sys.path.insert(0, pyorb_build_path)

setup_environment()

# Pre-load libraries
import ctypes
pangolin_libs = ['libpango_core.so.0', 'libpango_opengl.so.0', 'libpango_windowing.so.0',
                 'libpango_image.so.0', 'libpango_vars.so.0', 'libpango_display.so.0']

for lib in pangolin_libs:
    try:
        ctypes.CDLL(f'/usr/local/lib/{lib}', ctypes.RTLD_GLOBAL)
    except:
        pass

import pyOrbSlam
print("✅ pyOrbSlam imported successfully!")


class NymeriaVideoPreprocessor:
    """Preprocessor for Nymeria RGB videos for SLAM."""
    
    def __init__(self):
        # Camera calibration parameters
        self.fx = 517.306408
        self.fy = 516.469215
        self.cx = 318.643040
        self.cy = 255.313989
        
        # Distortion coefficients
        self.dist_coeffs = np.array([0.262383, -0.953104, -0.005358, 0.002628, 1.163314], dtype=np.float32)
        
        # Target resolution
        self.target_width = 640
        self.target_height = 480
        
        # Camera matrix
        self.camera_matrix = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Compute undistortion maps
        self.new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
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
        h, w = frame.shape[:2]
        
        if h == 1408 and w == 1408:
            # Center crop
            crop_height = int(1408 * 480 / 640)
            y_start = (h - crop_height) // 2
            y_end = y_start + crop_height
            frame = frame[y_start:y_end, :, :]
        
        # Resize
        frame = cv2.resize(frame, (self.target_width, self.target_height))
        
        # Undistort
        frame = cv2.remap(frame, self.mapx, self.mapy, cv2.INTER_LINEAR)
        
        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        return gray


def run_orbslam3_detailed(video_path, max_frames=300):
    """Run ORB-SLAM3 and collect detailed tracking information."""
    
    preprocessor = NymeriaVideoPreprocessor()
    
    # ORB-SLAM3 paths
    vocab_path = "/home/naoto/docker_workspace/MobilePoser/third_party/pyOrbSlam3/pyOrbSlam3/modules/ORB_SLAM3/Vocabulary/ORBvoc.txt"
    config_path = "/home/naoto/docker_workspace/MobilePoser/mobileposer/slam_configs/nymeria_mono_base.yaml"
    
    # Initialize ORB-SLAM3
    print("\nInitializing ORB-SLAM3...")
    slam = pyOrbSlam.OrbSlam(vocab_path, config_path, "Mono", False)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Failed to open video: {video_path}")
        return None, None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Process frames
    trajectory = []
    tracking_info = {
        'timestamps': [],
        'states': [],
        'confidences': [],
        'poses': []
    }
    
    frame_count = 0
    
    print(f"\nProcessing video (max {max_frames} frames)...")
    
    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preprocess frame
        gray = preprocessor.preprocess_frame(frame)
        
        # Timestamp
        timestamp = frame_count / fps
        
        # Process with SLAM
        try:
            pose = slam.process(gray, timestamp)
            
            # Get tracking state
            state = slam.GetTrackingState()
            state_names = ["NO_IMAGES_YET", "NOT_INITIALIZED", "OK", "LOST"]
            state_str = state_names[state] if state < len(state_names) else "UNKNOWN"
            
            # Calculate confidence based on state
            if state == 2:  # OK
                confidence = 0.9
            elif state == 1:  # NOT_INITIALIZED
                confidence = 0.2
            else:
                confidence = 0.0
            
            tracking_info['timestamps'].append(timestamp)
            tracking_info['states'].append(state)
            tracking_info['confidences'].append(confidence)
            
            if pose is not None and isinstance(pose, np.ndarray) and pose.shape == (4, 4):
                tracking_info['poses'].append(pose.copy())
                x, y, z = pose[0, 3], pose[1, 3], pose[2, 3]
                trajectory.append([timestamp, x, y, z])
                
                if frame_count % 50 == 0:
                    print(f"  Frame {frame_count}: State={state_str}, Pose=[{x:.3f}, {y:.3f}, {z:.3f}]")
            else:
                tracking_info['poses'].append(None)
                if frame_count % 50 == 0:
                    print(f"  Frame {frame_count}: State={state_str}, No pose")
        
        except Exception as e:
            print(f"  Error at frame {frame_count}: {e}")
            tracking_info['timestamps'].append(timestamp)
            tracking_info['states'].append(-1)
            tracking_info['confidences'].append(0.0)
            tracking_info['poses'].append(None)
        
        frame_count += 1
    
    cap.release()
    slam.shutdown()
    
    print(f"\nProcessed {frame_count} frames")
    print(f"Tracked poses: {len(trajectory)}")
    
    return np.array(trajectory) if trajectory else None, tracking_info


def load_reference_trajectory(csv_path):
    """Load reference trajectory from Nymeria SLAM CSV."""
    df = pd.read_csv(csv_path)
    
    # Extract columns
    timestamps = df['tracking_timestamp_us'].values / 1e6
    tx = df['tx_world_device'].values
    ty = df['ty_world_device'].values
    tz = df['tz_world_device'].values
    
    # Extract rotation (quaternion)
    qw = df['qw_world_device'].values
    qx = df['qx_world_device'].values
    qy = df['qy_world_device'].values
    qz = df['qz_world_device'].values
    
    # Make timestamps relative
    timestamps -= timestamps[0]
    
    # Make positions relative to first frame (start at origin)
    positions = np.column_stack([tx, ty, tz])
    positions -= positions[0]  # Subtract initial position
    
    return {
        'timestamps': timestamps,
        'positions': positions,
        'quaternions': np.column_stack([qw, qx, qy, qz])
    }


def align_trajectories(slam_traj, ref_traj, scale_factor=None):
    """Align SLAM trajectory to reference using similarity transformation."""
    if slam_traj is None or len(slam_traj) < 10:
        return None, 1.0
    
    # Extract positions
    slam_pos = slam_traj[:, 1:4]
    
    # Find corresponding reference positions by timestamp
    ref_times = ref_traj['timestamps']
    ref_pos = ref_traj['positions']
    
    # Interpolate reference positions at SLAM timestamps
    aligned_ref_pos = []
    for t in slam_traj[:, 0]:
        idx = np.searchsorted(ref_times, t)
        if idx > 0 and idx < len(ref_times):
            # Linear interpolation
            t0, t1 = ref_times[idx-1], ref_times[idx]
            alpha = (t - t0) / (t1 - t0)
            pos = (1 - alpha) * ref_pos[idx-1] + alpha * ref_pos[idx]
            aligned_ref_pos.append(pos)
    
    if len(aligned_ref_pos) < 10:
        return slam_pos, 1.0
    
    aligned_ref_pos = np.array(aligned_ref_pos[:len(slam_pos)])
    
    # Estimate scale if not provided
    if scale_factor is None:
        # Use median of distance ratios
        slam_dists = np.linalg.norm(np.diff(slam_pos, axis=0), axis=1)
        ref_dists = np.linalg.norm(np.diff(aligned_ref_pos, axis=0), axis=1)
        
        # Ensure same length
        min_len = min(len(slam_dists), len(ref_dists))
        slam_dists = slam_dists[:min_len]
        ref_dists = ref_dists[:min_len]
        
        valid_mask = (slam_dists > 1e-6) & (ref_dists > 1e-6)
        if np.any(valid_mask):
            scale_factor = np.median(ref_dists[valid_mask] / slam_dists[valid_mask])
        else:
            scale_factor = 1.0
    
    # Apply scale
    scaled_slam_pos = slam_pos * scale_factor
    
    return scaled_slam_pos, scale_factor


def calculate_errors(slam_pos, ref_traj, timestamps):
    """Calculate translation and rotation errors."""
    errors = {
        'translation': [],
        'rotation': [],
        'timestamps': []
    }
    
    ref_times = ref_traj['timestamps']
    ref_pos = ref_traj['positions']
    ref_quats = ref_traj['quaternions']
    
    for i, t in enumerate(timestamps):
        idx = np.searchsorted(ref_times, t)
        if idx > 0 and idx < len(ref_times):
            # Interpolate reference
            t0, t1 = ref_times[idx-1], ref_times[idx]
            alpha = (t - t0) / (t1 - t0)
            
            ref_p = (1 - alpha) * ref_pos[idx-1] + alpha * ref_pos[idx]
            ref_q = Rotation.from_quat(ref_quats[idx-1]).as_matrix()  # Simplified - no slerp
            
            # Translation error
            trans_err = np.linalg.norm(slam_pos[i] - ref_p)
            errors['translation'].append(trans_err)
            
            # Rotation error (placeholder - would need SLAM rotations)
            errors['rotation'].append(0.0)  # Not available from monocular SLAM
            
            errors['timestamps'].append(t)
    
    return errors


def create_detailed_visualization(slam_traj, ref_traj, tracking_info, output_path):
    """Create detailed visualization similar to reference image."""
    
    # Align trajectories
    if slam_traj is not None and len(slam_traj) > 0:
        aligned_slam_pos, scale_factor = align_trajectories(slam_traj, ref_traj)
        errors = calculate_errors(aligned_slam_pos, ref_traj, slam_traj[:, 0])
    else:
        aligned_slam_pos = None
        scale_factor = 1.0
        errors = None
    
    # Create figure with GridSpec
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. 3D Trajectory Plot
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    ref_pos = ref_traj['positions'][:300]  # Limit to processed frames
    
    # Plot reference trajectory
    ax1.plot(ref_pos[:, 0], ref_pos[:, 1], ref_pos[:, 2], 'g-', linewidth=2, label='Ground Truth (Xsens)')
    
    if aligned_slam_pos is not None and len(aligned_slam_pos) > 0:
        # Scale SLAM trajectory for better visualization
        ax1.plot(aligned_slam_pos[:, 0], aligned_slam_pos[:, 1], aligned_slam_pos[:, 2], 
                'b-', linewidth=2, label=f'SLAM (ORB-SLAM3) x{scale_factor:.1f}')
        # Mark start and end
        ax1.scatter(*aligned_slam_pos[0], color='green', s=100, marker='o', label='Start')
        ax1.scatter(*aligned_slam_pos[-1], color='red', s=100, marker='s', label='End')
    
    # Also mark GT start and end
    ax1.scatter(*ref_pos[0], color='lightgreen', s=100, marker='o')
    ax1.scatter(*ref_pos[-1], color='salmon', s=100, marker='s')
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Head Trajectory: SLAM vs Ground Truth')
    ax1.legend()
    ax1.grid(True)
    
    # 2. Translation Error Over Time
    ax2 = fig.add_subplot(gs[0, 1])
    if errors and len(errors['translation']) > 0:
        times = np.array(errors['timestamps'])
        trans_errors = np.array(errors['translation'])
        
        ax2.fill_between(times, 0, trans_errors, alpha=0.3, color='blue')
        ax2.plot(times, trans_errors, 'b-', linewidth=2)
        
        mean_error = np.mean(trans_errors)
        ax2.axhline(mean_error, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {mean_error:.3f}m')
        
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Translation Error (m)')
        ax2.set_title('Translation Error Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 10)
    
    # 3. Rotation Error Over Time (placeholder)
    ax3 = fig.add_subplot(gs[0, 2])
    if tracking_info:
        times = np.array(tracking_info['timestamps'])
        # Since we don't have rotation from monocular SLAM, show tracking state
        states = np.array(tracking_info['states'])
        
        ax3.fill_between(times, 0, states * 40, alpha=0.3, color='red')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Tracking State')
        ax3.set_title('SLAM Tracking State Over Time')
        ax3.set_ylim(0, 130)
        ax3.axhline(120, color='black', linestyle='--', linewidth=1, label='OK State')
        ax3.text(5, 125, 'State: 0=NoImg, 1=Init, 2=OK, 3=Lost', fontsize=9)
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(0, 10)
    
    # 4. Top-Down View with Error Heatmap
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(ref_pos[:, 0], ref_pos[:, 1], 'g-', linewidth=2, label='Ground Truth')
    
    if aligned_slam_pos is not None and errors is not None and len(errors['translation']) > 0:
        # Create scatter plot colored by error
        # Ensure same length
        min_len = min(len(aligned_slam_pos), len(errors['translation']))
        scatter = ax4.scatter(aligned_slam_pos[:min_len, 0], aligned_slam_pos[:min_len, 1], 
                            c=errors['translation'][:min_len], cmap='YlOrRd', s=20, alpha=0.8)
        cbar = plt.colorbar(scatter, ax=ax4)
        cbar.set_label('Translation Error (m)')
    
    ax4.set_xlabel('X (m)')
    ax4.set_ylabel('Y (m)')
    ax4.set_title('Top-Down View with Error Heatmap')
    ax4.axis('equal')
    ax4.grid(True, alpha=0.3)
    
    # 5. SLAM Tracking Confidence
    ax5 = fig.add_subplot(gs[1, 1])
    if tracking_info:
        times = np.array(tracking_info['timestamps'])
        confidences = np.array(tracking_info['confidences'])
        
        ax5.fill_between(times, 0, confidences, alpha=0.3, color='blue')
        ax5.plot(times, confidences, 'b-', linewidth=2)
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('SLAM Confidence')
        ax5.set_title('SLAM Tracking Confidence')
        ax5.set_ylim(0, 1.0)
        ax5.grid(True, alpha=0.3)
        ax5.set_xlim(0, 10)
    
    # 6. Results Summary
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    # Create text summary
    if errors and len(errors['translation']) > 0:
        mean_err = np.mean(errors['translation'])
        std_err = np.std(errors['translation'])
        max_err = np.max(errors['translation'])
        p95_err = np.percentile(errors['translation'], 95)
    else:
        mean_err = std_err = max_err = p95_err = 0.0
    
    summary_text = f"""SLAM Evaluation Results
Sequence: 20231031_s0_jason_brown_act0_zb36n0

Translation Error:
• Mean: {mean_err:.3f} m
• Std: {std_err:.3f} m
• Max: {max_err:.3f} m
• 95th %ile: {p95_err:.3f} m

Scale Factor: {scale_factor:.2f}

Tracking:
• Frames: {len(slam_traj) if slam_traj is not None else 0}
• Lost Tracking: {np.sum(np.array(tracking_info['states']) == 3) if tracking_info else 0} frames
• Initialization: {np.sum(np.array(tracking_info['states']) < 2) if tracking_info else 0} frames
"""
    
    # Add background
    rect = Rectangle((0.05, 0.05), 0.9, 0.9, 
                    facecolor='lightgray', alpha=0.3, 
                    transform=ax6.transAxes)
    ax6.add_patch(rect)
    
    ax6.text(0.1, 0.85, summary_text, transform=ax6.transAxes, 
            fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    # Overall title
    fig.suptitle(f'SLAM Evaluation on Nymeria Sequence: 20231031_s0_jason_brown_act0_zb36n0', 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Detailed evaluation saved to: {output_path}")
    plt.close()


def main():
    """Main evaluation function."""
    print("=" * 60)
    print("Detailed SLAM Evaluation with ORB-SLAM3")
    print("=" * 60)
    
    # Video path
    video_path = "/mnt/nas2/naoto/nymeria_dataset/data_video_main_rgb/20231031_s0_jason_brown_act0_zb36n0/video_main_rgb.mp4"
    
    # Run ORB-SLAM3
    print("\n1. Running ORB-SLAM3 with detailed tracking...")
    slam_trajectory, tracking_info = run_orbslam3_detailed(video_path, max_frames=300)
    
    # Load reference trajectory
    print("\n2. Loading reference trajectory...")
    ref_traj_path = "/home/naoto/docker_workspace/MobilePoser/tmp/recording_head/mps/slam/closed_loop_trajectory.csv"
    ref_trajectory = load_reference_trajectory(ref_traj_path)
    print(f"✅ Loaded reference trajectory: {len(ref_trajectory['timestamps'])} poses")
    
    # Create detailed visualization
    print("\n3. Creating detailed evaluation visualization...")
    output_path = "/home/naoto/docker_workspace/MobilePoser/tmp/slam_detailed_evaluation.png"
    create_detailed_visualization(slam_trajectory, ref_trajectory, tracking_info, output_path)
    
    print("\n✅ Evaluation completed!")


if __name__ == "__main__":
    main()