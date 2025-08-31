#!/usr/bin/env python3
"""
Example script demonstrating head pose ensemble estimation.
Combines MobilePoser head IMU with Visual-Inertial SLAM for improved
head position and orientation estimation.
"""

import os
import cv2
import numpy as np
import torch
from pathlib import Path
import time
import argparse
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from mobileposer.head_pose_ensemble import HeadPoseEnsemble, HeadPoseData
from mobileposer.config import paths


def load_nymeria_data(sequence_path: str) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
    """
    Load synchronized RGB video and IMU data from Nymeria dataset.
    
    Args:
        sequence_path: Path to Nymeria sequence directory
        
    Returns:
        Tuple of (rgb_frames, imu_data_list, timestamps)
    """
    sequence_dir = Path(sequence_path)
    
    # Load RGB video
    video_path = sequence_dir / "video_main_rgb.mp4"
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Load video frames
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    rgb_frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frames.append(frame_rgb)
        frame_count += 1
    
    cap.release()
    
    # Generate corresponding timestamps
    timestamps = [i / fps for i in range(len(rgb_frames))]
    
    # For demonstration, generate mock IMU data
    # In practice, you would load actual IMU data from the Nymeria dataset
    imu_data_list = generate_realistic_imu_data(len(rgb_frames), fps)
    
    print(f"Loaded {len(rgb_frames)} RGB frames")
    print(f"Generated {len(imu_data_list)} IMU samples")
    print(f"Video FPS: {fps}, Duration: {timestamps[-1]:.2f}s")
    
    return rgb_frames, imu_data_list, timestamps


def generate_realistic_imu_data(num_frames: int, fps: float) -> List[np.ndarray]:
    """
    Generate realistic IMU data that simulates head movement.
    This mimics the structure of Nymeria dataset IMU data.
    
    Args:
        num_frames: Number of frames to generate
        fps: Frame rate
        
    Returns:
        List of IMU data arrays
    """
    imu_data_list = []
    
    for i in range(num_frames):
        # Time for motion simulation
        t = i / fps
        
        # Simulate 6 IMU sensors (Nymeria configuration)
        # Sensors: left_wrist(0), right_wrist(1), left_foot(2), right_foot(3), head(4), pelvis(5)
        imu_frame = np.zeros((6, 12))  # 6 sensors, 12 values each (3 acc + 9 rot matrix)
        
        for sensor_idx in range(6):
            if sensor_idx == 4:  # Head sensor (index 4)
                # Realistic head motion: nodding and turning
                head_nod = 0.3 * np.sin(t * 1.5)  # Nodding motion
                head_turn = 0.2 * np.sin(t * 0.8)  # Head turning
                
                # Head acceleration (includes gravity and motion)
                acc = np.array([
                    0.1 * np.sin(t * 2),           # x: lateral motion
                    -9.81 + 0.3 * np.cos(t * 1.5), # y: gravity + vertical motion
                    0.1 * np.cos(t * 2.5)          # z: forward/back motion
                ])
                
                # Head rotation matrix (nodding + turning)
                # Rotation around X (nodding)
                R_nod = np.array([
                    [1, 0, 0],
                    [0, np.cos(head_nod), -np.sin(head_nod)],
                    [0, np.sin(head_nod), np.cos(head_nod)]
                ])
                
                # Rotation around Y (turning)
                R_turn = np.array([
                    [np.cos(head_turn), 0, np.sin(head_turn)],
                    [0, 1, 0],
                    [-np.sin(head_turn), 0, np.cos(head_turn)]
                ])
                
                # Combined rotation
                R_combined = R_turn @ R_nod
                rot_matrix = R_combined.flatten()
                
            else:  # Other sensors
                # Simpler motion for non-head sensors
                motion_phase = t + sensor_idx * 0.5
                
                acc = np.array([
                    0.05 * np.sin(motion_phase * 2),
                    -9.81 + 0.1 * np.cos(motion_phase),
                    0.05 * np.cos(motion_phase * 1.5)
                ])
                
                # Small rotation perturbations
                angle = 0.05 * np.sin(motion_phase)
                rot_matrix = np.array([
                    [np.cos(angle), -np.sin(angle), 0],
                    [np.sin(angle), np.cos(angle), 0],
                    [0, 0, 1]
                ]).flatten()
            
            # Combine acceleration and rotation
            imu_frame[sensor_idx] = np.concatenate([acc, rot_matrix])
        
        # Flatten to match expected format (72 values total)
        imu_data_list.append(imu_frame.flatten())
    
    return imu_data_list


def run_head_pose_demo(sequence_path: str,
                      mobileposer_weights: str,
                      slam_type: str = "mock_vi",
                      fusion_method: str = "weighted_average",
                      max_frames: int = 300,
                      head_imu_index: int = 4):
    """
    Run head pose ensemble estimation demo.
    
    Args:
        sequence_path: Path to Nymeria sequence directory
        mobileposer_weights: Path to MobilePoser weights
        slam_type: Type of Visual-Inertial SLAM
        fusion_method: Pose fusion strategy
        max_frames: Maximum frames to process
        head_imu_index: Index of head IMU sensor
    """
    print(f"Starting head pose ensemble demo")
    print(f"Sequence: {sequence_path}")
    print(f"SLAM type: {slam_type}, Fusion: {fusion_method}")
    print(f"Head IMU index: {head_imu_index}")
    
    # Load data
    try:
        rgb_frames, imu_data_list, timestamps = load_nymeria_data(sequence_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Limit frames for demo
    if len(rgb_frames) > max_frames:
        rgb_frames = rgb_frames[:max_frames]
        imu_data_list = imu_data_list[:max_frames]
        timestamps = timestamps[:max_frames]
    
    # Initialize head pose ensemble
    try:
        head_ensemble = HeadPoseEnsemble(
            mobileposer_weights=mobileposer_weights,
            slam_type=slam_type,
            head_imu_index=head_imu_index,
            fusion_method=fusion_method
        )
    except Exception as e:
        print(f"Error initializing ensemble: {e}")
        return
    
    # Process frames
    print("Processing frames...")
    head_poses = []
    processing_times = []
    
    for i, (rgb_frame, imu_data, timestamp) in enumerate(
        zip(rgb_frames, imu_data_list, timestamps)):
        
        start_time = time.time()
        
        # Process frame
        head_pose = head_ensemble.process_frame(rgb_frame, imu_data, timestamp)
        
        processing_time = time.time() - start_time
        processing_times.append(processing_time)
        
        if head_pose is not None:
            head_poses.append(head_pose)
        
        # Progress update
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(rgb_frames)} frames")
    
    # Analyze results
    print(f"\nProcessing completed!")
    print(f"Total head poses: {len(head_poses)}")
    print(f"Average processing time: {np.mean(processing_times):.4f}s")
    print(f"Processing FPS: {1.0/np.mean(processing_times):.1f}")
    
    if head_poses:
        # Analyze pose sources
        imu_poses = [p for p in head_poses if p.source == "imu"]
        slam_poses = [p for p in head_poses if p.source == "vi_slam"]
        fused_poses = [p for p in head_poses if p.source == "fused"]
        
        print(f"IMU-only poses: {len(imu_poses)}")
        print(f"VI-SLAM-only poses: {len(slam_poses)}")
        print(f"Fused poses: {len(fused_poses)}")
        
        # Calculate statistics
        avg_confidence = np.mean([p.confidence for p in head_poses])
        print(f"Average confidence: {avg_confidence:.3f}")
        
        if fused_poses:
            avg_scale = np.mean([p.scale_factor for p in fused_poses if hasattr(p, 'scale_factor')])
            print(f"Average scale factor: {avg_scale:.3f}")
        
        # Visualize results
        visualize_head_trajectory(head_poses)
        analyze_pose_accuracy(head_poses)
    
    # Print performance stats
    stats = head_ensemble.get_performance_stats()
    print(f"\nEnsemble Performance:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


def visualize_head_trajectory(head_poses: List[HeadPoseData]):
    """
    Visualize head pose trajectory and analysis.
    
    Args:
        head_poses: List of head pose estimates
    """
    if not head_poses:
        return
    
    # Extract data
    positions = np.array([p.position for p in head_poses])
    timestamps = np.array([p.timestamp for p in head_poses])
    confidences = np.array([p.confidence for p in head_poses])
    sources = [p.source for p in head_poses]
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(16, 12))
    
    # 3D trajectory
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    scatter = ax1.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                         c=confidences, cmap='viridis', alpha=0.7)
    ax1.set_xlabel('X (meters)')
    ax1.set_ylabel('Y (meters)')
    ax1.set_zlabel('Z (meters)')
    ax1.set_title('3D Head Trajectory')
    plt.colorbar(scatter, ax=ax1, label='Confidence')
    
    # Top-down view
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.scatter(positions[:, 0], positions[:, 2], c=confidences, cmap='viridis', alpha=0.7)
    ax2.set_xlabel('X (meters)')
    ax2.set_ylabel('Z (meters)')
    ax2.set_title('Head Trajectory (Top View)')
    ax2.grid(True)
    
    # Position vs time
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(timestamps, positions[:, 0], label='X', alpha=0.7)
    ax3.plot(timestamps, positions[:, 1], label='Y', alpha=0.7)
    ax3.plot(timestamps, positions[:, 2], label='Z', alpha=0.7)
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('Position (meters)')
    ax3.set_title('Head Position vs Time')
    ax3.legend()
    ax3.grid(True)
    
    # Confidence over time by source
    ax4 = fig.add_subplot(2, 3, 4)
    colors = {'imu': 'blue', 'vi_slam': 'red', 'fused': 'green'}
    for source in set(sources):
        mask = np.array([s == source for s in sources])
        if np.any(mask):
            ax4.scatter(timestamps[mask], confidences[mask], 
                       label=source, color=colors.get(source, 'gray'), alpha=0.7)
    ax4.set_xlabel('Time (seconds)')
    ax4.set_ylabel('Confidence')
    ax4.set_title('Confidence vs Time by Source')
    ax4.legend()
    ax4.grid(True)
    
    # Source distribution
    ax5 = fig.add_subplot(2, 3, 5)
    source_counts = {source: sources.count(source) for source in set(sources)}
    ax5.pie(source_counts.values(), labels=source_counts.keys(), autopct='%1.1f%%')
    ax5.set_title('Pose Source Distribution')
    
    # Confidence histogram
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.hist(confidences, bins=20, alpha=0.7, edgecolor='black')
    ax6.set_xlabel('Confidence')
    ax6.set_ylabel('Frequency')
    ax6.set_title('Confidence Distribution')
    ax6.grid(True)
    
    plt.tight_layout()
    plt.show()


def analyze_pose_accuracy(head_poses: List[HeadPoseData]):
    """
    Analyze pose estimation accuracy and consistency.
    
    Args:
        head_poses: List of head pose estimates
    """
    if len(head_poses) < 10:
        return
    
    print("\n--- Pose Accuracy Analysis ---")
    
    # Position analysis
    positions = np.array([p.position for p in head_poses])
    
    # Calculate motion statistics
    position_diffs = np.diff(positions, axis=0)
    motion_magnitudes = np.linalg.norm(position_diffs, axis=1)
    
    print(f"Average motion per frame: {np.mean(motion_magnitudes):.4f} m")
    print(f"Max motion per frame: {np.max(motion_magnitudes):.4f} m")
    print(f"Motion standard deviation: {np.std(motion_magnitudes):.4f} m")
    
    # Trajectory smoothness (lower values indicate smoother motion)
    if len(motion_magnitudes) > 1:
        motion_jerk = np.diff(motion_magnitudes)
        smoothness = np.std(motion_jerk)
        print(f"Trajectory smoothness (jerk std): {smoothness:.4f}")
    
    # Analyze by source
    sources = [p.source for p in head_poses]
    unique_sources = set(sources)
    
    for source in unique_sources:
        source_poses = [p for p in head_poses if p.source == source]
        if len(source_poses) > 5:
            source_positions = np.array([p.position for p in source_poses])
            source_confidences = np.array([p.confidence for p in source_poses])
            
            print(f"\n{source.upper()} Source Analysis:")
            print(f"  Count: {len(source_poses)}")
            print(f"  Avg confidence: {np.mean(source_confidences):.3f}")
            print(f"  Position range: X={np.ptp(source_positions[:, 0]):.3f}m, "
                  f"Y={np.ptp(source_positions[:, 1]):.3f}m, "
                  f"Z={np.ptp(source_positions[:, 2]):.3f}m")
    
    # Scale analysis for VI-SLAM poses
    vi_slam_poses = [p for p in head_poses if hasattr(p, 'scale_factor') and p.source in ['vi_slam', 'fused']]
    if vi_slam_poses:
        scale_factors = [p.scale_factor for p in vi_slam_poses]
        print(f"\nScale Factor Analysis:")
        print(f"  Mean scale: {np.mean(scale_factors):.3f}")
        print(f"  Scale std: {np.std(scale_factors):.3f}")
        print(f"  Scale range: {np.min(scale_factors):.3f} - {np.max(scale_factors):.3f}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Head pose ensemble estimation demo")
    parser.add_argument('--sequence', type=str, required=True,
                       help='Path to Nymeria sequence directory')
    parser.add_argument('--weights', type=str, default=str(paths.weights_file),
                       help='Path to MobilePoser weights')
    parser.add_argument('--slam-type', type=str, default='mock_vi',
                       choices=['mock_vi', 'orb_slam3_vi', 'real_vi', 'orb_slam3_mono', 'real_mono'],
                       help='Type of Visual-Inertial SLAM (mock_vi for testing, real_vi/orb_slam3_vi for real ORB-SLAM3)')
    parser.add_argument('--fusion-method', type=str, default='weighted_average',
                       choices=['weighted_average', 'confidence_based'],
                       help='Head pose fusion method')
    parser.add_argument('--max-frames', type=int, default=300,
                       help='Maximum frames to process')
    parser.add_argument('--head-imu-index', type=int, default=4,
                       help='Index of head IMU sensor')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.sequence):
        print(f"Error: Sequence directory not found: {args.sequence}")
        print("Please provide a valid Nymeria sequence directory.")
        print("Example: /mnt/nas2/naoto/nymeria_dataset/data_video_main_rgb/20231031_s0_jason_brown_act0_zb36n0/")
        return
    
    if not os.path.exists(args.weights):
        print(f"Error: MobilePoser weights not found: {args.weights}")
        return
    
    # Run demo
    run_head_pose_demo(
        sequence_path=args.sequence,
        mobileposer_weights=args.weights,
        slam_type=args.slam_type,
        fusion_method=args.fusion_method,
        max_frames=args.max_frames,
        head_imu_index=args.head_imu_index
    )


if __name__ == "__main__":
    main()