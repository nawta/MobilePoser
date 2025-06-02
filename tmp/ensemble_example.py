#!/usr/bin/env python3
"""
Example script demonstrating the ensemble pose estimation system
combining MobilePoser (IMU) with ORB-SLAM3 (visual) for improved accuracy.
"""

import os
import cv2
import numpy as np
import torch
from pathlib import Path
import time
import argparse
from typing import List, Tuple
import matplotlib.pyplot as plt

from mobileposer.ensemble import EnsemblePoseEstimator
from mobileposer.models.fusion import PoseEstimate
from mobileposer.config import paths


def load_nymeria_rgb_video(video_path: str) -> Tuple[List[np.ndarray], List[float]]:
    """
    Load RGB video frames from Nymeria dataset.
    
    Args:
        video_path: Path to video_main_rgb.mp4 file
        
    Returns:
        Tuple of (frames, timestamps)
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
        
    frames = []
    timestamps = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        
        # Calculate timestamp based on frame rate
        timestamp = frame_count / fps
        timestamps.append(timestamp)
        
        frame_count += 1
        
    cap.release()
    print(f"Loaded {len(frames)} frames from {video_path}")
    print(f"Video FPS: {fps}, Duration: {timestamps[-1]:.2f}s")
    
    return frames, timestamps


def generate_mock_imu_data(num_frames: int, fps: float = 30.0) -> Tuple[List[np.ndarray], List[float]]:
    """
    Generate mock IMU data for testing when real IMU data is not available.
    
    Args:
        num_frames: Number of IMU samples to generate
        fps: Sampling rate
        
    Returns:
        Tuple of (imu_data_list, timestamps)
    """
    imu_data_list = []
    timestamps = []
    
    for i in range(num_frames):
        # Generate synthetic IMU data (6 sensors, each with 3 acc + 9 orientation values)
        # This mimics the structure expected by MobilePoser
        
        # 6 IMU sensors: head, left wrist, right wrist, left foot, right foot, pelvis
        imu_frame = np.zeros((6, 12))  # 6 sensors, 12 values each (3 acc + 9 rot matrix)
        
        # Add some realistic motion patterns
        t = i / fps
        
        for sensor_idx in range(6):
            # Acceleration (gravity + motion)
            acc = np.array([
                0.1 * np.sin(t * 2 + sensor_idx),  # x
                -9.81 + 0.2 * np.cos(t * 1.5),    # y (gravity + motion)
                0.1 * np.sin(t * 3 + sensor_idx)   # z
            ])
            
            # Rotation matrix (identity + small perturbations)
            angle = 0.1 * np.sin(t + sensor_idx)
            rot_matrix = np.array([
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1]
            ]).flatten()
            
            # Combine acceleration and rotation
            imu_frame[sensor_idx] = np.concatenate([acc, rot_matrix])
        
        # Flatten to match expected input format
        imu_data_list.append(imu_frame.flatten())  # Shape: (72,)
        timestamps.append(t)
    
    return imu_data_list, timestamps


def run_ensemble_demo(video_path: str, 
                     mobileposer_weights: str,
                     slam_type: str = "mock",
                     fusion_method: str = "weighted_average",
                     max_frames: int = 300):
    """
    Run ensemble pose estimation demo.
    
    Args:
        video_path: Path to RGB video file
        mobileposer_weights: Path to MobilePoser model weights
        slam_type: Type of SLAM system to use
        fusion_method: Pose fusion strategy
        max_frames: Maximum number of frames to process
    """
    print(f"Starting ensemble demo with video: {video_path}")
    print(f"SLAM type: {slam_type}, Fusion method: {fusion_method}")
    
    # Load RGB video
    try:
        rgb_frames, rgb_timestamps = load_nymeria_rgb_video(video_path)
    except Exception as e:
        print(f"Error loading video: {e}")
        return
        
    # Limit frames for demo
    if len(rgb_frames) > max_frames:
        rgb_frames = rgb_frames[:max_frames]
        rgb_timestamps = rgb_timestamps[:max_frames]
        
    # Generate mock IMU data (in practice, this would come from real sensors)
    imu_data_list, imu_timestamps = generate_mock_imu_data(len(rgb_frames))
    
    # Initialize ensemble system
    try:
        ensemble = EnsemblePoseEstimator(
            mobileposer_weights=mobileposer_weights,
            slam_type=slam_type,
            fusion_method=fusion_method,
            sync_tolerance=0.1  # 100ms tolerance
        )
    except Exception as e:
        print(f"Error initializing ensemble: {e}")
        return
    
    # Start processing
    ensemble.start_processing()
    
    # Feed data to ensemble system
    print("Processing frames...")
    start_time = time.time()
    
    for i, (rgb_frame, rgb_ts, imu_data, imu_ts) in enumerate(
        zip(rgb_frames, rgb_timestamps, imu_data_list, imu_timestamps)):
        
        # Add data to processing queues
        ensemble.process_rgb_frame(rgb_frame, rgb_ts)
        ensemble.process_imu_data(imu_data, imu_ts)
        
        # Small delay to simulate real-time processing
        time.sleep(0.01)
        
        # Print progress
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(rgb_frames)} frames")
            
    # Wait a bit for processing to complete
    time.sleep(2.0)
    
    # Collect results
    poses = ensemble.get_all_poses()
    processing_time = time.time() - start_time
    
    # Stop ensemble
    ensemble.stop_processing()
    
    # Print results
    print(f"\nProcessing completed in {processing_time:.2f}s")
    print(f"Total poses generated: {len(poses)}")
    
    if poses:
        # Analyze pose sources
        imu_poses = [p for p in poses if p.source == "imu"]
        visual_poses = [p for p in poses if p.source == "visual"]
        fused_poses = [p for p in poses if p.source == "fused"]
        
        print(f"IMU-only poses: {len(imu_poses)}")
        print(f"Visual-only poses: {len(visual_poses)}")
        print(f"Fused poses: {len(fused_poses)}")
        
        # Calculate average confidence
        avg_confidence = np.mean([p.confidence for p in poses])
        print(f"Average confidence: {avg_confidence:.3f}")
        
        # Show trajectory
        visualize_trajectory(poses)
        
    # Show performance stats
    stats = ensemble.get_performance_stats()
    print(f"\nPerformance Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


def visualize_trajectory(poses: List[PoseEstimate]):
    """
    Visualize the estimated trajectory.
    
    Args:
        poses: List of pose estimates
    """
    if not poses:
        return
        
    # Extract translations
    translations = np.array([p.translation for p in poses])
    timestamps = np.array([p.timestamp for p in poses])
    confidences = np.array([p.confidence for p in poses])
    sources = [p.source for p in poses]
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Ensemble Pose Estimation Results')
    
    # 3D trajectory
    ax = axes[0, 0]
    scatter = ax.scatter(translations[:, 0], translations[:, 2], c=confidences, 
                        cmap='viridis', alpha=0.7)
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Z (meters)')
    ax.set_title('2D Trajectory (Top View)')
    plt.colorbar(scatter, ax=ax, label='Confidence')
    
    # Position over time
    ax = axes[0, 1]
    ax.plot(timestamps, translations[:, 0], label='X', alpha=0.7)
    ax.plot(timestamps, translations[:, 1], label='Y', alpha=0.7)
    ax.plot(timestamps, translations[:, 2], label='Z', alpha=0.7)
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Position (meters)')
    ax.set_title('Position vs Time')
    ax.legend()
    ax.grid(True)
    
    # Confidence over time
    ax = axes[1, 0]
    colors = {'imu': 'blue', 'visual': 'red', 'fused': 'green'}
    for source in set(sources):
        mask = np.array([s == source for s in sources])
        if np.any(mask):
            ax.scatter(timestamps[mask], confidences[mask], 
                      label=source, color=colors.get(source, 'gray'), alpha=0.7)
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Confidence')
    ax.set_title('Confidence vs Time')
    ax.legend()
    ax.grid(True)
    
    # Source distribution
    ax = axes[1, 1]
    source_counts = {source: sources.count(source) for source in set(sources)}
    ax.pie(source_counts.values(), labels=source_counts.keys(), autopct='%1.1f%%')
    ax.set_title('Pose Source Distribution')
    
    plt.tight_layout()
    plt.show()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Ensemble pose estimation demo")
    parser.add_argument('--video', type=str, required=True,
                       help='Path to RGB video file (e.g., video_main_rgb.mp4)')
    parser.add_argument('--weights', type=str, default=str(paths.weights_file),
                       help='Path to MobilePoser weights')
    parser.add_argument('--slam-type', type=str, default='mock',
                       choices=['mock', 'orb_slam3'],
                       help='Type of SLAM system to use')
    parser.add_argument('--fusion-method', type=str, default='weighted_average',
                       choices=['weighted_average', 'kalman', 'confidence_based'],
                       help='Pose fusion method')
    parser.add_argument('--max-frames', type=int, default=300,
                       help='Maximum number of frames to process')
    
    args = parser.parse_args()
    
    # Check if video file exists
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        print("Please provide a valid path to a Nymeria RGB video file.")
        print("Example: /mnt/nas2/naoto/nymeria_dataset/data_video_main_rgb/20231031_s0_jason_brown_act0_zb36n0/video_main_rgb.mp4")
        return
        
    # Check if weights file exists
    if not os.path.exists(args.weights):
        print(f"Error: MobilePoser weights file not found: {args.weights}")
        print("Please train MobilePoser first or provide a valid weights file path.")
        return
    
    # Run demo
    run_ensemble_demo(
        video_path=args.video,
        mobileposer_weights=args.weights,
        slam_type=args.slam_type,
        fusion_method=args.fusion_method,
        max_frames=args.max_frames
    )


if __name__ == "__main__":
    main()