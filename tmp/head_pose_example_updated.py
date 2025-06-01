#!/usr/bin/env python3
"""
Example script demonstrating head pose ensemble estimation.
Combines MobilePoser head IMU with Visual-Inertial SLAM for improved
head position and orientation estimation.

This updated version uses the slam_selector for better SLAM type management.
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
import logging

from mobileposer.head_pose_ensemble import HeadPoseEnsemble, HeadPoseData
from mobileposer.slam_selector import slam_selector, create_slam_with_fallback
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
        # Try alternative path
        alt_path = Path("/mnt/nas2/naoto/nymeria_dataset/data_video_main_rgb") / sequence_dir.name / "video_main_rgb.mp4"
        if alt_path.exists():
            video_path = alt_path
        else:
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
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frames.append(rgb_frame)
        frame_count += 1
    
    cap.release()
    
    # Generate timestamps
    timestamps = [i / fps for i in range(frame_count)]
    
    # Load IMU data (placeholder - in practice, load from actual dataset)
    # For demo purposes, generate synthetic IMU data
    imu_data_list = []
    for i in range(frame_count):
        # Generate 6 sensors x 12 values = 72 values
        imu_data = np.random.randn(72) * 0.1
        imu_data_list.append(imu_data)
    
    print(f"Loaded {len(rgb_frames)} RGB frames and {len(imu_data_list)} IMU samples")
    
    return rgb_frames, imu_data_list, timestamps


def visualize_head_poses(head_poses: List[HeadPoseData], title: str = "Head Pose Trajectory"):
    """
    Visualize head pose trajectory in 3D.
    
    Args:
        head_poses: List of head pose estimates
        title: Plot title
    """
    if not head_poses:
        print("No head poses to visualize")
        return
    
    # Extract positions
    positions = np.array([pose.position for pose in head_poses if pose is not None])
    
    if len(positions) == 0:
        print("No valid positions to visualize")
        return
    
    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot trajectory
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=2)
    ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], c='g', s=100, label='Start')
    ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], c='r', s=100, label='End')
    
    # Color code by source
    sources = [pose.source for pose in head_poses if pose is not None]
    colors = {'imu': 'blue', 'vi_slam': 'red', 'fused': 'green'}
    
    for i, (pos, src) in enumerate(zip(positions, sources)):
        if i % 10 == 0:  # Plot every 10th point to avoid clutter
            ax.scatter(pos[0], pos[1], pos[2], c=colors.get(src, 'gray'), s=20, alpha=0.5)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)
    ax.legend()
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Head pose ensemble estimation example')
    parser.add_argument('--sequence', type=str, required=True,
                       help='Path to Nymeria sequence directory')
    parser.add_argument('--weights', type=str, required=True,
                       help='Path to MobilePoser weights')
    parser.add_argument('--slam-type', type=str, default='adaptive',
                       choices=slam_selector.available_types + ['mock_vi'],
                       help='Type of SLAM to use')
    parser.add_argument('--fusion-method', type=str, default='weighted_average',
                       choices=['weighted_average', 'confidence_based'],
                       help='Fusion method for combining estimates')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Maximum number of frames to process')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize head pose trajectory')
    parser.add_argument('--allow-mock', action='store_true',
                       help='Allow fallback to mock SLAM if real SLAM unavailable')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Print SLAM status
    print("\n" + "="*60)
    slam_selector.print_status()
    print("="*60 + "\n")
    
    # Check if requested SLAM type is available
    if args.slam_type not in slam_selector.available_types and args.slam_type != 'mock_vi':
        if args.allow_mock:
            print(f"WARNING: {args.slam_type} not available, will fall back to mock")
        else:
            print(f"ERROR: {args.slam_type} not available and --allow-mock not specified")
            print("Available types:", slam_selector.available_types)
            return 1
    
    # Load data
    print("Loading Nymeria data...")
    try:
        rgb_frames, imu_data_list, timestamps = load_nymeria_data(args.sequence)
    except Exception as e:
        print(f"Error loading data: {e}")
        return 1
    
    # Limit frames if specified
    if args.max_frames:
        rgb_frames = rgb_frames[:args.max_frames]
        imu_data_list = imu_data_list[:args.max_frames]
        timestamps = timestamps[:args.max_frames]
    
    # Initialize head pose ensemble
    print("Initializing head pose ensemble...")
    try:
        # Handle special case for mock_vi which is in head_pose_ensemble
        if args.slam_type == 'mock_vi':
            ensemble = HeadPoseEnsemble(
                mobileposer_weights=args.weights,
                slam_type='mock_vi',
                fusion_method=args.fusion_method
            )
        else:
            # Use slam_selector for other types
            if args.allow_mock:
                # This will fall back to mock if real SLAM unavailable
                ensemble = HeadPoseEnsemble(
                    mobileposer_weights=args.weights,
                    slam_type=args.slam_type,
                    fusion_method=args.fusion_method
                )
            else:
                # This will raise error if real SLAM unavailable
                ensemble = HeadPoseEnsemble(
                    mobileposer_weights=args.weights,
                    slam_type=args.slam_type,
                    fusion_method=args.fusion_method
                )
    except Exception as e:
        print(f"Error initializing ensemble: {e}")
        if not args.allow_mock:
            print("Tip: Use --allow-mock to fall back to mock SLAM for testing")
        return 1
    
    # Process frames
    print(f"Processing {len(rgb_frames)} frames...")
    head_poses = []
    processing_times = []
    
    for i, (rgb_frame, imu_data, timestamp) in enumerate(zip(rgb_frames, imu_data_list, timestamps)):
        if i % 10 == 0:
            print(f"Processing frame {i}/{len(rgb_frames)}")
        
        start_time = time.time()
        
        # Process frame
        try:
            head_pose = ensemble.process_frame(rgb_frame, imu_data, timestamp)
            head_poses.append(head_pose)
        except Exception as e:
            print(f"Error processing frame {i}: {e}")
            head_poses.append(None)
        
        processing_times.append(time.time() - start_time)
    
    # Print statistics
    print("\n" + "="*60)
    print("Processing Statistics:")
    print("="*60)
    print(f"Total frames: {len(rgb_frames)}")
    print(f"Successful poses: {sum(1 for p in head_poses if p is not None)}")
    print(f"Average processing time: {np.mean(processing_times):.3f} seconds")
    print(f"FPS: {1.0 / np.mean(processing_times):.1f}")
    
    # Get ensemble performance stats
    perf_stats = ensemble.get_performance_stats()
    print(f"\nEnsemble Statistics:")
    for key, value in perf_stats.items():
        print(f"  {key}: {value}")
    
    # Visualize if requested
    if args.visualize:
        print("\nVisualizing head pose trajectory...")
        visualize_head_poses(head_poses, title=f"Head Pose Trajectory - {args.slam_type}")
    
    print("\nDone!")
    return 0


if __name__ == '__main__':
    exit(main())