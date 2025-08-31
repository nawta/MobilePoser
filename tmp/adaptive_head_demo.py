#!/usr/bin/env python3
"""
Demonstration script for the adaptive head pose ensemble system.
This system automatically adapts between different SLAM modes based on available data:
- RGB + Head IMU -> Visual-Inertial SLAM
- RGB only -> Monocular SLAM
- No RGB -> IMU-only mode

The ensemble uses dynamic weight calculation and temporal feedback.
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

from mobileposer.adaptive_head_ensemble import (
    AdaptiveHeadPoseEnsemble, AdaptiveEnsembleInput, AdaptiveEnsembleOutput
)
from mobileposer.models.adaptive_slam import SlamMode
from mobileposer.config import paths


def load_nymeria_sequence_data(sequence_path: str, max_frames: int = 300) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
    """
    Load RGB video and generate corresponding IMU data from Nymeria sequence.
    
    Args:
        sequence_path: Path to Nymeria sequence directory
        max_frames: Maximum number of frames to load
        
    Returns:
        Tuple of (rgb_frames, imu_data_list, timestamps)
    """
    sequence_dir = Path(sequence_path)
    
    # Load RGB video
    video_path = sequence_dir / "video_main_rgb.mp4"
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    rgb_frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    frame_count = 0
    while len(rgb_frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frames.append(frame_rgb)
        frame_count += 1
    
    cap.release()
    
    # Generate timestamps
    timestamps = [i / fps for i in range(len(rgb_frames))]
    
    # Generate realistic IMU data for demonstration
    imu_data_list = generate_realistic_head_motion_imu_data(len(rgb_frames), fps)
    
    print(f"Loaded {len(rgb_frames)} RGB frames")
    print(f"Generated {len(imu_data_list)} IMU samples")
    print(f"Video FPS: {fps:.2f}, Duration: {timestamps[-1]:.2f}s")
    
    return rgb_frames, imu_data_list, timestamps


def generate_realistic_head_motion_imu_data(num_frames: int, fps: float) -> List[np.ndarray]:
    """
    Generate realistic IMU data with head motion patterns.
    Simulates natural head movements like nodding, turning, and walking motion.
    
    Args:
        num_frames: Number of frames to generate
        fps: Frame rate
        
    Returns:
        List of IMU data arrays for full sensor configuration
    """
    imu_data_list = []
    
    for i in range(num_frames):
        t = i / fps
        
        # Simulate 6 IMU sensors (Nymeria configuration)
        imu_frame = np.zeros((6, 12))  # 6 sensors, 12 values each
        
        for sensor_idx in range(6):
            if sensor_idx == 4:  # Head sensor (index 4)
                # Complex head motion patterns
                
                # Walking motion (vertical oscillation)
                walk_freq = 1.8  # Steps per second
                walk_motion = 0.2 * np.sin(2 * np.pi * walk_freq * t)
                
                # Head turning (looking around)
                turn_freq = 0.3  # Slow turning
                head_turn = 0.4 * np.sin(2 * np.pi * turn_freq * t)
                
                # Nodding motion
                nod_freq = 0.7
                head_nod = 0.2 * np.sin(2 * np.pi * nod_freq * t)
                
                # Random small movements
                noise_amplitude = 0.05
                random_motion = noise_amplitude * np.random.randn(3)
                
                # Head acceleration
                acc = np.array([
                    0.1 * np.sin(t * 2) + random_motion[0],           # x: lateral + noise
                    -9.81 + walk_motion + random_motion[1],           # y: gravity + walking + noise
                    0.1 * np.cos(t * 2.5) + random_motion[2]          # z: forward/back + noise
                ])
                
                # Head rotation matrix (combination of turning and nodding)
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
                
                # Small random rotations
                small_angle = 0.02 * np.random.randn()
                R_noise = np.array([
                    [np.cos(small_angle), -np.sin(small_angle), 0],
                    [np.sin(small_angle), np.cos(small_angle), 0],
                    [0, 0, 1]
                ])
                
                # Combined rotation
                R_combined = R_turn @ R_nod @ R_noise
                rot_matrix = R_combined.flatten()
                
            else:  # Other sensors (arms, legs, torso)
                # Simpler motion for other body parts
                motion_phase = t + sensor_idx * 0.7
                
                # Walking affects all body parts
                walk_influence = 0.1 * np.sin(2 * np.pi * 1.8 * t + sensor_idx)
                
                acc = np.array([
                    0.05 * np.sin(motion_phase * 2) + walk_influence,
                    -9.81 + 0.1 * np.cos(motion_phase) + walk_influence,
                    0.05 * np.cos(motion_phase * 1.5) + walk_influence
                ])
                
                # Simple rotation
                angle = 0.05 * np.sin(motion_phase)
                rot_matrix = np.array([
                    [np.cos(angle), -np.sin(angle), 0],
                    [np.sin(angle), np.cos(angle), 0],
                    [0, 0, 1]
                ]).flatten()
            
            # Combine acceleration and rotation
            imu_frame[sensor_idx] = np.concatenate([acc, rot_matrix])
        
        # Flatten to match expected format
        imu_data_list.append(imu_frame.flatten())
    
    return imu_data_list


def simulate_data_dropout(rgb_frames: List[np.ndarray],
                         imu_data_list: List[np.ndarray],
                         dropout_scenarios: List[Tuple[int, int, str]]) -> Tuple[List[Optional[np.ndarray]], List[Optional[np.ndarray]]]:
    """
    Simulate realistic data dropout scenarios to test adaptive behavior.
    
    Args:
        rgb_frames: Original RGB frames
        imu_data_list: Original IMU data
        dropout_scenarios: List of (start_frame, end_frame, dropout_type)
                          dropout_type: "rgb", "imu", "head_imu", "both"
        
    Returns:
        Modified frames and IMU data with dropouts
    """
    modified_rgb = rgb_frames.copy()
    modified_imu = imu_data_list.copy()
    
    for start_frame, end_frame, dropout_type in dropout_scenarios:
        print(f"Simulating {dropout_type} dropout from frame {start_frame} to {end_frame}")
        
        for frame_idx in range(start_frame, min(end_frame, len(modified_rgb))):
            if dropout_type == "rgb" or dropout_type == "both":
                modified_rgb[frame_idx] = None
                
            if dropout_type == "imu" or dropout_type == "both":
                modified_imu[frame_idx] = None
                
            elif dropout_type == "head_imu":
                # Zero out head IMU data (keep other sensors)
                if modified_imu[frame_idx] is not None:
                    imu_data = modified_imu[frame_idx].copy()
                    # Zero head sensor data (sensor index 4, positions 48-59 in flattened array)
                    imu_data[48:60] = 0.0
                    modified_imu[frame_idx] = imu_data
    
    return modified_rgb, modified_imu


def run_adaptive_ensemble_demo(sequence_path: str,
                              mobileposer_weights: str,
                              max_frames: int = 300,
                              simulate_dropouts: bool = True,
                              head_imu_index: int = 4):
    """
    Run comprehensive adaptive ensemble demonstration.
    
    Args:
        sequence_path: Path to Nymeria sequence
        mobileposer_weights: Path to MobilePoser weights
        max_frames: Maximum frames to process
        simulate_dropouts: Whether to simulate data dropouts
        head_imu_index: Index of head IMU sensor
    """
    print(f"Starting adaptive head pose ensemble demo")
    print(f"Sequence: {sequence_path}")
    print(f"Max frames: {max_frames}")
    
    # Load data
    try:
        rgb_frames, imu_data_list, timestamps = load_nymeria_sequence_data(
            sequence_path, max_frames
        )
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Simulate data dropouts for testing adaptive behavior
    if simulate_dropouts:
        dropout_scenarios = [
            (50, 80, "rgb"),        # RGB dropout -> should use IMU only
            (120, 150, "head_imu"), # Head IMU dropout -> should use Monocular SLAM
            (200, 230, "both"),     # Both dropout -> should handle gracefully
        ]
        rgb_frames, imu_data_list = simulate_data_dropout(
            rgb_frames, imu_data_list, dropout_scenarios
        )
    
    # Initialize adaptive ensemble
    try:
        ensemble = AdaptiveHeadPoseEnsemble(
            mobileposer_weights=mobileposer_weights,
            head_imu_index=head_imu_index,
            enable_temporal_feedback=True
        )
        
        if not ensemble.initialize():
            print("Failed to initialize ensemble")
            return
            
    except Exception as e:
        print(f"Error initializing ensemble: {e}")
        return
    
    # Process frames
    print("Processing frames with adaptive ensemble...")
    outputs = []
    processing_times = []
    
    for i, (rgb_frame, imu_data, timestamp) in enumerate(
        zip(rgb_frames, imu_data_list, timestamps)):
        
        # Create input
        ensemble_input = AdaptiveEnsembleInput(
            rgb_frame=rgb_frame,
            full_imu_data=imu_data,
            timestamp=timestamp,
            frame_id=i
        )
        
        # Process frame
        start_time = time.time()
        output = ensemble.process_frame(ensemble_input)
        processing_time = time.time() - start_time
        
        outputs.append(output)
        processing_times.append(processing_time)
        
        # Progress reporting
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(rgb_frames)} frames")
            print(f"  Current mode: {output.mode_used.value}")
            print(f"  Ensemble weights: IMU={output.ensemble_weights[0]:.3f}, SLAM={output.ensemble_weights[1]:.3f}")
            if output.head_pose:
                print(f"  Head pose confidence: {output.head_pose.confidence:.3f}")
    
    # Analyze results
    print(f"\nProcessing completed!")
    analyze_adaptive_ensemble_results(outputs, processing_times, simulate_dropouts)
    
    # Visualize results
    visualize_adaptive_ensemble_results(outputs, timestamps, simulate_dropouts)
    
    # Show performance stats
    stats = ensemble.get_performance_stats()
    print(f"\nPerformance Statistics:")
    for key, value in stats.items():
        if key != 'processing_times':  # Skip the large array
            print(f"  {key}: {value}")


def analyze_adaptive_ensemble_results(outputs: List[AdaptiveEnsembleOutput],
                                     processing_times: List[float],
                                     had_dropouts: bool = False):
    """Analyze the results of adaptive ensemble processing."""
    
    print(f"=== Adaptive Ensemble Analysis ===")
    
    # Basic statistics
    total_frames = len(outputs)
    successful_poses = len([o for o in outputs if o.head_pose is not None])
    
    print(f"Total frames processed: {total_frames}")
    print(f"Successful pose estimates: {successful_poses} ({successful_poses/total_frames*100:.1f}%)")
    print(f"Average processing time: {np.mean(processing_times):.4f}s")
    print(f"Processing FPS: {1.0/np.mean(processing_times):.1f}")
    
    # Mode distribution
    mode_counts = {}
    for output in outputs:
        mode = output.mode_used.value
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
    
    print(f"\nSLAM Mode Distribution:")
    for mode, count in mode_counts.items():
        print(f"  {mode}: {count} frames ({count/total_frames*100:.1f}%)")
    
    # Weight analysis
    imu_weights = [o.ensemble_weights[0] for o in outputs if o.head_pose is not None]
    slam_weights = [o.ensemble_weights[1] for o in outputs if o.head_pose is not None]
    
    if imu_weights:
        print(f"\nEnsemble Weight Analysis:")
        print(f"  Average IMU weight: {np.mean(imu_weights):.3f}")
        print(f"  Average SLAM weight: {np.mean(slam_weights):.3f}")
        print(f"  IMU weight std: {np.std(imu_weights):.3f}")
        print(f"  SLAM weight std: {np.std(slam_weights):.3f}")
    
    # Confidence analysis
    confidences = [o.head_pose.confidence for o in outputs if o.head_pose is not None]
    if confidences:
        print(f"\nConfidence Analysis:")
        print(f"  Average confidence: {np.mean(confidences):.3f}")
        print(f"  Confidence std: {np.std(confidences):.3f}")
        print(f"  Min confidence: {np.min(confidences):.3f}")
        print(f"  Max confidence: {np.max(confidences):.3f}")
    
    # Adaptive behavior analysis
    if had_dropouts:
        print(f"\nAdaptive Behavior Analysis:")
        
        # Count mode switches
        modes = [o.mode_used for o in outputs]
        mode_switches = sum(1 for i in range(1, len(modes)) if modes[i] != modes[i-1])
        print(f"  Mode switches: {mode_switches}")
        
        # Analyze response to dropouts
        none_frames = sum(1 for mode in modes if mode == SlamMode.NONE)
        mono_frames = sum(1 for mode in modes if mode == SlamMode.MONOCULAR)
        vi_frames = sum(1 for mode in modes if mode == SlamMode.VISUAL_INERTIAL)
        
        print(f"  IMU-only frames: {none_frames}")
        print(f"  Monocular SLAM frames: {mono_frames}")
        print(f"  Visual-Inertial frames: {vi_frames}")


def visualize_adaptive_ensemble_results(outputs: List[AdaptiveEnsembleOutput],
                                       timestamps: List[float],
                                       had_dropouts: bool = False):
    """Create comprehensive visualizations of the adaptive ensemble results."""
    
    # Extract data for visualization
    positions = []
    confidences = []
    imu_weights = []
    slam_weights = []
    modes = []
    
    for output in outputs:
        if output.head_pose is not None:
            positions.append(output.head_pose.translation)
            confidences.append(output.head_pose.confidence)
            imu_weights.append(output.ensemble_weights[0])
            slam_weights.append(output.ensemble_weights[1])
            modes.append(output.mode_used.value)
        else:
            positions.append(np.array([np.nan, np.nan, np.nan]))
            confidences.append(0.0)
            imu_weights.append(1.0)
            slam_weights.append(0.0)
            modes.append("none")
    
    positions = np.array(positions)
    timestamps_array = np.array(timestamps[:len(positions)])
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('Adaptive Head Pose Ensemble Results', fontsize=16)
    
    # 3D trajectory
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    valid_mask = ~np.isnan(positions[:, 0])
    if np.any(valid_mask):
        scatter = ax1.scatter(positions[valid_mask, 0], 
                            positions[valid_mask, 1], 
                            positions[valid_mask, 2],
                            c=np.array(confidences)[valid_mask], 
                            cmap='viridis', alpha=0.7)
        ax1.set_xlabel('X (meters)')
        ax1.set_ylabel('Y (meters)')
        ax1.set_zlabel('Z (meters)')
        ax1.set_title('3D Head Trajectory')
        plt.colorbar(scatter, ax=ax1, label='Confidence')
    
    # Position vs time
    ax2 = fig.add_subplot(2, 3, 2)
    if np.any(valid_mask):
        ax2.plot(timestamps_array[valid_mask], positions[valid_mask, 0], 
                label='X', alpha=0.7, linewidth=1)
        ax2.plot(timestamps_array[valid_mask], positions[valid_mask, 1], 
                label='Y', alpha=0.7, linewidth=1)
        ax2.plot(timestamps_array[valid_mask], positions[valid_mask, 2], 
                label='Z', alpha=0.7, linewidth=1)
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Position (meters)')
    ax2.set_title('Head Position vs Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Ensemble weights over time
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.fill_between(timestamps_array, 0, imu_weights, 
                     alpha=0.6, label='IMU Weight', color='blue')
    ax3.fill_between(timestamps_array, imu_weights, 
                     np.array(imu_weights) + np.array(slam_weights), 
                     alpha=0.6, label='SLAM Weight', color='red')
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('Weight')
    ax3.set_title('Ensemble Weights vs Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)
    
    # Confidence over time with mode indication
    ax4 = fig.add_subplot(2, 3, 4)
    mode_colors = {'none': 'gray', 'monocular': 'orange', 'visual_inertial': 'green'}
    
    for mode_name, color in mode_colors.items():
        mode_mask = np.array(modes) == mode_name
        if np.any(mode_mask):
            ax4.scatter(timestamps_array[mode_mask], np.array(confidences)[mode_mask],
                       label=mode_name, color=color, alpha=0.7, s=20)
    
    ax4.set_xlabel('Time (seconds)')
    ax4.set_ylabel('Confidence')
    ax4.set_title('Confidence vs Time (colored by SLAM mode)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 1)
    
    # Mode distribution pie chart
    ax5 = fig.add_subplot(2, 3, 5)
    mode_counts = {}
    for mode in modes:
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
    
    colors = [mode_colors.get(mode, 'purple') for mode in mode_counts.keys()]
    ax5.pie(mode_counts.values(), labels=mode_counts.keys(), autopct='%1.1f%%',
            colors=colors)
    ax5.set_title('SLAM Mode Distribution')
    
    # Processing performance
    ax6 = fig.add_subplot(2, 3, 6)
    processing_times = [o.processing_time for o in outputs]
    ax6.plot(timestamps_array, processing_times, alpha=0.7, linewidth=1)
    ax6.axhline(y=np.mean(processing_times), color='red', linestyle='--', 
                label=f'Mean: {np.mean(processing_times):.4f}s')
    ax6.set_xlabel('Time (seconds)')
    ax6.set_ylabel('Processing Time (seconds)')
    ax6.set_title('Processing Performance')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Additional analysis plot for dropout scenarios
    if had_dropouts:
        fig2, (ax7, ax8) = plt.subplots(2, 1, figsize=(12, 8))
        fig2.suptitle('Adaptive Behavior During Data Dropouts', fontsize=14)
        
        # Mode switches over time
        mode_numeric = []
        mode_map = {'none': 0, 'monocular': 1, 'visual_inertial': 2}
        for mode in modes:
            mode_numeric.append(mode_map.get(mode, 0))
        
        ax7.plot(timestamps_array, mode_numeric, 'o-', markersize=3, linewidth=1)
        ax7.set_ylabel('SLAM Mode')
        ax7.set_title('SLAM Mode Adaptation Over Time')
        ax7.set_yticks([0, 1, 2])
        ax7.set_yticklabels(['None', 'Monocular', 'Visual-Inertial'])
        ax7.grid(True, alpha=0.3)
        
        # Weight adaptation
        ax8.plot(timestamps_array, imu_weights, label='IMU Weight', alpha=0.8)
        ax8.plot(timestamps_array, slam_weights, label='SLAM Weight', alpha=0.8)
        ax8.set_xlabel('Time (seconds)')
        ax8.set_ylabel('Ensemble Weight')
        ax8.set_title('Dynamic Weight Adaptation')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
        ax8.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.show()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Adaptive head pose ensemble demo")
    parser.add_argument('--sequence', type=str, required=True,
                       help='Path to Nymeria sequence directory')
    parser.add_argument('--weights', type=str, default=str(paths.weights_file),
                       help='Path to MobilePoser weights')
    parser.add_argument('--max-frames', type=int, default=300,
                       help='Maximum frames to process')
    parser.add_argument('--head-imu-index', type=int, default=4,
                       help='Index of head IMU sensor')
    parser.add_argument('--no-dropouts', action='store_true',
                       help='Disable data dropout simulation')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.sequence):
        print(f"Error: Sequence directory not found: {args.sequence}")
        print("Please provide a valid Nymeria sequence directory.")
        return
    
    if not os.path.exists(args.weights):
        print(f"Error: MobilePoser weights not found: {args.weights}")
        return
    
    # Run demo
    run_adaptive_ensemble_demo(
        sequence_path=args.sequence,
        mobileposer_weights=args.weights,
        max_frames=args.max_frames,
        simulate_dropouts=not args.no_dropouts,
        head_imu_index=args.head_imu_index
    )


if __name__ == "__main__":
    main()