#!/usr/bin/env python3
"""
SLAM Processing Example - Visual demonstration of inputs and outputs
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches

def create_slam_example_visualization():
    """Create a comprehensive visualization of SLAM inputs and outputs."""
    
    fig = plt.figure(figsize=(16, 10))
    
    # Create grid layout
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Title
    fig.suptitle('SLAM Processing Example: Inputs → Processing → Outputs', fontsize=18, fontweight='bold')
    
    # 1. Input RGB Frame (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    create_sample_rgb_frame(ax1)
    ax1.set_title('Input: RGB Frame', fontsize=12, fontweight='bold')
    
    # 2. Input IMU Data (top middle-left)
    ax2 = fig.add_subplot(gs[0, 1])
    create_imu_visualization(ax2)
    ax2.set_title('Input: Head IMU Data', fontsize=12, fontweight='bold')
    
    # 3. Feature Detection (top middle-right)
    ax3 = fig.add_subplot(gs[0, 2])
    create_feature_detection_viz(ax3)
    ax3.set_title('Processing: ORB Features', fontsize=12, fontweight='bold')
    
    # 4. Tracking State (top right)
    ax4 = fig.add_subplot(gs[0, 3])
    create_tracking_state_viz(ax4)
    ax4.set_title('Processing: SLAM State', fontsize=12, fontweight='bold')
    
    # 5. Output Pose Matrix (middle left)
    ax5 = fig.add_subplot(gs[1, 0:2])
    create_pose_matrix_viz(ax5)
    ax5.set_title('Output: 4x4 Pose Matrix', fontsize=12, fontweight='bold')
    
    # 6. 3D Trajectory (middle right)
    ax6 = fig.add_subplot(gs[1, 2:], projection='3d')
    create_trajectory_viz(ax6)
    ax6.set_title('Output: Camera Trajectory', fontsize=12, fontweight='bold')
    
    # 7. Confidence & Weights (bottom left)
    ax7 = fig.add_subplot(gs[2, 0:2])
    create_confidence_viz(ax7)
    ax7.set_title('Output: Confidence & Ensemble Weights', fontsize=12, fontweight='bold')
    
    # 8. Final Head Pose (bottom right)
    ax8 = fig.add_subplot(gs[2, 2:], projection='3d')
    create_head_pose_viz(ax8)
    ax8.set_title('Output: Final Head Pose (6DOF)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    return fig

def create_sample_rgb_frame(ax):
    """Create a sample RGB frame visualization."""
    # Create synthetic indoor scene
    img = np.ones((240, 320, 3), dtype=np.uint8) * 200
    
    # Add floor grid
    for i in range(0, 240, 20):
        cv2.line(img, (0, i), (320, i), (150, 150, 150), 1)
    for j in range(0, 320, 20):
        cv2.line(img, (j, 0), (j, 240), (150, 150, 150), 1)
    
    # Add some "furniture"
    cv2.rectangle(img, (50, 80), (120, 160), (100, 150, 200), -1)  # Blue object
    cv2.rectangle(img, (200, 100), (280, 180), (200, 150, 100), -1)  # Brown object
    
    # Add text/markers
    cv2.putText(img, "Nymeria RGB", (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(img, "640x480 @ 15FPS", (200, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    # Add feature points preview
    np.random.seed(42)
    for _ in range(30):
        x, y = np.random.randint(10, 310), np.random.randint(10, 230)
        cv2.circle(img, (x, y), 2, (0, 255, 0), -1)
    
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.axis('off')
    
    # Add annotations
    ax.text(0.5, -0.05, '15 FPS RGB from head-mounted camera', 
            ha='center', transform=ax.transAxes, fontsize=9, style='italic')

def create_imu_visualization(ax):
    """Create IMU data visualization."""
    t = np.linspace(0, 1, 100)
    
    # Simulated IMU data
    acc_x = 0.1 * np.sin(2 * np.pi * t) + np.random.normal(0, 0.02, 100)
    acc_y = 0.05 * np.cos(2 * np.pi * t) + np.random.normal(0, 0.02, 100)
    acc_z = 9.81 + 0.1 * np.sin(4 * np.pi * t) + np.random.normal(0, 0.02, 100)
    
    gyro_x = 0.5 * np.sin(3 * np.pi * t) + np.random.normal(0, 0.05, 100)
    gyro_y = 0.3 * np.cos(2 * np.pi * t) + np.random.normal(0, 0.05, 100)
    gyro_z = 0.2 * np.sin(np.pi * t) + np.random.normal(0, 0.05, 100)
    
    # Plot accelerometer
    ax.plot(t, acc_x, 'r-', label='Acc X', alpha=0.7, linewidth=1)
    ax.plot(t, acc_y, 'g-', label='Acc Y', alpha=0.7, linewidth=1)
    ax.plot(t, acc_z/10, 'b-', label='Acc Z/10', alpha=0.7, linewidth=1)
    
    # Plot gyroscope
    ax.plot(t, gyro_x, 'r--', label='Gyro X', alpha=0.5, linewidth=1)
    ax.plot(t, gyro_y, 'g--', label='Gyro Y', alpha=0.5, linewidth=1)
    ax.plot(t, gyro_z, 'b--', label='Gyro Z', alpha=0.5, linewidth=1)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Sensor Value')
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1, 1.5)
    
    # Add annotation
    ax.text(0.5, -0.15, '30 FPS IMU (6-axis: 3 acc + 3 gyro)', 
            ha='center', transform=ax.transAxes, fontsize=9, style='italic')

def create_feature_detection_viz(ax):
    """Show ORB feature detection visualization."""
    # Create image with detected features
    img = np.ones((240, 320, 3), dtype=np.uint8) * 220
    
    # Add some structure
    cv2.rectangle(img, (50, 80), (120, 160), (180, 180, 180), -1)
    cv2.rectangle(img, (200, 100), (280, 180), (180, 180, 180), -1)
    
    # Add ORB keypoints
    np.random.seed(42)
    keypoints = []
    for _ in range(150):  # ORB features
        x, y = np.random.randint(10, 310), np.random.randint(10, 230)
        size = np.random.randint(3, 8)
        cv2.circle(img, (x, y), size, (0, 255, 0), 1)
        cv2.circle(img, (x, y), 1, (0, 255, 0), -1)
        
        # Add orientation line for some keypoints
        if np.random.rand() > 0.7:
            angle = np.random.rand() * 2 * np.pi
            x2 = int(x + size * np.cos(angle))
            y2 = int(y + size * np.sin(angle))
            cv2.line(img, (x, y), (x2, y2), (0, 200, 0), 1)
    
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.axis('off')
    
    # Add statistics
    ax.text(0.02, 0.98, 'ORB Features: 150\nMatched: 87\nInliers: 72', 
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

def create_tracking_state_viz(ax):
    """Show SLAM tracking state."""
    ax.axis('off')
    
    # State diagram
    states = [
        ('NOT_INITIALIZED', 0.5, 0.8, 'lightgray'),
        ('INITIALIZING', 0.5, 0.6, 'yellow'),
        ('TRACKING', 0.5, 0.4, 'lightgreen'),
        ('LOST', 0.5, 0.2, 'lightcoral')
    ]
    
    current_state = 'TRACKING'
    
    for state, x, y, color in states:
        if state == current_state:
            ax.add_patch(plt.Rectangle((x-0.15, y-0.05), 0.3, 0.1, 
                                     facecolor=color, edgecolor='black', linewidth=2))
            ax.text(x, y, state, ha='center', va='center', fontweight='bold')
        else:
            ax.add_patch(plt.Rectangle((x-0.15, y-0.05), 0.3, 0.1, 
                                     facecolor=color, edgecolor='gray', alpha=0.3))
            ax.text(x, y, state, ha='center', va='center', alpha=0.5)
    
    # Add arrows
    arrow_props = dict(arrowstyle='->', lw=1.5, color='gray')
    ax.annotate('', xy=(0.5, 0.75), xytext=(0.5, 0.85), arrowprops=arrow_props)
    ax.annotate('', xy=(0.5, 0.55), xytext=(0.5, 0.65), arrowprops=arrow_props)
    ax.annotate('', xy=(0.5, 0.35), xytext=(0.5, 0.45), arrowprops=arrow_props)
    
    # Add mode indicator
    ax.text(0.5, 0.05, 'Mode: VISUAL_INERTIAL', ha='center', 
            fontsize=10, fontweight='bold')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

def create_pose_matrix_viz(ax):
    """Show the 4x4 transformation matrix output."""
    # Example pose matrix
    pose_matrix = np.array([
        [0.9950, -0.0998,  0.0052,  0.2341],
        [0.0997,  0.9949,  0.0174, -0.1532],
        [-0.0070, -0.0168,  0.9998,  0.0823],
        [0.0000,  0.0000,  0.0000,  1.0000]
    ])
    
    # Create table
    cell_text = []
    for i in range(4):
        row = []
        for j in range(4):
            if i < 3 and j < 3:  # Rotation part
                row.append(f'{pose_matrix[i,j]:.4f}')
            elif i < 3 and j == 3:  # Translation part
                row.append(f'{pose_matrix[i,j]:.4f}')
            else:  # Last row
                row.append(f'{pose_matrix[i,j]:.0f}')
        cell_text.append(row)
    
    # Color code the matrix
    colors = []
    for i in range(4):
        row_colors = []
        for j in range(4):
            if i < 3 and j < 3:  # Rotation
                row_colors.append('#E8F4FD')
            elif i < 3 and j == 3:  # Translation
                row_colors.append('#FFF4E6')
            else:  # Homogeneous
                row_colors.append('#F5F5F5')
        colors.append(row_colors)
    
    table = ax.table(cellText=cell_text, cellColours=colors,
                     colLabels=['', '', '', ''],
                     rowLabels=['', '', '', ''],
                     cellLoc='center',
                     loc='center',
                     bbox=[0.1, 0.2, 0.8, 0.6])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    # Add labels
    ax.text(0.25, 0.85, 'Rotation (3x3)', ha='center', fontsize=10, 
            color='blue', fontweight='bold')
    ax.text(0.75, 0.85, 'Translation (3x1)', ha='center', fontsize=10, 
            color='orange', fontweight='bold')
    ax.text(0.5, 0.1, 'SE(3) Transformation Matrix', ha='center', fontsize=11,
            style='italic')
    
    # Add interpretation
    ax.text(0.5, -0.05, 'Position: (0.234, -0.153, 0.082)m | Rotation: ~5.7°', 
            ha='center', transform=ax.transAxes, fontsize=9)
    
    ax.axis('off')

def create_trajectory_viz(ax):
    """Create 3D trajectory visualization."""
    # Generate sample trajectory
    t = np.linspace(0, 4*np.pi, 100)
    x = 2 * np.cos(t) + 0.1 * np.random.randn(100)
    y = 2 * np.sin(t) + 0.1 * np.random.randn(100)
    z = 0.5 * t / (4*np.pi) + 0.05 * np.random.randn(100)
    
    # Plot trajectory
    ax.plot(x, y, z, 'b-', linewidth=2, label='SLAM trajectory')
    
    # Add start and end markers
    ax.scatter(x[0], y[0], z[0], c='green', s=100, marker='o', label='Start')
    ax.scatter(x[-1], y[-1], z[-1], c='red', s=100, marker='s', label='Current')
    
    # Add some keyframes
    keyframe_idx = [20, 40, 60, 80]
    ax.scatter(x[keyframe_idx], y[keyframe_idx], z[keyframe_idx], 
              c='yellow', s=50, marker='^', label='Keyframes')
    
    # Add coordinate frame at current position
    pos = np.array([x[-1], y[-1], z[-1]])
    ax.quiver(pos[0], pos[1], pos[2], 0.5, 0, 0, color='red', arrow_length_ratio=0.2)
    ax.quiver(pos[0], pos[1], pos[2], 0, 0.5, 0, color='green', arrow_length_ratio=0.2)
    ax.quiver(pos[0], pos[1], pos[2], 0, 0, 0.5, color='blue', arrow_length_ratio=0.2)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_box_aspect([1,1,0.5])
    
    # Set view angle
    ax.view_init(elev=20, azim=45)

def create_confidence_viz(ax):
    """Create confidence and weight visualization."""
    frames = np.arange(0, 50)
    
    # Simulated confidence scores
    slam_conf = 0.2 + 0.6 * (1 - np.exp(-frames/10)) + 0.1 * np.random.randn(50)
    slam_conf = np.clip(slam_conf, 0, 1)
    imu_conf = 0.8 + 0.1 * np.sin(frames/5) + 0.05 * np.random.randn(50)
    imu_conf = np.clip(imu_conf, 0, 1)
    
    # Calculate ensemble weights
    slam_weight = slam_conf / (slam_conf + imu_conf)
    imu_weight = imu_conf / (slam_conf + imu_conf)
    
    # Plot confidences
    ax2 = ax.twinx()
    
    l1 = ax.plot(frames, slam_conf, 'b-', linewidth=2, label='SLAM Confidence')
    l2 = ax.plot(frames, imu_conf, 'r-', linewidth=2, label='IMU Confidence')
    
    l3 = ax2.plot(frames, slam_weight, 'b--', linewidth=1.5, alpha=0.7, label='SLAM Weight')
    l4 = ax2.plot(frames, imu_weight, 'r--', linewidth=1.5, alpha=0.7, label='IMU Weight')
    
    ax.set_xlabel('Frame Number')
    ax.set_ylabel('Confidence Score', color='black')
    ax2.set_ylabel('Ensemble Weight', color='gray')
    
    ax.set_ylim(0, 1.1)
    ax2.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)
    
    # Combine legends
    lns = l1 + l2 + l3 + l4
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc='right', fontsize=9)
    
    # Add phases
    ax.axvspan(0, 10, alpha=0.2, color='gray', label='Init')
    ax.axvspan(10, 40, alpha=0.2, color='green', label='Tracking')
    ax.axvspan(40, 50, alpha=0.2, color='yellow', label='Stable')
    
    ax.text(5, 0.05, 'Init', ha='center', fontsize=8)
    ax.text(25, 0.05, 'Tracking', ha='center', fontsize=8)
    ax.text(45, 0.05, 'Stable', ha='center', fontsize=8)

def create_head_pose_viz(ax):
    """Create final head pose visualization."""
    # Head model vertices (simplified)
    head_size = 0.15
    head_vertices = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],  # back
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],      # front
    ]) * head_size
    
    # Apply rotation (example)
    angle = np.pi/6  # 30 degrees
    R = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
    
    rotated_vertices = (R @ head_vertices.T).T
    
    # Translation
    translation = np.array([0.2, -0.1, 0.05])
    final_vertices = rotated_vertices + translation
    
    # Plot head box
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # back face
        [4, 5], [5, 6], [6, 7], [7, 4],  # front face
        [0, 4], [1, 5], [2, 6], [3, 7]   # connecting edges
    ]
    
    for edge in edges:
        points = final_vertices[edge]
        ax.plot3D(*points.T, 'b-', linewidth=2)
    
    # Add face indicator (nose direction)
    face_center = np.mean(final_vertices[4:8], axis=0)
    face_normal = R @ np.array([0, 0, 0.2])
    ax.quiver(face_center[0], face_center[1], face_center[2],
              face_normal[0], face_normal[1], face_normal[2],
              color='red', arrow_length_ratio=0.3, linewidth=3)
    
    # Add coordinate frame
    origin = translation
    axes_length = 0.3
    ax.quiver(origin[0], origin[1], origin[2], axes_length, 0, 0, 
              color='red', arrow_length_ratio=0.1, alpha=0.7)
    ax.quiver(origin[0], origin[1], origin[2], 0, axes_length, 0, 
              color='green', arrow_length_ratio=0.1, alpha=0.7)
    ax.quiver(origin[0], origin[1], origin[2], 0, 0, axes_length, 
              color='blue', arrow_length_ratio=0.1, alpha=0.7)
    
    # Labels
    ax.text(origin[0] + axes_length, origin[1], origin[2], 'X', color='red')
    ax.text(origin[0], origin[1] + axes_length, origin[2], 'Y', color='green')
    ax.text(origin[0], origin[1], origin[2] + axes_length, 'Z', color='blue')
    
    # Ground plane
    xx, yy = np.meshgrid(np.linspace(-0.5, 0.5, 10), np.linspace(-0.5, 0.5, 10))
    zz = np.zeros_like(xx) - 0.2
    ax.plot_surface(xx, yy, zz, alpha=0.2, color='gray')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(-0.3, 0.3)
    
    # Add 6DOF info
    ax.text2D(0.05, 0.95, '6DOF Output:\nX: 0.234m\nY: -0.153m\nZ: 0.082m\nRoll: 5.7°\nPitch: 2.3°\nYaw: 30.0°',
              transform=ax.transAxes, fontsize=9, va='top',
              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.view_init(elev=20, azim=-60)

if __name__ == '__main__':
    print("Creating SLAM processing example visualization...")
    fig = create_slam_example_visualization()
    
    # Save the figure
    fig.savefig('slam_processing_example.png', dpi=150, bbox_inches='tight')
    print("Saved as: slam_processing_example.png")
    
    # Also create a simplified data flow diagram
    fig2, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    # Title
    fig2.suptitle('SLAM Data Flow: From Sensors to Head Pose', fontsize=16, fontweight='bold')
    
    # Draw flow diagram
    # Input boxes
    ax.add_patch(plt.Rectangle((0.1, 0.7), 0.15, 0.1, facecolor='lightblue', edgecolor='black'))
    ax.text(0.175, 0.75, 'RGB Frame\n640x480\n15 FPS', ha='center', va='center', fontsize=9)
    
    ax.add_patch(plt.Rectangle((0.1, 0.5), 0.15, 0.1, facecolor='lightgreen', edgecolor='black'))
    ax.text(0.175, 0.55, 'Head IMU\n6-axis\n30 FPS', ha='center', va='center', fontsize=9)
    
    # SLAM processing
    ax.add_patch(plt.Rectangle((0.35, 0.6), 0.2, 0.15, facecolor='lightyellow', edgecolor='black'))
    ax.text(0.45, 0.675, 'ORB-SLAM3\nProcessing', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Outputs
    outputs = [
        (0.65, 0.8, '4x4 Pose Matrix\n[R|t]', 'lightcoral'),
        (0.65, 0.65, 'Confidence Score\n0.0 - 1.0', 'lightgray'),
        (0.65, 0.5, 'Tracking State\nINIT/TRACK/LOST', 'lightyellow'),
        (0.65, 0.35, '3D Trajectory\nKeyframes', 'lightgreen'),
    ]
    
    for x, y, text, color in outputs:
        ax.add_patch(plt.Rectangle((x, y-0.05), 0.15, 0.1, facecolor=color, edgecolor='black'))
        ax.text(x+0.075, y, text, ha='center', va='center', fontsize=9)
    
    # Ensemble fusion
    ax.add_patch(plt.Rectangle((0.35, 0.25), 0.2, 0.15, facecolor='lavender', edgecolor='black'))
    ax.text(0.45, 0.325, 'Ensemble\nFusion', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Final output
    ax.add_patch(plt.Rectangle((0.65, 0.1), 0.25, 0.15, facecolor='lightgreen', edgecolor='black', linewidth=2))
    ax.text(0.775, 0.175, 'Final Head Pose\n6DOF (X,Y,Z,R,P,Y)\n+ Confidence', ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Draw arrows
    arrow_props = dict(arrowstyle='->', lw=2, color='black')
    # Inputs to SLAM
    ax.annotate('', xy=(0.35, 0.7), xytext=(0.25, 0.75), arrowprops=arrow_props)
    ax.annotate('', xy=(0.35, 0.65), xytext=(0.25, 0.55), arrowprops=arrow_props)
    
    # SLAM to outputs
    ax.annotate('', xy=(0.65, 0.8), xytext=(0.55, 0.7), arrowprops=arrow_props)
    ax.annotate('', xy=(0.65, 0.65), xytext=(0.55, 0.675), arrowprops=arrow_props)
    ax.annotate('', xy=(0.65, 0.5), xytext=(0.55, 0.65), arrowprops=arrow_props)
    ax.annotate('', xy=(0.65, 0.35), xytext=(0.55, 0.625), arrowprops=arrow_props)
    
    # To ensemble
    ax.annotate('', xy=(0.45, 0.4), xytext=(0.45, 0.6), arrowprops=arrow_props)
    ax.annotate('', xy=(0.35, 0.325), xytext=(0.25, 0.55), arrowprops=dict(arrowstyle='->', lw=2, color='green'))
    ax.text(0.28, 0.45, 'IMU\nPose', ha='center', fontsize=8, color='green')
    
    # Ensemble to final
    ax.annotate('', xy=(0.65, 0.175), xytext=(0.55, 0.325), arrowprops=dict(arrowstyle='->', lw=3, color='darkgreen'))
    
    # Add labels
    ax.text(0.3, 0.75, 'Visual', ha='center', fontsize=8, style='italic')
    ax.text(0.3, 0.58, 'Inertial', ha='center', fontsize=8, style='italic')
    
    # Add processing details
    details = [
        "• Feature extraction (ORB)",
        "• Feature matching",
        "• Motion estimation", 
        "• Bundle adjustment",
        "• Loop closure detection"
    ]
    ax.text(0.05, 0.4, 'SLAM Processing:', fontsize=10, fontweight='bold')
    for i, detail in enumerate(details):
        ax.text(0.05, 0.35-i*0.05, detail, fontsize=9)
    
    # Add fusion details
    fusion_details = [
        "• Dynamic weight calculation",
        "• Temporal consistency check",
        "• Scale ambiguity handling",
        "• Confidence-based fusion"
    ]
    ax.text(0.85, 0.4, 'Ensemble Fusion:', fontsize=10, fontweight='bold')
    for i, detail in enumerate(fusion_details):
        ax.text(0.85, 0.35-i*0.05, detail, fontsize=9)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.9)
    
    fig2.savefig('slam_data_flow.png', dpi=150, bbox_inches='tight')
    print("Saved as: slam_data_flow.png")
    
    plt.show()