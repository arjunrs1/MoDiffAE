#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import os
from mpl_toolkits.mplot3d import Axes3D
import argparse

def get_skeleton_connections():
    """Define connections between joints based on the provided joint ordering"""
    connections = [
        # Head
        (0, 1),  # LFHD to RFHD (front head)
        (2, 3),  # LBHD to RBHD (back head)
        (0, 2),  # LFHD to LBHD (left head)
        (1, 3),  # RFHD to RBHD (right head)
        (4, 0),  # C7 to LFHD
        
        # Torso
        (4, 6),  # C7 to CLAV
        (6, 7),  # CLAV to STRN
        (4, 5),  # C7 to T10
        (5, 8),  # T10 to BACK
        
        # Left arm
        # (6, 9),   # CLAV to LSHO
        # (9, 10),  # LSHO to LUPA
        # (10, 11), # LUPA to LELB
        # (11, 12), # LELB to LFRM
        # (12, 13), # LFRM to LWRA
        # (13, 15), # LWRA to LFIN
        
        # # Right arm
        # (6, 16),  # CLAV to RSHO
        # (16, 17), # RSHO to RUPA
        # (17, 18), # RUPA to RELB
        # (18, 19), # RELB to RFRM
        # (19, 20), # RFRM to RWRA
        # (20, 22), # RWRA to RFIN
        
        # Pelvis
        (5, 23),  # T10 to LASI
        (5, 24),  # T10 to RASI
        (23, 24), # LASI to RASI
        (23, 25), # LASI to LPSI
        (24, 26), # RASI to RPSI
        (25, 26), # LPSI to RPSI
        
        # Left leg
        (23, 27), # LASI to LTHI
        (27, 28), # LTHI to LKNE
        (28, 29), # LKNE to LTIB
        (29, 30), # LTIB to LANK
        (30, 31), # LANK to LHEE
        (30, 32), # LANK to LTOE
        
        # Right leg
        (24, 33), # RASI to RTHI
        (33, 34), # RTHI to RKNE
        (34, 35), # RKNE to RTIB
        (35, 36), # RTIB to RANK
        (36, 37), # RANK to RHEE
        (36, 38), # RANK to RTOE
    ]
    return connections

def visualize_poses(og_path, manipulated_path, output_dir=None, fps=10):
    # Load data
    og_xyz = np.load(og_path)
    manipulated_xyz = np.load(manipulated_path)
    
    print(f"Original data shape: {og_xyz.shape}")
    print(f"Manipulated data shape: {manipulated_xyz.shape}")
    
    # Make sure the data has the right shape
    assert og_xyz.shape[1] == 117, "Expected 117 values per frame (39 joints × 3 coordinates)"
    
    # Reshape data to (frames, joints, 3)
    n_frames = og_xyz.shape[0]
    og_xyz_reshaped = og_xyz.reshape(n_frames, 39, 3)
    manipulated_xyz_reshaped = manipulated_xyz.reshape(n_frames, 39, 3)
    
    # Define arm joints to exclude (LSHO through LFIN and RSHO through RFIN)
    arm_joint_indices = list(range(9, 16)) + list(range(16, 23))  # indices 9-15 and 16-22
    
    # Create mask for non-arm joints (all joints except arms)
    valid_joints = np.ones(39, dtype=bool)
    valid_joints[arm_joint_indices] = False
    
    # Get filtered skeleton connections (no arm connections)
    connections = get_skeleton_connections()
    
    # Create figure and 3D axes for animation
    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    
    # Function to set consistent axis properties
    def set_axes_properties(ax, title):
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        ax.grid(False)
        ax.view_init(elev=20, azim=-60)
    
    set_axes_properties(ax1, 'Original Pose')
    set_axes_properties(ax2, 'Manipulated Pose')
    
    # Set consistent axis limits for better visualization
    # Filter out arm joints when calculating min/max
    filtered_og = og_xyz_reshaped[:, valid_joints, :]
    filtered_manip = manipulated_xyz_reshaped[:, valid_joints, :]
    
    all_data = np.vstack([filtered_og.reshape(-1, 3), filtered_manip.reshape(-1, 3)])
    min_vals = np.min(all_data, axis=0) - 0.1  # Add small margin
    max_vals = np.max(all_data, axis=0) + 0.1
    
    # Create consistent view range
    x_range = max_vals[0] - min_vals[0]
    y_range = max_vals[1] - min_vals[1]
    z_range = max_vals[2] - min_vals[2]
    max_range = max(x_range, y_range, z_range) / 2
    
    mid_x = (max_vals[0] + min_vals[0]) / 2
    mid_y = (max_vals[1] + min_vals[1]) / 2
    mid_z = (max_vals[2] + min_vals[2]) / 2
    
    for ax in [ax1, ax2]:
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # For animation using frame-by-frame approach
    for frame in range(n_frames):
        print(f"Rendering frame {frame+1}/{n_frames}")
        
        # Get data for current frame (only non-arm joints)
        og_frame_data = og_xyz_reshaped[frame][valid_joints]
        manip_frame_data = manipulated_xyz_reshaped[frame][valid_joints]
        
        # Clear previous frame
        ax1.clear()
        ax2.clear()
        set_axes_properties(ax1, 'Original Pose')
        set_axes_properties(ax2, 'Manipulated Pose')
        
        # Reset consistent view limits
        for ax in [ax1, ax2]:
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # Plot only non-arm joints
        ax1.scatter(og_frame_data[:, 0], og_frame_data[:, 1], og_frame_data[:, 2], c='b', marker='o', s=30)
        ax2.scatter(manip_frame_data[:, 0], manip_frame_data[:, 1], manip_frame_data[:, 2], c='b', marker='o', s=30)
        
        # Draw skeleton connections (skip connections involving arm joints)
        for start, end in connections:
            # Skip connections involving arm joints
            if start in arm_joint_indices or end in arm_joint_indices:
                continue
                
            # Original pose
            ax1.plot([og_xyz_reshaped[frame, start, 0], og_xyz_reshaped[frame, end, 0]],
                     [og_xyz_reshaped[frame, start, 1], og_xyz_reshaped[frame, end, 1]],
                     [og_xyz_reshaped[frame, start, 2], og_xyz_reshaped[frame, end, 2]], 'r-', linewidth=2)
            
            # Manipulated pose
            ax2.plot([manipulated_xyz_reshaped[frame, start, 0], manipulated_xyz_reshaped[frame, end, 0]],
                     [manipulated_xyz_reshaped[frame, start, 1], manipulated_xyz_reshaped[frame, end, 1]],
                     [manipulated_xyz_reshaped[frame, start, 2], manipulated_xyz_reshaped[frame, end, 2]], 'r-', linewidth=2)
        
        # Add frame counter
        fig.suptitle(f'Frame {frame+1}/{n_frames}', fontsize=16)
        
        # Save frame if required
        if output_dir:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            # Create temp folder for frames if it doesn't exist
            temp_dir = os.path.join(output_dir, 'temp_frames')
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
                
            # Save the current frame
            plt.savefig(f'{temp_dir}/frame_{frame:04d}.png')
        else:
            plt.pause(1/fps)  # Display frame when not saving
    
    # Combine frames into video if saving
    if output_dir:
        base_name = os.path.basename(og_path).split('og_')[-1].split('.npy')[0]
        output_file = os.path.join(output_dir, f'comparison_{base_name}.mp4')
        temp_dir = os.path.join(output_dir, 'temp_frames')
        
        print(f"Combining frames into video: {output_file}")
        
        # Use FFmpeg to create video
        import subprocess
        cmd = f'ffmpeg -y -framerate {fps} -i {temp_dir}/frame_%04d.png -c:v libx264 -pix_fmt yuv420p {output_file}'
        print(f"Running command: {cmd}")
        subprocess.call(cmd, shell=True)
        
        print(f"Video created at {output_file}")
        
        # Clean up temporary files
        import shutil
        shutil.rmtree(temp_dir)
    else:
        plt.show()

def render_single_frame(og_path, manipulated_path, frame_num=0):
    # Load data
    og_xyz = np.load(og_path)
    manipulated_xyz = np.load(manipulated_path)
    
    # Reshape data
    og_xyz_reshaped = og_xyz.reshape(og_xyz.shape[0], 39, 3)
    manipulated_xyz_reshaped = manipulated_xyz.reshape(manipulated_xyz.shape[0], 39, 3)

    # Define arm joints to exclude (LSHO through LFIN and RSHO through RFIN)
    arm_joint_indices = list(range(9, 16)) + list(range(16, 23))  # indices 9-15 and 16-22
    
    # Create mask for non-arm joints
    valid_joints = np.ones(39, dtype=bool)
    valid_joints[arm_joint_indices] = False
    
    # Get connections
    connections = get_skeleton_connections()
    
    # Create plot
    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    # Get data for frame_num (only non-arm joints)
    print(og_xyz_reshaped.shape)
    og_frame_data = og_xyz_reshaped[:, valid_joints,:]
    print(og_frame_data.shape)
    manip_frame_data = manipulated_xyz_reshaped[:, valid_joints,:]
    
    # Plot points for frame_num
    ax1.scatter(og_frame_data[frame_num, :, 0], 
                og_frame_data[frame_num, :, 1], 
                og_frame_data[frame_num, :, 2], c='b')
    ax2.scatter(manip_frame_data[frame_num, :, 0], 
                manip_frame_data[frame_num, :, 1], 
                manip_frame_data[frame_num, :, 2], c='b')
    
    # Plot lines
    for start, end in connections:
        # Skip connections involving arm joints
        if start in arm_joint_indices or end in arm_joint_indices:
            continue
        # Original
        ax1.plot([og_xyz_reshaped[frame_num, start, 0], og_xyz_reshaped[frame_num, end, 0]],
                 [og_xyz_reshaped[frame_num, start, 1], og_xyz_reshaped[frame_num, end, 1]],
                 [og_xyz_reshaped[frame_num, start, 2], og_xyz_reshaped[frame_num, end, 2]], 'r-')
        
        # Manipulated
        ax2.plot([manipulated_xyz_reshaped[frame_num, start, 0], manipulated_xyz_reshaped[frame_num, end, 0]],
                 [manipulated_xyz_reshaped[frame_num, start, 1], manipulated_xyz_reshaped[frame_num, end, 1]],
                 [manipulated_xyz_reshaped[frame_num, start, 2], manipulated_xyz_reshaped[frame_num, end, 2]], 'r-')
    
    ax1.set_title('Original')
    ax2.set_title('Manipulated')
    plt.tight_layout()
    
    # Save the static image
    plt.savefig('static_frame.png')
    print("Saved static frame to static_frame.png")
    plt.close()

def animate_poses(og_path, manipulated_path, output_dir=None, fps=10):
    """Create a direct frame-by-frame video without FuncAnimation"""
    import matplotlib as mpl
    # Force non-interactive backend for headless rendering
    mpl.use('Agg')
    
    # Load and prepare data
    og_xyz = np.load(og_path)
    manipulated_xyz = np.load(manipulated_path)
    
    n_frames = og_xyz.shape[0]
    og_xyz_reshaped = og_xyz.reshape(n_frames, 39, 3)
    manipulated_xyz_reshaped = manipulated_xyz.reshape(n_frames, 39, 3)

    # Define arm joints to exclude (LSHO through LFIN and RSHO through RFIN)
    arm_joint_indices = list(range(9, 16)) + list(range(16, 23))  # indices 9-15 and 16-22
    
    # Create mask for non-arm joints
    valid_joints = np.ones(39, dtype=bool)
    valid_joints[arm_joint_indices] = False
    
    connections = get_skeleton_connections()
    
    # Create output directory for frames
    import tempfile
    temp_dir = tempfile.mkdtemp()
    print(f"Created temp directory: {temp_dir}")
    
    # Set up figure once
    for frame in range(n_frames):
        print(f"Rendering frame {frame+1}/{n_frames}")
        
        fig = plt.figure(figsize=(16, 8))
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122, projection='3d')
        
        # Render the frame
        ax1.scatter(og_xyz_reshaped[frame, valid_joints, 0], 
                    og_xyz_reshaped[frame, valid_joints, 1], 
                    og_xyz_reshaped[frame, valid_joints, 2], c='b')
        ax2.scatter(manipulated_xyz_reshaped[frame, valid_joints, 0], 
                    manipulated_xyz_reshaped[frame, valid_joints, 1], 
                    manipulated_xyz_reshaped[frame, valid_joints, 2], c='b')
        
        # Plot lines
        for start, end in connections:
            # Original
            ax1.plot([og_xyz_reshaped[frame, start, 0], og_xyz_reshaped[frame, end, 0]],
                     [og_xyz_reshaped[frame, start, 1], og_xyz_reshaped[frame, end, 1]],
                     [og_xyz_reshaped[frame, start, 2], og_xyz_reshaped[frame, end, 2]], 'r-')
            
            # Manipulated
            ax2.plot([manipulated_xyz_reshaped[frame, start, 0], manipulated_xyz_reshaped[frame, end, 0]],
                     [manipulated_xyz_reshaped[frame, start, 1], manipulated_xyz_reshaped[frame, end, 1]],
                     [manipulated_xyz_reshaped[frame, start, 2], manipulated_xyz_reshaped[frame, end, 2]], 'r-')
        
        ax1.set_title('Novice')
        ax2.set_title('Expert')
        fig.suptitle(f'Frame {frame+1}/{n_frames}', fontsize=16)
        
        # Save individual frame
        plt.savefig(f'{temp_dir}/frame_{frame:04d}.png')
        plt.close(fig)
    
    # Combine frames into video using FFmpeg directly
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Get base filename from original path
        base_name = os.path.basename(og_path).split('og_')[-1].split('.npy')[0]
        output_file = os.path.join(output_dir, f'comparison_{base_name}.mp4')
        
        print(f"Combining frames into video: {output_file}")
        
        # Use FFmpeg directly
        import subprocess
        cmd = f'ffmpeg -y -framerate {fps} -i {temp_dir}/frame_%04d.png -c:v libx264 -pix_fmt yuv420p {output_file}'
        print(f"Running command: {cmd}")
        subprocess.call(cmd, shell=True)
        
        print(f"Video created at {output_file}")
        
        # Clean up temporary files
        import shutil
        shutil.rmtree(temp_dir)

def main():
    parser = argparse.ArgumentParser(description='Visualize 3D pose animations side by side')
    parser.add_argument('--base_dir', type=str, default='evaluation/test_manipulations', 
                        help='Base directory containing the pose data')
    parser.add_argument('--output_dir', type=str, default='evaluation/visualizations',
                        help='Directory to save the animations')
    parser.add_argument('--fps', type=int, default=10, help='Frames per second for the animation')
    
    args = parser.parse_args()
    
    base_dir = args.base_dir
    if not os.path.isabs(base_dir):
        base_dir = os.path.join(os.getcwd(), base_dir)
    
    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(os.getcwd(), output_dir)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for root, dirs, files in os.walk(base_dir):

        # Find all unique filenames (without og_ or manipulated_ prefix)
        trimmed_file_names = [f.split('og_')[-1] if 'og_' in f else f.split('manipulated_')[-1] for f in files]
        unique_names = list(set(trimmed_file_names))
        unique_names = [f for f in unique_names if f.endswith('.npy')]
        
        if len(unique_names) > 0:
            animation_dir = os.path.join(output_dir, os.path.basename(root))
            
            for file_name in unique_names:
                og_xyz_path = os.path.join(root, f'og_{file_name}')
                manipulated_xyz_path = os.path.join(root, f'manipulated_{file_name}')
                
                if os.path.exists(og_xyz_path) and os.path.exists(manipulated_xyz_path):
                    print(f"\nProcessing {file_name}...")
                    
                    # First make sure we can render static frames
                    render_single_frame(og_xyz_path, manipulated_xyz_path, frame_num=0)
                    
                    # Then try the direct frame-by-frame animation approach
                    animate_poses(og_xyz_path, manipulated_xyz_path, animation_dir, args.fps)

if __name__ == '__main__':
    main()