import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from src.common.constants import KITTI_SEQUENCE_TO_DATE, KITTI_SEQUENCE_TO_DRIVE

def generate_hybrid_vo_pose_estimation(vo_2d2d_path: str, vo_2d3d_path: str, save_dir: str):
    
    vo_relative_pose_path = os.path.join(save_dir, "relative_pose")
    vo_absolute_pose_path = os.path.join(save_dir, "absolute_pose")
    date = vo_2d2d_path.split("/")[-3]
    seq = vo_2d2d_path.split("/")[-2]
    relative_pose_output_dir = os.path.join(vo_relative_pose_path, date, seq)
    vo_absolute_pose_output_dir = os.path.join(vo_absolute_pose_path, date, seq)
    os.makedirs(relative_pose_output_dir, exist_ok=True)
    os.makedirs(vo_absolute_pose_output_dir, exist_ok=True)


    vo_2d2d_data = pd.read_csv(vo_2d2d_path, header=None).values.reshape(-1, 4, 4)[:, :3, :]
    vo_2d3d_data = pd.read_csv(vo_2d3d_path, header=None).values.reshape(-1, 4, 4)[:, :3, :]

    # take 2d2d rotation and 2d3d translation
    hybrid_vo_data = np.zeros((vo_2d2d_data.shape[0], 3, 4))
    hybrid_vo_data[:, :3, :3] = vo_2d2d_data[:, :3, :3] # Rotation from 2d2d epipolar geometry pose estimation
    hybrid_vo_data[:, :, 3] = vo_2d3d_data[:, :, 3] # Translation from 2d3d pnp pose estimation
    
    # compute absolute pose

    vo_absolute_pose = np.eye(4)
    hybrid_absolute_vo_data = []
    for vo_data in hybrid_vo_data:
        _vo_data = np.eye(4)
        _vo_data[:3, :] = vo_data
        vo_absolute_pose = vo_absolute_pose @ _vo_data
        hybrid_absolute_vo_data.append(vo_absolute_pose[:3, :].flatten())
    
    hybrid_absolute_vo_data = np.array(hybrid_absolute_vo_data)

    hybrid_relative_vo_data = hybrid_vo_data.reshape(-1, 12)
    hybrid_relative_vo_df = pd.DataFrame(hybrid_relative_vo_data)

    hybrid_absolute_vo_df = pd.DataFrame(hybrid_absolute_vo_data)

    hybrid_relative_vo_df.to_csv(os.path.join(relative_pose_output_dir, "data.csv"), header=False, index=False)
    logging.info(f"Hybrid VO pose estimation data saved to {relative_pose_output_dir}")
    hybrid_absolute_vo_df.to_csv(os.path.join(vo_absolute_pose_output_dir, "data.csv"), header=False, index=False)
    logging.info(f"Hybrid VO pose estimation data saved to {vo_absolute_pose_output_dir}")
    return hybrid_absolute_vo_data.reshape(-1, 3, 4)

def save_visualization(vo_pose: np.ndarray, gt_pose_dir: str, save_dir: str, sequence_nr: str):
    
    output_dir = os.path.join(save_dir, "images")
    os.makedirs(output_dir, exist_ok=True)

    gt_pose_path = os.path.join(gt_pose_dir, f"{sequence_nr}.txt")
    gt_pose = pd.read_csv(gt_pose_path, sep=" ", header=None).values.reshape(-1, 3, 4)
    min_index = min(gt_pose.shape[0], vo_pose.shape[0])
    gt_positions = gt_pose[:min_index, :3, 3]
    vo_positions = vo_pose[:min_index, :3, 3]

    plt.figure(figsize=(10, 8))
    px, py, pz = gt_positions.T
    plt.plot(px, pz, marker='o', markersize=1, label='Ground Truth Trajectory', color='black')
    px, py, pz = vo_positions.T
    plt.plot(px, pz, marker='o', markersize=1, label='Estimated Trajectory', color='blue')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title(f'Trajectory estimation (seq: {sequence_nr})')
    plt.axis('equal')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, f"{sequence_nr}.png"))
    plt.close()

def extract_vo_pose_path(vo_2d2d_base_dir: str, vo_2d3d_base_dir: str, date: str, drive: str):
    vo_2d2d_path = os.path.join(vo_2d2d_base_dir, date, drive, "data.csv")
    vo_2d3d_path = os.path.join(vo_2d3d_base_dir, date, drive, "data.csv")
    if not os.path.exists(vo_2d2d_path) or not os.path.exists(vo_2d3d_path):
        logging.warning(f"VO paths do not exist for date: {date}, drive: {drive}")
        return None
    
    return vo_2d2d_path, vo_2d3d_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    vo_2d2d_base_dir = "/Volumes/Data_EXT/data/workspaces/sensor_fusion/outputs/vo_estimates/pose_estimation/relative_pose"
    vo_2d3d_base_dir = "/Volumes/Data_EXT/data/workspaces/sensor_fusion/outputs/vo_estimates/pose_estimation_2d3d/relative_pose"
    ground_truth_dir = "/Volumes/Data_EXT/data/workspaces/sensor_fusion/data/KITTI/ground_truth"

    save_dir = "/Volumes/Data_EXT/data/workspaces/sensor_fusion/outputs/vo_estimates/hybrid_vo_pose_estimation"

    sequences = [f"0{n}" if n < 10 else f"{n}" for n in range(1, 11)]

    for seq in sequences:
        date = KITTI_SEQUENCE_TO_DATE[seq]
        drive = KITTI_SEQUENCE_TO_DRIVE[seq]
        vo_paths = extract_vo_pose_path(vo_2d2d_base_dir, vo_2d3d_base_dir, date, drive)
        if vo_paths is None:
            continue

        vo_2d2d_path, vo_2d3d_path = vo_paths
        logging.info(f"Processing sequence {seq} with date {date} and drive {drive}")
        logging.info(f"2D-2D VO path: {vo_2d2d_path}")
        logging.info(f"2D-3D VO path: {vo_2d3d_path}")
        hybrid_absolute_vo_data = generate_hybrid_vo_pose_estimation(vo_2d2d_path, vo_2d3d_path, save_dir)
        save_visualization(hybrid_absolute_vo_data, ground_truth_dir, save_dir, seq)
