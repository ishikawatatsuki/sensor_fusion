
import os
import sys
import cv2
import yaml
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from ..visual_odometry import VisualOdometry
from ...common.config import VisualOdometryConfig
from ...internal.extended_common.extended_config import DatasetConfig
from ...common.datatypes import ImageData
from ...common.constants import EUROC_SEQUENCE_MAPS, KITTI_SEQUENCE_TO_DATE, KITTI_SEQUENCE_TO_DRIVE

import logging
import time
import sys
import os
import pykitti
from tqdm import tqdm
from src.internal.visualizers.vo_visualizer import VO_Visualizer, VO_VisualizationData
from src.common.constants import KITTI_SEQUENCE_TO_DATE, KITTI_SEQUENCE_TO_DRIVE, EUROC_SEQUENCE_MAPS


logging.basicConfig(level=logging.INFO)
logging.getLogger('matplotlib.font_manager').disabled = True

use_kitti = False

if use_kitti:
    rootpath = "/Volumes/Data_EXT/data/workspaces/sensor_fusion/data/KITTI"
    variant = "09"
    date = KITTI_SEQUENCE_TO_DATE.get(variant, "2011_09_30")
    drive = KITTI_SEQUENCE_TO_DRIVE.get(variant, "0033")
    dataset_config = DatasetConfig(
        type='kitti',
        mode='stream',
        root_path=rootpath,
        variant=variant,
    )
    image_path = os.path.join(rootpath, f"{date}/{date}_drive_{drive}_sync/image_00/data")
    image_files = sorted([f for f in os.listdir(image_path) if f.endswith('.png')])
    date = KITTI_SEQUENCE_TO_DATE.get(variant)
    drive = KITTI_SEQUENCE_TO_DRIVE.get(variant)
    kitti_dataset = pykitti.raw(rootpath, date, drive)
    image_timestamps = np.array([datetime.timestamp(ts) for ts in kitti_dataset.timestamps])
    start = 0
    image_files = image_files[start:]
    image_timestamps = image_timestamps[start:]
    logging.debug(f"Found {len(image_files)} image files.")
    ground_truth_path = os.path.join(rootpath, f"ground_truth/{variant}.txt")
    ground_truth = pd.read_csv(ground_truth_path, sep=' ', header=None, skiprows=1).values
    ground_truth = ground_truth.reshape(-1, 3, 4)
    gt_pos = ground_truth[start:, :3, 3]

    config_filepath = "/Volumes/Data_EXT/data/workspaces/sensor_fusion/configs/kitti_config.yaml"
else:
    rootpath = "/Volumes/Data_EXT/data/workspaces/sensor_fusion/data/EuRoC"
    variant = "01"
    sequence = EUROC_SEQUENCE_MAPS.get(variant, "MH_01_easy")
    dataset_config = DatasetConfig(
        type='euroc',
        mode='stream',
        root_path=rootpath,
        variant=variant,
    )
    offset_start = 0
    end = offset_start + 5000
    image_path = os.path.join(rootpath, f"{sequence}/cam0/data")
    image_timestamps = pd.read_csv(os.path.join(rootpath, f"{sequence}/cam0/data.csv"), names=["#timestamp [ns]", "filename"], skiprows=1)['#timestamp [ns]'].values
    image_files = sorted([f for f in os.listdir(image_path) if f.endswith('.png')])
    logging.debug(f"Found {len(image_files)} image files.")

    image_timestamps = image_timestamps[offset_start:end]
    image_files = image_files[offset_start:end]
    logging.debug(f"Selected {len(image_files)} image files from index {offset_start} to {end}.")

    ground_truth_path = os.path.join(rootpath, f"{sequence}/state_groundtruth_estimate0/data.csv")
    columns = "#timestamp, p_RS_R_x [m], p_RS_R_y [m], p_RS_R_z [m], q_RS_w [], q_RS_x [], q_RS_y [], q_RS_z [], v_RS_R_x [m s^-1], v_RS_R_y [m s^-1], v_RS_R_z [m s^-1], b_w_RS_S_x [rad s^-1], b_w_RS_S_y [rad s^-1], b_w_RS_S_z [rad s^-1], b_a_RS_S_x [m s^-2], b_a_RS_S_y [m s^-2], b_a_RS_S_z [m s^-2]".split(", ")
    df = pd.read_csv(ground_truth_path, names=columns, skiprows=1)
    initial_timestamp = image_timestamps[0]
    gt_pos = df[df["#timestamp"] > initial_timestamp][["p_RS_R_x [m]", "p_RS_R_y [m]", "p_RS_R_z [m]"]].values

    config_filepath = "/Volumes/Data_EXT/data/workspaces/sensor_fusion/configs/euroc_config.yaml"

with open(config_filepath, 'r') as f:
    config = yaml.safe_load(f)
    vo_json_config = config.get("visual_odometry", None)

if vo_json_config is None:
    logging.error("Visual Odometry configuration not found in the config file.")
    exit(1)

config = VisualOdometryConfig.from_json(vo_json_config)
config.type = "monocular"
config.estimator = "2d3d"

vo = VisualOdometry(config=config, dataset_config=dataset_config, debug=True)
logging.debug("Visual Odometry initialized.")

logging.debug(gt_pos.shape)
logging.debug(f"Loaded ground truth data with {len(gt_pos)} entries.")

ground_truth_position = []
estimated_position = []
estimated_pose = []
current_pose = np.eye(4)
if use_kitti:
    current_pose[:3, 3] = np.array([gt_pos[0, 0], gt_pos[0, 1], gt_pos[0, 2]])
else:
    current_pose[:3, 3] = np.array([-gt_pos[0, 1], -gt_pos[0, 2], gt_pos[0, 0]])

if vo.is_debugging:
    vis = VO_Visualizer(
        save_path=f"/Volumes/Data_EXT/data/workspaces/sensor_fusion/outputs/vo_estimates/vo_debug/{config.estimator}_{variant}_test",
        save_frame=True
    )
    vis.start()

for i, (image_file, timestamp) in enumerate(tqdm(zip(image_files, image_timestamps), total=len(image_files))):
    idx = i + 1
    frame_path = os.path.join(image_path, image_file)
    frame = cv2.imread(frame_path)
    vo_data = vo.compute_pose(ImageData(image=frame, timestamp=timestamp))
    if vo_data.success:
        if idx >= len(gt_pos):
            continue

        pose = vo_data.relative_pose
        current_pose = current_pose @ pose
        estimated_pose.append(current_pose[:3, :].flatten())
        estimated_position.append(current_pose[:3, 3])
        ground_truth_position.append(gt_pos[idx])
        debugging_data = vo.get_debugging_data()
        if debugging_data.prev_pts is not None:
            x, y, z = current_pose[:3, 3]
            _est_pose = np.array([x, z])
            px, py, pz = gt_pos[idx]
            if use_kitti:
                _gt_pose = np.array([px, pz])
            else:
                _gt_pose = np.array([-py, px])
            vis_data = VO_VisualizationData(
                frame=frame, 
                mask=debugging_data.mask,
                pts_prev=debugging_data.prev_pts, 
                pts_curr=debugging_data.next_pts, 
                estimated_pose=_est_pose, 
                gt_pose=_gt_pose
            )
            vis.send(vis_data)
            time.sleep(0.05)

estimated_pose = np.array(estimated_pose)

ground_truth_position = np.array(ground_truth_position)
estimated_position = np.array(estimated_position)

min_len = min(len(ground_truth_position), len(estimated_position))
ground_truth_position = ground_truth_position[:min_len]
estimated_position = estimated_position[:min_len]
logging.info(f"Estimated Positions shape: {ground_truth_position.shape}")
logging.info(f"Estimated Pose shape: {estimated_position.shape}")

# save poses in npz format
filename = f"vo_estimates_{config.estimator}_{variant}_v3.npz"
np.savez_compressed(
    filename,
    ground_truth=ground_truth_position,
    estimated=estimated_position,
    estimated_pose=estimated_pose
)
logging.info(f"Saved VO estimates to '{filename}'.")


# mae = mean_absolute_error(ground_truth_position, estimated_position)
# logging.info(f"Mean Absolute Error (MAE) of the estimated positions: {mae:.4f}m")

vis.stop()

# # Plot the estimated trajectory
# plt.figure(figsize=(10, 8))
# px, py, pz = gt_pos.T
# plt.plot(px, py, marker='o', markersize=1, label='Ground Truth Trajectory', color='black')
# px, py, pz = estimated_position.T
# plt.plot(pz, -px, marker='o', markersize=1, label='Estimated Trajectory', color='blue')
# plt.title('Estimated Trajectory from Visual Odometry')
# plt.xlabel('X Position (m)')
# plt.ylabel('Y Position (m)')
# plt.axis('equal')
# plt.grid()
# plt.legend()
# # plt.show()
# plt.savefig('estimated_trajectory.png')

logging.info("Trajectory plot saved as 'estimated_trajectory.png'.")