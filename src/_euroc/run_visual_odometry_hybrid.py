import os
import cv2
import sys
import time
import yaml
import logging
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error
from ..visual_odometry.visual_odometry import VisualOdometry
from ..misc import setup_logging
from ..internal.extended_common import VisualOdometryConfig, DatasetConfig
from ..common.constants import EUROC_SEQUENCE_MAPS
from ..common.datatypes import ImageData


def parse_args():
    parser = argparse.ArgumentParser(description="Run Visual Odometry")
    parser.add_argument(
        "--dataset_path", 
        type=str, 
        required=True, 
        help="Path to the dataset directory"
    )
    parser.add_argument(
        "--config_file", 
        type=str, 
        default="config.yaml", 
        help="Path to the configuration file"
    )
    
    return parser.parse_args()


def run_vo(
        rootpath: str, 
        variant: str,
        output_dir: str, 
        config: VisualOdometryConfig,
    ):

    dataset_config = DatasetConfig(
        type='euroc',
        mode='stream',
        root_path=rootpath,
        variant=variant,
    )

    logging.info("VO Config: %s", config)
    logging.info("Dataset Config: %s", dataset_config)

    vo = VisualOdometry(config=config, dataset_config=dataset_config, debug=True)


    logging.info("Visual Odometry initialized.")
    
    sequence = EUROC_SEQUENCE_MAPS.get(variant, None)
    if sequence is None:
        raise ValueError(f"Unknown driving date or sequence number for variant {variant}")
    
    timestamp_path = os.path.join(rootpath, f"{sequence}/cam0/data.csv")
    timestamps_df = pd.read_csv(timestamp_path, names=["timestamp", "filename"], skiprows=1)
    timestamps = timestamps_df["timestamp"].values / 1e9  # Convert to seconds
    image_path = os.path.join(rootpath, f"{sequence}/cam0/data")
    image_files = sorted([f for f in os.listdir(image_path) if f.endswith('.png')])

    logging.info(f"Found {len(image_files)} image files.")

    ground_truth_path = os.path.join(rootpath, f"{sequence}/state_groundtruth_estimate0/data.csv")
    columns = "#timestamp, p_RS_R_x [m], p_RS_R_y [m], p_RS_R_z [m], q_RS_w [], q_RS_x [], q_RS_y [], q_RS_z [], v_RS_R_x [m s^-1], v_RS_R_y [m s^-1], v_RS_R_z [m s^-1], b_w_RS_S_x [rad s^-1], b_w_RS_S_y [rad s^-1], b_w_RS_S_z [rad s^-1], b_a_RS_S_x [m s^-2], b_a_RS_S_y [m s^-2], b_a_RS_S_z [m s^-2]".split(", ")
    df = pd.read_csv(ground_truth_path, names=columns, skiprows=1)
    gt_position = df[["p_RS_R_x [m]", "p_RS_R_y [m]", "p_RS_R_z [m]"]].values

    logging.info(f"Loaded ground truth data with {len(gt_position)} entries.")


    ground_truth_position = []
    estimated_position = []

    estimated_pose = []
    current_pose = np.eye(4)
    current_pose[:3, 3] = np.array([-gt_position[0, 1], -gt_position[0, 2], gt_position[0, 0]])
    estimated_position.append(current_pose[:3, 3])
    ground_truth_position.append(gt_position[0])

    vo_relative_pose = []

    detected_points = []

    for i, (image_file, ts) in enumerate(tqdm(zip(image_files, timestamps), total=len(image_files))):
        idx = i + 1
        frame_path = os.path.join(image_path, image_file)
        frame = cv2.imread(frame_path)
        vo_output = vo.compute_pose(ImageData(image=frame, timestamp=ts))
        if vo_output.success:
            pose = vo_output.relative_pose
            current_pose = current_pose @ pose
            estimated_pose.append(current_pose[:3, :].flatten())

            estimated_position.append(current_pose[:3, 3])
            if idx < len(gt_position):
                ground_truth_position.append(gt_position[idx])
            vo_relative_pose.append(pose.flatten())

            debugging_data = vo.get_debugging_data()
            if debugging_data is not None:
                points = len(debugging_data.prev_pts) if debugging_data.prev_pts is not None else 0
                detected_points.append(points)


    estimated_pose = np.array(estimated_pose)
    
    estimated_position = np.array(estimated_position)

    vo_relative_pose = np.array(vo_relative_pose)

    detected_points = np.array(detected_points)

    logging.info(f"Estimated Pose shape: {estimated_position.shape}")

    abs_output_filename = os.path.join(output_dir, f"absolute_pose/{sequence}/data.csv")
    relative_output_filename = os.path.join(output_dir, f"relative_pose/{sequence}/data.csv")
    keypoints_output_filename = os.path.join(output_dir, f"keypoints/{sequence}/data.csv")
    output_image_filename = os.path.join(output_dir, f"images/{variant}.png")

    abs_csv_dir_name = os.path.dirname(abs_output_filename)
    rel_csv_dir_name = os.path.dirname(relative_output_filename)
    keypoints_csv_dir_name = os.path.dirname(keypoints_output_filename)
    image_dir_name = os.path.dirname(output_image_filename)
    
    if not os.path.exists(abs_csv_dir_name):
        os.makedirs(abs_csv_dir_name, exist_ok=True)
    if not os.path.exists(rel_csv_dir_name):
        os.makedirs(rel_csv_dir_name, exist_ok=True)
    if not os.path.exists(keypoints_csv_dir_name):
        os.makedirs(keypoints_csv_dir_name, exist_ok=True)
    if not os.path.exists(image_dir_name):
        os.makedirs(image_dir_name, exist_ok=True)

    df = pd.DataFrame(estimated_pose)
    df.to_csv(abs_output_filename, index=False, header=False)

    df = pd.DataFrame(vo_relative_pose)
    df.to_csv(relative_output_filename, index=False, header=False)

    df = pd.DataFrame(detected_points)
    df.to_csv(keypoints_output_filename, index=False, header=False)

    plt.figure(figsize=(10, 8))
    px, py, pz = gt_position.T
    plt.plot(px, pz, marker='o', markersize=1, label='Ground Truth Trajectory', color='black')
    px, py, pz = estimated_position.T
    plt.plot(-px, pz, marker='o', markersize=1, label='Estimated Trajectory', color='blue')
    plt.title(f'Trajectory estimation (seq: {variant})')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.axis('equal')
    plt.grid()
    plt.legend()
    plt.savefig(output_image_filename)
    plt.close()

if __name__ == "__main__":

    args = parse_args()
    setup_logging(log_level='INFO', log_output='.debugging/export_vo_estimate_log_hybrid_euroc')

    logging.info("Starting Visual Odometry experiment.")
    logging.debug(f"args: {args}")

    if not os.path.exists(args.config_file):
        logging.error(f"Config file not found: {args.config_file}")
        exit(1)

    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)
        vo_json_config = config.get("visual_odometry", None)

    if vo_json_config is None:
        logging.error("Visual Odometry configuration not found in the config file.")
        exit(1)

    output_dir = "/Volumes/Data_EXT/data/workspaces/sensor_fusion/outputs/vo_estimates/pose_estimates_hybrid_euroc_test"
    os.makedirs(output_dir, exist_ok=True)

    variants = [
        "01",
    ]

    for variant in variants:

        config = VisualOdometryConfig.from_json(vo_json_config)
        config.estimator = "hybrid"

        run_vo(
            rootpath=args.dataset_path,
            variant=variant,
            output_dir=output_dir,
            config=config
        )