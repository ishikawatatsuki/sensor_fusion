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
from ..common.constants import KITTI_SEQUENCE_TO_DATE, KITTI_SEQUENCE_TO_DRIVE
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
        "--output_path", 
        type=str, 
        required=True, 
        help="Path to save the output results"
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
        type='kitti',
        mode='stream',
        root_path=rootpath,
        variant=variant,
    )

    vo = VisualOdometry(config=config, dataset_config=dataset_config, debug=False)


    logging.info("Visual Odometry initialized.")
    
    driving_date = KITTI_SEQUENCE_TO_DATE.get(variant, None)
    sequence_nr = KITTI_SEQUENCE_TO_DRIVE.get(variant, None)
    if driving_date is None or sequence_nr is None:
        raise ValueError(f"Unknown driving date or sequence number for variant {variant}")
    
    image_folder = f"{driving_date}/{driving_date}_drive_{sequence_nr}_sync/image_00/data"

    image_path = os.path.join(rootpath, image_folder)
    image_files = sorted([f for f in os.listdir(image_path) if f.endswith('.png')])

    logging.info(f"Found {len(image_files)} image files.")

    ground_truth_path = os.path.join(rootpath, f"ground_truth/{variant}.txt")
    ground_truth = pd.read_csv(ground_truth_path, sep=' ', header=None, skiprows=1).values
    ground_truth_mat = ground_truth.reshape(-1, 3, 4)
    gt_position = ground_truth_mat[:, :3, 3]

    logging.info(f"Loaded ground truth data with {len(ground_truth)} entries.")


    ground_truth_position = []
    estimated_position = []

    estimated_pose = []
    current_pose = np.eye(4)
    current_pose[:3, :3] = ground_truth_mat[0, :3, :3]
    estimated_position.append(current_pose[:3, 3])
    ground_truth_position.append(gt_position[0])

    vo_relative_pose = []

    for i, image_file in enumerate(tqdm(image_files)):
        idx = i + 1
        frame_path = os.path.join(image_path, image_file)
        frame = cv2.imread(frame_path)
        vo_output = vo.compute_pose(ImageData(image=frame, timestamp=time.time()))
        if vo_output.success:
            pose = vo_output.relative_pose
            current_pose = current_pose @ pose
            estimated_pose.append(current_pose[:3, :].flatten())

            estimated_position.append(current_pose[:3, 3])
            if idx < len(gt_position):
                ground_truth_position.append(gt_position[idx])
            vo_relative_pose.append(pose.flatten())


    estimated_pose = np.array(estimated_pose)
    
    ground_truth_position = np.array(ground_truth_position)
    estimated_position = np.array(estimated_position)
    min_len = min(len(ground_truth_position), len(estimated_position))
    ground_truth_position = ground_truth_position[:min_len]
    estimated_position = estimated_position[:min_len]

    vo_relative_pose = np.array(vo_relative_pose)

    logging.info(f"Estimated Positions shape: {ground_truth_position.shape}")
    logging.info(f"Estimated Pose shape: {estimated_position.shape}")

    mae = mean_absolute_error(ground_truth_position, estimated_position)
    logging.info(f"Mean Absolute Error (MAE) of the estimated positions: {mae:.4f}m")


    abs_output_filename = os.path.join(output_dir, f"absolute_pose/{driving_date}/{sequence_nr}/data.csv")
    relative_output_filename = os.path.join(output_dir, f"relative_pose/{driving_date}/{sequence_nr}/data.csv")
    output_image_filename = os.path.join(output_dir, f"images/{variant}.png")

    abs_csv_dir_name = os.path.dirname(abs_output_filename)
    rel_csv_dir_name = os.path.dirname(relative_output_filename)
    image_dir_name = os.path.dirname(output_image_filename)
    
    if not os.path.exists(abs_csv_dir_name):
        os.makedirs(abs_csv_dir_name, exist_ok=True)
    if not os.path.exists(rel_csv_dir_name):
        os.makedirs(rel_csv_dir_name, exist_ok=True)
    if not os.path.exists(image_dir_name):
        os.makedirs(image_dir_name, exist_ok=True)

    df = pd.DataFrame(estimated_pose)
    df.to_csv(abs_output_filename, index=False, header=False)

    df = pd.DataFrame(vo_relative_pose)
    df.to_csv(relative_output_filename, index=False, header=False)

    plt.figure(figsize=(10, 8))
    px, py, pz = ground_truth_position.T
    plt.plot(px, pz, marker='o', markersize=1, label='Ground Truth Trajectory', color='black')
    px, py, pz = estimated_position.T
    plt.plot(px, pz, marker='o', markersize=1, label='Estimated Trajectory', color='blue')
    plt.title(f'Trajectory estimation (seq: {variant})')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.axis('equal')
    plt.grid()
    plt.legend()
    plt.savefig(output_image_filename)

if __name__ == "__main__":

    args = parse_args()
    setup_logging(log_level='INFO', log_output='.debugging/export_vo_estimate_log')

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

    os.makedirs(args.output_path, exist_ok=True)

    variants = [
        # "01",
        # "02",
        "03",
        "04",
        # "05",
        # "06",
        "07",
        # "08",
        "09",
        # "10"
    ]

    for variant in variants:

        config = VisualOdometryConfig.from_json(vo_json_config)
        config.type = "monocular"
        config.estimator = "2d2d"

        run_vo(
            rootpath=args.dataset_path,
            variant=variant,
            output_dir=args.output_path,
            config=config
        )