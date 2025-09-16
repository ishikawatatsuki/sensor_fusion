""" Run experiments on KITTI dataset sequences.
    This scripts conduct experiments on all sequences in the KITTI dataset
    and saves the results in a CSV file.
    The experimental setups are the followings:
    - different sequences of KITTI dataset
    - different variants of Kalaman Filter algorithms (e.g., EKF, UKF)
    - different motion models (e.g., kinematics, velocity)
    - different sensor configurations:
        - IMU only
        - IMU + GPS
        - IMU + VO (position)
        - IMU + VO (Velocity)
        - IMU + GPS (position) + VO (position)
        - IMU + GPS (position) + VO (Velocity)

    The objectives are to evaluate the performance of different Kalman Filter
    algorithms and sensor configurations on the KITTI dataset sequences by comparing their accuracy, robustness, and computational efficiency (inference time).
    The results are saved in a csv file for further analysis and visualization.
"""

from logging import config
import os
import sys
import logging
from time import sleep
import numpy as np
import pandas as pd
from tqdm import tqdm
from enum import Enum
from typing import List
from itertools import product
from collections import namedtuple
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

from ..misc import setup_logging
from ..internal.error_reporter.kitti_error_reporter import evaluate_kitti_odom
from ..pipeline import SingleThreadedPipeline
from ..internal.extended_common import (
    FusionData,
    KITTI_SensorType,
    CoordinateFrame,
    SensorType,
    SensorConfig,
    ExtendedConfig,
    FilterConfig,
    DatasetConfig
)
from ..utils.geometric_transformer import TransformationField
from ..common.constants import KITTI_SEQUENCE_MAPS

Result = namedtuple('Field', ['gt_position', 'estimated_position', 'vo_position', 'mae', 'ate', 'rpe_m', 'rpe_deg', 'inference_time_update', 'inference_measurement_update'])

SENSOR_MAPS = {
    'imu_vo_pos': {
        'oxts_imu': {'fields': ['linear_acceleration', 'angular_velocity']},
        'kitti_vo': {'fields': ['position']}
    },
    'imu_vo_vel': {
        'oxts_imu': {'fields': ['linear_acceleration', 'angular_velocity']},
        'kitti_vo': {'fields': ['linear_velocity']}
    },
    'imu_gps': {
        'oxts_imu': {'fields': ['linear_acceleration', 'angular_velocity']},
        'oxts_gps': {'fields': ['position']}
    },
    'imu_gps_vo_pos': {
        'oxts_imu': {'fields': ['linear_acceleration', 'angular_velocity']},
        'oxts_gps': {'fields': ['position']},
        'kitti_vo': {'fields': ['position']}
    },
    'imu_gps_vo_vel': {
        'oxts_imu': {'fields': ['linear_acceleration', 'angular_velocity']},
        'oxts_gps': {'fields': ['position']},
        'kitti_vo': {'fields': ['linear_velocity']}
    },
}

class ExperimentalPipeline(SingleThreadedPipeline):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self):

        self.dataset.start()
        
        sleep(0.5)
        logging.info("Starting the process.")
        time_update_step_durations = []
        measurement_update_step_durations = []

        gt_position = []
        estimated_position = []
        vo_position = []

        gt_pose = []
        estimated_pose = []

        is_diverged = False
        
        try:
            while True:
                if self.dataset.is_queue_empty():
                    logging.debug("Dataset queue is empty")
                    break

                sensor_data = self.dataset.get_sensor_data()
                if sensor_data is None:
                    continue

                # logging.debug(f"Processing sensor data: {sensor_data.type} at time {sensor_data.timestamp}")
                
                if SensorType.is_time_update(sensor_data.type):
                    _, duration = self.sensor_fusion.run_time_update(sensor_data)
                    time_update_step_durations.append(duration)
                    

                elif SensorType.is_measurement_update(sensor_data.type):
                    _, duration = self.sensor_fusion.run_measurement_update(sensor_data)
                    measurement_update_step_durations.append(duration)
                
                elif SensorType.is_reference_data(sensor_data.type):
                    # For visualization
                    if self.config.dataset.type != 'kitti':
                        continue

                    z = sensor_data.data.z # 3x4 matrix in camera frame
                    gt_inertial = self.sensor_fusion.geo_transformer.transform(fields=TransformationField(
                        state=self.sensor_fusion.kalman_filter.x,
                        value=z,
                        coord_from=CoordinateFrame.STEREO_LEFT,
                        coord_to=CoordinateFrame.INERTIAL))
                    current_estimate = self.sensor_fusion.kalman_filter.get_current_estimate().matrix(pose_only=True) # inertial frame
                    current_vo_estimate = self.sensor_fusion.independent_vo_pose.matrix(pose_only=True) if self.sensor_fusion.independent_vo_pose is not None else np.eye(4)[:3, :4] # inertial frame

                    gt_position_inertial = gt_inertial[:3, 3]
                    estimated_position_inertial = current_estimate[:3, 3]
                    vo_position_inertial = current_vo_estimate[:3, 3]

                    # estimation in camera coordinate frame
                    estimated_pose_camera = self.sensor_fusion.geo_transformer.transform(fields=TransformationField(
                            state=self.sensor_fusion.kalman_filter.x,
                            value=current_estimate,
                            coord_from=CoordinateFrame.INERTIAL,
                            coord_to=CoordinateFrame.STEREO_LEFT))
                    
                    estimated_pose_camera_flat = estimated_pose_camera.flatten() # 12 elements
                    gt_pose_camera_flat = z[:3, :].flatten() # 12 elements

                    gt_pose.append(gt_pose_camera_flat)
                    estimated_pose.append(estimated_pose_camera_flat)

                    # divergence check
                    if np.linalg.norm(estimated_position_inertial - gt_position_inertial) > 1e6:
                        logging.warning("Divergence detected. Stopping the process.")
                        is_diverged = True
                        break

                    gt_position.append(gt_position_inertial.flatten())
                    estimated_position.append(estimated_position_inertial.flatten())
                    vo_position.append(vo_position_inertial.flatten())

        except Exception as e:
            logging.error(e)
            logging.error(f"Data remaining in queue: {self.dataset.get_queue_size()}")
        finally:
            f.close()
            self.dataset.stop()
            logging.info("Process finished.")

        gt_position = np.array(gt_position)
        estimated_position = np.array(estimated_position)
        vo_position = np.array(vo_position)
        min_length_for_position = min(min(len(gt_position), len(estimated_position)), len(vo_position))
        gt_position = gt_position[:min_length_for_position]
        estimated_position = estimated_position[:min_length_for_position]
        vo_position = vo_position[:min_length_for_position]

        if is_diverged:
            return Result(
                gt_position=gt_position,
                estimated_position=estimated_position,
                vo_position=vo_position,
                mae=np.inf,
                ate=np.inf,
                rpe_m=np.inf,
                rpe_deg=np.inf,
                inference_time_update=np.inf,
                inference_measurement_update=np.inf
                )
        
        inference_time_update_step = np.mean(time_update_step_durations) if time_update_step_durations else 0
        inference_measurement_update_step = np.mean(measurement_update_step_durations) if measurement_update_step_durations else 0

        gt_pose = np.array(gt_pose)
        estimated_pose = np.array(estimated_pose)
        min_length = min(len(gt_pose), len(estimated_pose))
        gt_pose = gt_pose[:min_length]
        estimated_pose = estimated_pose[:min_length]

        with open("gt_pose.txt", "w") as f1:
            for pose in gt_pose:
                pose_str = ' '.join(map(str, pose))
                f1.write(f"{pose_str}\n")
        with open("estimated_pose.txt", "w") as f2:
            for pose in estimated_pose:
                pose_str = ' '.join(map(str, pose))
                f2.write(f"{pose_str}\n")

        ate, rpe_m, rpe_deg = evaluate_kitti_odom(gt_pose.reshape(-1, 3, 4), estimated_pose.reshape(-1, 3, 4))
        mae = mean_absolute_error(gt_pose.reshape(-1, 3, 4)[:, :3, 3], estimated_pose.reshape(-1, 3, 4)[:, :3, 3])

        return Result(
            gt_position=gt_position,
            estimated_position=estimated_position,
            vo_position=vo_position,
            mae=mae,
            ate=ate,
            rpe_m=rpe_m,
            rpe_deg=rpe_deg,
            inference_time_update=inference_time_update_step,
            inference_measurement_update=inference_measurement_update_step
        )

def save_trajectory_plots(
        gt_positions: np.ndarray, 
        estimated_positions: np.ndarray, 
        vo_positions: np.ndarray, 
        title: str = "Trajectory",
        output_dir: str = "."
    ):
    """ Visualize the trajectory of the estimated positions against the ground truth positions.

    Args:
        gt_positions (np.ndarray): Ground truth positions (N x 3).
        estimated_positions (np.ndarray): Estimated positions (N x 3).
        vo_positions (np.ndarray): Visual Odometry positions (N x 3).
        title (str): Title of the plot.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(gt_positions[:, 0], gt_positions[:, 1], label='Ground Truth trajectory', color='black', linestyle='--')
    ax.plot(estimated_positions[:, 0], estimated_positions[:, 1], label='Fusion estimate', color='blue')

    if vo_positions is not None and np.sum(vo_positions) > 10:
        ax.plot(vo_positions[:, 0], vo_positions[:, 1], label='Visual Odometry estimate', color='red', alpha=0.5)

    ax.set_title(title)
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.legend()
    ax.grid()
    plt.axis('equal')
    plt.savefig(f"{output_dir}/{title}_trajectory.png")
    plt.close(fig)

def create_config(config_file: str, sequence: str, filter_type: str, motion_model: str, sensor: str, vo_estimation_type: str) -> ExtendedConfig:
    
    config = ExtendedConfig(config_file)

    selected_sensors = SENSOR_MAPS.get(sensor, {})
    if filter_type == "ekf_simple_noise" or filter_type == "ekf_correct" or filter_type == "ekf_correct_simple_noise":
        filter_type = "ekf"
    
    filter_config = config.filter
    filter_config.type = filter_type
    filter_config.motion_model = motion_model

    sensors = {}
    sensors_for_dataset = []
    fusion_data_fields = FusionData.get_enum_name_list()
    for sensor_type_str, values in selected_sensors.items():
        sensor_type = KITTI_SensorType.get_kitti_sensor_from_str(sensor_type_str)
        fields = values.get("fields", [])
        if sensor_type is not None:
            sensors[sensor_type] = [FusionData.get_type(field) for field in fields if field in fusion_data_fields]

            if sensor_type == KITTI_SensorType.OXTS_IMU:
                args = {
                    'frequency': 10,
                    'gyroscope_noise_density': 5.817764173314432e-05,
                    'accelerometer_noise_density': 8.333333333333333e-05,
                    'gyroscope_random_walk': 0.0005817764173314433,
                    'accelerometer_random_walk': 0.0008333333333333333
                }
            elif sensor_type == KITTI_SensorType.OXTS_GPS:
                args = {
                    'frequency': 10,
                }
            elif sensor_type == KITTI_SensorType.KITTI_VO:
                args = {
                    'frequency': 10,
                    'estimation_type': vo_estimation_type  # Select form: epipolar, pnp, hybrid, stereo
                }
            else:
                args = {}
            sensors_for_dataset.append(SensorConfig(
                name=sensor_type_str,
                dropout_ratio=0,
                window_size=1,
                args=args
            ))

    filter_config.sensors = sensors
    dataset_config = config.dataset
    dataset_config.variant = sequence
    dataset_config.sensors = sensors_for_dataset

    config.filter = filter_config
    config.dataset = dataset_config

    config.hardware = config._get_sensor_hardware_config()

    return config

def dump_config(filter_config: FilterConfig, dataset_config: DatasetConfig, output_filepath: str):
    """ Dump the configuration to a YAML file.

    Args:
        config (ExtendedConfig): Configuration object.
        output_filepath (str): Path to the output YAML file.
    """
    import yaml
    with open(output_filepath, 'w') as f:
        yaml.dump({"filter": filter_config.to_dict(), "dataset": dataset_config.to_dict()}, f)
    logging.info(f"Configuration dumped to {output_filepath}")

if __name__ == "__main__":
    # Define the experiment configurations
    import argparse
    parser = argparse.ArgumentParser(
        description='Run all KITTI experiments with different configurations.')
    parser.add_argument(
        '--output_path',
        type=str,
        required=True,
        help='Path to save the output CSV file with results.')
    parser.add_argument(
        '--log_output',
        type=str,
        default='../_debugging/',
        help='Path to the log output directory.')
    parser.add_argument(
        '--config_file',
        type=str,
        default="./configs/config_uav.yaml",
        help='Path to the configuration file.')
    parser.add_argument(
        '--checkpoint_file',
        type=str,
        default=None,
        help='Path to the checkpoint file.')
    parser.add_argument(
        '--summary_file',
        type=str,
        default="experiment_results.csv",
        help='Filename for the summary CSV file.'
    )
    args = parser.parse_args()

    setup_logging(log_level="INFO", log_output=args.log_output)
    logger = logging.getLogger(__name__)

    sequences = ["04", "06", "07", "09"]  # Different KITTI sequences
    # filter_types = ["ekf", "ukf", "pf", "enkf", "ckf"]  # Different Kalman Filter variants
    # motion_models = ["kinematics", "velocity"]  # Different motion models
    sensors = SENSOR_MAPS.keys()  # ["imu_vo_pos", "imu_vo_vel", "imu_gps_vo_pos", "imu_gps_vo_vel"]
    vo_estimation_types = ["epipolar", "pnp", "hybrid", "stereo"]  # Different VO estimation types


    # sequences = ["04"]  # Different KITTI sequences
    filter_types = ["ekf_correct_simple_noise"]  # Different Kalman Filter variants
    motion_models = ["kinematics", "velocity"]
    # motion_models = ["kinematics"]  # Different motion models
    # sensors = ["imu_gps_vo_pos"]  # ["imu_vo_pos", "imu_vo_vel", "imu_gps_vo_pos", "imu_gps_vo_vel"]
    # vo_estimation_types = ["hybrid"]  # Different VO estimation types

    already_done = set()
    checkpoint_path = args.checkpoint_file
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            for line in f:
                already_done.add(line.strip())
        logger.info(f"Already done experiments: {len(already_done)}")
    else:
        # create checkpoint file
        with open(checkpoint_path, "w") as f:
            f.write("")

    if checkpoint_path is not None:
        checkpoint_path = os.path.join(args.output_path, "checkpoint.txt")

    summary_filepath = os.path.join(args.output_path, args.summary_file)
    if os.path.exists(summary_filepath):
        summary_df = pd.read_csv(summary_filepath)
    else:
        summary_df = pd.DataFrame(columns=[
            "sequence", "filter_type", "motion_model", "sensor", "vo_estimation_type", "mae",
            "ate", "rpe_m", "rpe_deg", "inference_time_update", "inference_measurement_update"
        ])
        

    # Run experiments for each combination of sequence, filter type, and sensor-motion model type
    # Store at sequence/filter_type/motion_model/sensor/vo_estimation_type
    for seq, filter_type, motion_model, sensor, vo_estimation_type in tqdm(product(sequences, filter_types, motion_models, sensors, vo_estimation_types)):

        logger.info(f"Running experiment: Sequence={seq}, Filter={filter_type}, Motion Model={motion_model}, Sensor={sensor}, VO Estimation={vo_estimation_type}")
        experiment_id = f"seq{seq}_{filter_type}_{motion_model}_{sensor}_{vo_estimation_type}"

        if experiment_id in already_done:
            logger.info(f"Skipping already done experiment: {experiment_id}")
            continue

        output_dir = f"{args.output_path}/seq{seq}/{filter_type}/{motion_model}/{sensor}/{vo_estimation_type}/"
        os.makedirs(output_dir, exist_ok=True)

        experiment_config = create_config(
            config_file=args.config_file,
            sequence=seq,
            filter_type=filter_type,
            motion_model=motion_model,
            sensor=sensor,
            vo_estimation_type=vo_estimation_type
        )

        pipeline = ExperimentalPipeline(experiment_config)
        result = pipeline.run()

        save_trajectory_plots(
            gt_positions=result.gt_position,
            estimated_positions=result.estimated_position,
            vo_positions=result.vo_position,
            title=f"Seq{seq}_{filter_type}_{motion_model}_{sensor}_{vo_estimation_type}",
            output_dir=output_dir
        )

        # Append results to the summary DataFrame
        summary_df = pd.concat([summary_df, pd.DataFrame([{
            "sequence": seq,
            "filter_type": filter_type,
            "motion_model": motion_model,
            "sensor": sensor,
            "vo_estimation_type": vo_estimation_type,
            "mae": result.mae,
            "ate": result.ate,
            "rpe_m": result.rpe_m,
            "rpe_deg": result.rpe_deg,
            "inference_time_update": result.inference_time_update,
            "inference_measurement_update": result.inference_measurement_update
        }])], ignore_index=True)

        summary_df.to_csv(summary_filepath, index=False)
        logger.info(f"Experiment results appended to {summary_filepath}")

        dump_config(experiment_config.filter, experiment_config.dataset, os.path.join(output_dir, "config.yaml"))

        with open(checkpoint_path, "a") as f:
            f.write(f"{experiment_id}\n")
            f.flush()


