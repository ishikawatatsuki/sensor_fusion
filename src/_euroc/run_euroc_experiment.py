import os
import sys
import cv2
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
    State,
    ImageData,
    FusionData,
    KITTI_SensorType,
    CoordinateFrame,
    SensorType,
    SensorConfig,
    SensorDataField,
    ExtendedConfig,
    FilterConfig,
    DatasetConfig,

    dump_config
)
from ..utils.geometric_transformer import TransformationField
from ..common.constants import KITTI_SEQUENCE_MAPS

Result = namedtuple('Field', ['gt_pose', 'estimated_pose', 'gt_position', 'estimated_position', 'vo_position', 'mae', 'ate', 'rpe_m', 'rpe_deg', 'inference_time_update', 'inference_measurement_update'])

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
    ax.plot(estimated_positions[:, 0], estimated_positions[:, 1], label='Fusion estimate(IMU+VO velocity+Leica)', color='blue')

    # if vo_positions is not None:
    #     ax.plot(vo_positions[:, 0], vo_positions[:, 1], label='Visual Odometry estimate', color='red', alpha=0.5)

    ax.set_title(title, size=16)
    ax.set_xlabel('X [m]', fontsize=16)
    ax.set_ylabel('Y [m]', fontsize=16)
    ax.legend(prop={'size': 14})
    ax.grid()
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.axis('equal')
    plt.savefig(f"{output_dir}/{title}_trajectory.png")
    plt.close(fig)


class KittiExperimentalPipeline(SingleThreadedPipeline):
    
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
                    logging.warning("Dataset queue is empty")
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
                
                elif SensorType.is_stereo_image_data(sensor_data.type):
                    if self.config.dataset.should_run_visual_odometry:
                        # Run visual odometry
                        frame = cv2.imread(sensor_data.data.left_frame_id)
                        image_data = ImageData(image=frame, timestamp=sensor_data.timestamp)
                        vo_response = self.visual_odometry.compute_pose(data=image_data)
                        if vo_response.success:
                            _sensor_data = SensorDataField(
                                type=self.visual_odometry.get_datatype, 
                                timestamp=vo_response.estimate_timestamp, 
                                data=vo_response,
                                coordinate_frame=CoordinateFrame.STEREO_LEFT)
                            _, duration = self.sensor_fusion.run_measurement_update(_sensor_data)
                            measurement_update_step_durations.append(duration)

                elif SensorType.is_reference_data(sensor_data.type):
                    # For visualization
                    if self.config.dataset.type != 'euroc':
                        continue

                    z = sensor_data.data.z # 1x10 vector in inertial frame
                    gt_p, gt_q = z[:3], z[6:]
                    gt_R = State.get_rotation_matrix_from_quaternion_vector(gt_q)
                    gt_inertial = np.eye(4)
                    gt_inertial[:3, :3] = gt_R
                    gt_inertial[:3, 3] = gt_p
                    gt_inertial = gt_inertial[:3, :] # 3x4 matrix in inertial frame
                    
                    current_estimate = self.sensor_fusion.kalman_filter.get_current_estimate().matrix(pose_only=True) # inertial frame
                    current_vo_estimate = self.sensor_fusion.independent_vo_pose.matrix(pose_only=True) if self.sensor_fusion.independent_vo_pose is not None else np.eye(4)[:3, :4] # inertial frame

                    gt_position_inertial = gt_inertial[:3, 3]
                    estimated_position_inertial = current_estimate[:3, 3]
                    vo_position_inertial = current_vo_estimate[:3, 3]
                    
                    gt_pose_camera_flat = gt_inertial.flatten() # 12 elements
                    estimated_pose_camera_flat = current_estimate.flatten() # 12 elements

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
            self.dataset.stop()
            logging.info("Process finished.")

        gt_position = np.array(gt_position)
        estimated_position = np.array(estimated_position)
        vo_position = np.array(vo_position)
        min_length_for_position = min(min(len(gt_position), len(estimated_position)), len(vo_position))
        gt_position = gt_position[:min_length_for_position]
        estimated_position = estimated_position[:min_length_for_position]
        vo_position = vo_position[:min_length_for_position]


        gt_pose = np.array(gt_pose)
        estimated_pose = np.array(estimated_pose)
        min_length = min(len(gt_pose), len(estimated_pose))
        gt_pose = gt_pose[:min_length]
        estimated_pose = estimated_pose[:min_length]

        if is_diverged:
            return Result(
                gt_pose=gt_pose,
                estimated_pose=estimated_pose,
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
        
        # Compute average inference times
        inference_time_update_step = np.mean(time_update_step_durations) if time_update_step_durations else 0
        inference_measurement_update_step = np.mean(measurement_update_step_durations) if measurement_update_step_durations else 0

        ate, rpe_m, rpe_deg = evaluate_kitti_odom(gt_pose.reshape(-1, 3, 4), estimated_pose.reshape(-1, 3, 4))
        mae = mean_absolute_error(gt_pose.reshape(-1, 3, 4)[:, :3, 3], estimated_pose.reshape(-1, 3, 4)[:, :3, 3])

        return Result(
            gt_pose=gt_pose,
            estimated_pose=estimated_pose,
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
    args = parser.parse_args()

    setup_logging(log_level="INFO", log_output=args.log_output)
    logger = logging.getLogger(__name__)

    summary_filepath = os.path.join(args.output_path, "euroc_experiment.csv")
    summary_df = pd.DataFrame(columns=[
        "sequence", "filter_type", "motion_model", "sensor", "vo_estimation_type", "mae",
        "ate", "rpe_m", "rpe_deg", "avg_inference_time_prediction_step", "avg_inference_time_correction_step"
    ])
    
    config_file_path = "./configs/euroc_config_experiments.yaml"

    experiment_config = ExtendedConfig(config_file_path)

    seq = experiment_config.dataset.variant
    filter_type = experiment_config.filter.type
    sensor_str = "vo_vel_leica"

    output_dir = f"{args.output_path}/seq{seq}/{filter_type}_kinematic/{sensor_str}"
    os.makedirs(output_dir, exist_ok=True)

    pipeline = KittiExperimentalPipeline(experiment_config)
    result = pipeline.run()

    save_trajectory_plots(
        gt_positions=result.gt_position,
        estimated_positions=result.estimated_position,
        vo_positions=result.vo_position,
        title=f"CKF Fusion Result on EuRoC MAV Seq. MH_01",
        output_dir=output_dir
    )

    with open(os.path.join(output_dir, "ground_truth_pose.txt"), "w") as f1:
        for pose in result.gt_pose:
            pose_str = ' '.join(map(str, pose))
            f1.write(f"{pose_str}\n")
    with open(os.path.join(output_dir, "estimated_pose.txt"), "w") as f2:
        for pose in result.estimated_pose:
            pose_str = ' '.join(map(str, pose))
            f2.write(f"{pose_str}\n")

    # Append results to the summary DataFrame
    summary_df = pd.concat([summary_df, pd.DataFrame([{
        "sequence": seq,
        "filter_type": filter_type,
        "motion_model": "velocity",
        "sensor": sensor_str,
        "vo_estimation_type": "2d3d",
        "mae": result.mae,
        "ate": result.ate,
        "rpe_m": result.rpe_m,
        "rpe_deg": result.rpe_deg,
        "avg_inference_time_prediction_step": result.inference_time_update,
        "avg_inference_time_correction_step": result.inference_measurement_update
    }])], ignore_index=True)


    dump_config(experiment_config.filter, experiment_config.dataset, experiment_config.visual_odometry, os.path.join(output_dir, "config.yaml"))

    summary_df.to_csv(summary_filepath, index=False)
    logging.info(f"Experiment results saved to {summary_filepath}")