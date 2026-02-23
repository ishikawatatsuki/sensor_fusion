import os
import sys
import yaml
import cv2
import logging
import argparse
from time import sleep
import numpy as np
import pandas as pd
from tqdm import tqdm
from enum import Enum
from typing import List
from itertools import product
from collections import namedtuple
from sklearn.metrics import mean_absolute_error

from ..common.config import (
  Config,
  SensorConfig,
  FilterConfig,
  SensorNoiseConfig,
)
from ..common.constants import KITTI_SEQUENCE_MAPS
from ..common.datatypes import Pose, State, KITTI_SensorType
from ..internal.extended_common.extended_config import DatasetConfig, ExtendedConfig
from ..internal.extended_common import (
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

from ..pipeline import SingleThreadedPipeline
from ..misc import setup_logging
from ..internal.error_reporter.kitti_error_reporter import evaluate_kitti_odom
from .mics import save_trajectory_plots


Result = namedtuple('Field', ['gt_pose', 'estimated_pose', 'gt_position', 'estimated_position', 'vo_position', 'mae', 'ate', 'rpe_m', 'rpe_deg', 'inference_time_update', 'inference_measurement_update'])

def get_args():
    parser = argparse.ArgumentParser(description="Run noise stress test on KITTI dataset")
    parser.add_argument('--output_path', type=str, required=True, help='Path to output CSV file for storing results')
    parser.add_argument('--log_output', type=str, default=os.path.join(os.path.dirname(__file__), 'configs/experiments/kitti_noise_stress_test.yaml'), help='Path to experimental configuration YAML file')

    return parser.parse_args()

def get_experimental_config(
        config_path: str,
        imu_noise_scale: float,
        vo_noise_scale: float,
        gps_noise_scale: float
) -> ExtendedConfig:
        
    config = ExtendedConfig(config_path)
    
    sensors = config.filter.sensors.copy()
    for sensor_type, values in config.filter.sensors.items():
        noise_json = values['noise'].to_json()
        if sensor_type == KITTI_SensorType.OXTS_IMU:
            noise_json['scale'] = imu_noise_scale
        elif sensor_type == KITTI_SensorType.KITTI_VO:
            noise_json['scale'] = vo_noise_scale
        elif sensor_type == KITTI_SensorType.OXTS_GPS:
            noise_json['scale'] = gps_noise_scale

        sensors[sensor_type] = {
            'fields': values['fields'],
            'noise': SensorNoiseConfig.from_json(noise_json)
        }
    
    config.filter.sensors = sensors
    return config

def validate_noise_scale_factors(
        filter_config: FilterConfig,
        imu_scale_factor: float,
        vo_scale_factor: float,
        gps_scale_factor: float
    ):

    for sensor_type, sensor_values in filter_config.sensors.items():
        noise_config = sensor_values['noise']
        if sensor_type == KITTI_SensorType.OXTS_IMU:
            expected_scale = imu_scale_factor
        elif sensor_type == KITTI_SensorType.KITTI_VO:
            expected_scale = vo_scale_factor
        elif sensor_type == KITTI_SensorType.OXTS_GPS:
            expected_scale = gps_scale_factor
        else:
            continue
        
        actual_scale = noise_config.scale
        assert np.isclose(actual_scale, expected_scale), f"Noise scale factor mismatch for {sensor_type.name}: expected {expected_scale}, got {actual_scale}"
        print(f"✓ Noise scale factor for {sensor_type.name} is correct: {actual_scale}")



class KittiNoiseStressTestPipeline(SingleThreadedPipeline):
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
    

def run_adaptive_noise_management_experiment(
        config_filepath: str, 
        output_filepath: str
    ) -> pd.DataFrame:
    
    config = ExtendedConfig(config_filepath)
    adaptive_filter_config_path = "configs/.experiments/kitti_stress_test/filter_config_adaptive.yaml"
    config.filter = FilterConfig.from_yaml(adaptive_filter_config_path)
    config.filter.set_sensor_fields(config.dataset.type)

    output_dir = f"{output_filepath}/{config.filter.type}_velocity/adaptive"
    os.makedirs(output_dir, exist_ok=True)
    
    result = KittiNoiseStressTestPipeline(config).run()

    # Dummy result for testing visualization and result saving
    # result = Result(
    #     gt_pose=np.random.rand(100, 12),
    #     estimated_pose=np.random.rand(100, 12),
    #     gt_position=np.random.rand(100, 3),
    #     estimated_position=np.random.rand(100, 3),
    #     vo_position=np.random.rand(100, 3),
    #     mae=np.random.rand(),
    #     ate=np.random.rand(),
    #     rpe_m=np.random.rand(),
    #     rpe_deg=np.random.rand(),
    #     inference_time_update=np.random.rand(),
    #     inference_measurement_update=np.random.rand()
    # )
    
    save_trajectory_plots(
        gt_positions=result.gt_position,
        estimated_positions=result.estimated_position,
        vo_positions=result.vo_position,
        title=f"{config.filter.type}_velocity_{scale_str}",
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

    dump_config(
        config.filter, 
        config.dataset, 
        config.visual_odometry, 
        os.path.join(output_dir, "config.yaml"))

    return pd.DataFrame([{
        "sequence": "kitti_07",
        "filter_type": config.filter.type,
        "motion_model": "velocity",
        "sensor": "IMU+VO+GPS",
        "vo_estimation_type": "2d3d",
        "mae": result.mae,
        "ate": result.ate,
        "rpe_m": result.rpe_m,
        "rpe_deg": result.rpe_deg,
        "avg_inference_time_prediction_step": result.inference_time_update,
        "avg_inference_time_correction_step": result.inference_measurement_update,
        "imu_noise_scale": 1.0,
        "vo_noise_scale": 1.0,
        "gps_noise_scale": 1.0,
        "is_adaptive_noise_management": True,
    }])

if __name__ == "__main__":
   

    args = get_args()

    setup_logging(log_level="INFO", log_output=args.log_output)
    logger = logging.getLogger(__name__)

    config_filepath = "configs/.experiments/kitti_stress_test/noise_stress_test.yaml"

    IMU_SCALE_FACTOR = [0.1, 1.0, 10.0]
    VO_SCALE_FACTOR = [0.1, 1.0, 10.0]
    GPS_SCALE_FACTOR = [0.1, 1.0, 10.0]

    logging.info("Starting KITTI Noise Stress Test Pipeline...")
    
    summary_filepath = os.path.join(args.output_path, "noise_stress_test_results_seq_07.csv")
    summary_df = pd.DataFrame(columns=[
        "sequence", "filter_type", "motion_model", "sensor", "vo_estimation_type", "mae",
        "ate", "rpe_m", "rpe_deg", "avg_inference_time_prediction_step", "avg_inference_time_correction_step",
        "imu_noise_scale", "vo_noise_scale", "gps_noise_scale", "is_adaptive_noise_management"
    ])

    results = []
    combinations = list(product(IMU_SCALE_FACTOR, VO_SCALE_FACTOR, GPS_SCALE_FACTOR))
    for i, scale_factor in tqdm(enumerate(product(IMU_SCALE_FACTOR, VO_SCALE_FACTOR, GPS_SCALE_FACTOR)), total=len(combinations), desc="Running Noise Stress Test"):
        logging.info(f"Running ({i}): IMU Scale: {scale_factor[0]}, VO Scale: {scale_factor[1]}, GPS Scale: {scale_factor[2]}")

        scale_str = f"imu_{scale_factor[0]}_vo_{scale_factor[1]}_gps_{scale_factor[2]}"

        experimental_config = get_experimental_config(
            config_filepath, 
            imu_noise_scale=scale_factor[0], 
            vo_noise_scale=scale_factor[1], 
            gps_noise_scale=scale_factor[2])
        
        output_dir = f"{args.output_path}/{experimental_config.filter.type}_velocity/{scale_str}"
        os.makedirs(output_dir, exist_ok=True)
        # validate_noise_scale_factors(
        #     experimental_config.filter, 
        #     imu_scale_factor=scale_factor[0],
        #     vo_scale_factor=scale_factor[1],
        #     gps_scale_factor=scale_factor[2]
        # )
        print("--------------------------------------------------")
        print(experimental_config.dataset)
        print(experimental_config.filter)

        result = KittiNoiseStressTestPipeline(experimental_config).run()

        # Dummy result for testing visualization and result saving
        # result = Result(
        #     gt_pose=np.random.rand(100, 12),
        #     estimated_pose=np.random.rand(100, 12),
        #     gt_position=np.random.rand(100, 3),
        #     estimated_position=np.random.rand(100, 3),
        #     vo_position=np.random.rand(100, 3),
        #     mae=np.random.rand(),
        #     ate=np.random.rand(),
        #     rpe_m=np.random.rand(),
        #     rpe_deg=np.random.rand(),
        #     inference_time_update=np.random.rand(),
        #     inference_measurement_update=np.random.rand()
        # )
        
        save_trajectory_plots(
            gt_positions=result.gt_position,
            estimated_positions=result.estimated_position,
            vo_positions=result.vo_position,
            title=f"{experimental_config.filter.type}_velocity_{scale_str}",
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
        
        summary_df = pd.concat([summary_df, pd.DataFrame([{
            "sequence": "kitti_07",
            "filter_type": experimental_config.filter.type,
            "motion_model": "velocity",
            "sensor": "IMU+VO+GPS",
            "vo_estimation_type": "2d3d",
            "mae": result.mae,
            "ate": result.ate,
            "rpe_m": result.rpe_m,
            "rpe_deg": result.rpe_deg,
            "avg_inference_time_prediction_step": result.inference_time_update,
            "avg_inference_time_correction_step": result.inference_measurement_update,
            "imu_noise_scale": scale_factor[0],
            "vo_noise_scale": scale_factor[1],
            "gps_noise_scale": scale_factor[2],
            "is_adaptive_noise_management": False,
        }])], ignore_index=True)

        dump_config(
            experimental_config.filter, 
            experimental_config.dataset, 
            experimental_config.visual_odometry, 
            os.path.join(output_dir, "config.yaml"))
        
    # Run adaptive noise experiment with default noise scales (1.0) since the adaptive filter will adjust internally
    adaptive_result_df = run_adaptive_noise_management_experiment(
        config_filepath=config_filepath, 
        output_filepath=args.output_path
    )
    summary_df = pd.concat([summary_df, adaptive_result_df], ignore_index=True)

    summary_df.to_csv(summary_filepath, index=False)
    logging.info(f"Experiment results saved to {summary_filepath}")

