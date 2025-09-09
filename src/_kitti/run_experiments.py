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

import os
import sys
import logging
from time import sleep
import numpy as np
import pandas as pd
from tqdm import tqdm
from enum import Enum
from typing import List

from ..pipeline import SingleThreadedPipeline
from ..internal.extended_common import (
    CoordinateFrame,
    SensorType,
    SensorConfig,
    ExtendedConfig,
    ExtendedSensorConfig
)
from ..utils.geometric_transformer import TransformationField
from ..common.constants import KITTI_SEQUENCE_MAPS


logger = logging.getLogger(__name__)
if __name__ == "__main__":
    logging.basicConfig(format='[%(asctime)s] [%(levelname)5s] > %(message)s (%(filename)s:%(lineno)s)', 
                        datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)


class SensorMotionModelType(Enum):
    KINEMATICS_IMU_ONLY = "KINEMATICS_IMU_ONLY"
    KINEMATICS_IMU_GPS = "KINEMATICS_GPS"
    KINEMATICS_IMU_VO_POS = "KINEMATICS_VO_POS"
    KINEMATICS_IMU_VO_VEL = "KINEMATICS_VO_VEL"
    KINEMATICS_IMU_GPS_VO_POS = "KINEMATICS_IMU_GPS_VO_POS"
    KINEMATICS_IMU_GPS_VO_VEL = "KINEMATICS_IMU_GPS_VO_VEL"

    VELOCITY_IMU_ONLY = "VELOCITY_IMU_ONLY"
    VELOCITY_IMU_GPS = "VELOCITY_IMU_GPS"
    VELOCITY_IMU_VO_POS = "VELOCITY_IMU_VO_POS"
    VELOCITY_IMU_VO_VEL = "VELOCITY_IMU_VO_VEL"
    VELOCITY_IMU_GPS_VO_POS = "VELOCITY_IMU_GPS_VO_POS"
    VELOCITY_IMU_GPS_VO_VEL = "VELOCITY_IMU_GPS_VO_VEL"

    @staticmethod
    def get_motion_model(type_str: str) -> str:
        try:
            t = SensorMotionModelType(type_str.upper())
            match (t):
                case SensorMotionModelType.KINEMATICS_IMU_ONLY |\
                     SensorMotionModelType.KINEMATICS_IMU_GPS |\
                     SensorMotionModelType.KINEMATICS_IMU_VO_POS |\
                     SensorMotionModelType.KINEMATICS_IMU_VO_VEL |\
                     SensorMotionModelType.KINEMATICS_IMU_GPS_VO_POS |\
                     SensorMotionModelType.KINEMATICS_IMU_GPS_VO_VEL:
                    return "kinematics"
                case SensorMotionModelType.VELOCITY_IMU_ONLY |\
                     SensorMotionModelType.VELOCITY_IMU_GPS |\
                     SensorMotionModelType.VELOCITY_IMU_VO_POS |\
                     SensorMotionModelType.VELOCITY_IMU_VO_VEL |\
                     SensorMotionModelType.VELOCITY_IMU_GPS_VO_POS |\
                     SensorMotionModelType.VELOCITY_IMU_GPS_VO_VEL:
                    return "velocity"
                case _:
                    return "kinematics"
        except:
            return "kinematics"
        
    @staticmethod
    def get_error_folder_name(type_str: str) -> str:
        try:
            t = SensorMotionModelType(type_str.upper())
            return t.name.lower()
        except:
            return "not-specified"

    @staticmethod
    def get_sensor_config(type_str: str, new_dataset_config: dict) -> List[SensorConfig]:
        try:
            t = SensorMotionModelType(type_str.upper())
            is_gps_included = t is SensorMotionModelType.KINEMATICS_W_GPS or t is SensorMotionModelType.VELOCITY_W_GPS
            sensors = []
            for sensor in new_dataset_config['sensors']:
                include_gps = sensor.name == "oxts_gps" and is_gps_included
                selected = sensor.name != "oxts_gps" or include_gps
                
                if selected:
                    sensors.append(sensor)
            
            return sensors
        except:
            return new_dataset_config['sensors']


class ExperimentalPipeline(SingleThreadedPipeline):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self):

        self.dataset.start()
        self.visualizer.start()
        
        if self.config.general.log_sensor_data:
            f = open(self.config.general.sensor_data_output_filepath, "w")
        
        sleep(0.5)
        logging.info("Starting the process.")
        time_update_step_durations = []
        measurement_update_step_durations = []
        try:
            while True:
                if self.dataset.is_queue_empty():
                    logging.debug("Dataset queue is empty")
                    break

                sensor_data = self.dataset.get_sensor_data()
                if sensor_data is None:
                    continue
                
                if SensorType.is_time_update(sensor_data.type):
                    response, duration = self.sensor_fusion.run_time_update(sensor_data)
                    time_update_step_durations.append(duration)
                    

                elif SensorType.is_measurement_update(sensor_data.type):
                    response, duration = self.sensor_fusion.run_measurement_update(sensor_data)
                    measurement_update_step_durations.append(duration)
                
                elif SensorType.is_reference_data(sensor_data.type):
                    # For visualization
                    if self.config.dataset.type == "kitti":
                        value = sensor_data.data.z
                        gt_inertial = self.sensor_fusion.geo_transformer.transform(fields=TransformationField(
                            state=self.sensor_fusion.kalman_filter.x,
                            value=value,
                            coord_from=CoordinateFrame.STEREO_LEFT,
                            coord_to=CoordinateFrame.INERTIAL))
                        data = gt_inertial[:3, 3].flatten()
                    else:
                        continue
                    
                    response = self.sensor_fusion.get_current_estimate(sensor_data.timestamp)
                    

                    # NOTE: this works only for KITTI dataset
                    current_estimate = self.sensor_fusion.kalman_filter.get_current_estimate().t.flatten()
                    estimated = np.array([current_estimate[0], current_estimate[1], current_estimate[2]])
                    report1 = ReportMessage(t=VisualizationDataType.ESTIMATION, timestamp=sensor_data.timestamp, value=estimated)
                    report2 = ReportMessage(t=VisualizationDataType.GPS, timestamp=sensor_data.timestamp, value=sensor_data.data.z)

                    
                if self.config.general.log_sensor_data:
                    f.write(f"[{self.dataset.get_queue_size():05}] Sensor: {sensor_data.type.name} at {sensor_data.timestamp}\n")

        except Exception as e:
            logging.error(e)
            logging.error(f"Data remaining in queue: {self.dataset.get_queue_size()}")
        finally:
            f.close()
            self.dataset.stop()
            logging.info("Process finished.")

        # Calculate accuracy and average inference time
        accuracy = None

        inference_time_update_step = np.mean(time_update_step_durations) if time_update_step_durations else 0
        inference_measurement_update_step = np.mean(measurement_update_step_durations) if measurement_update_step_durations else 0
        total_inference_time = inference_time_update_step + inference_measurement_update_step
        


def run_filter_experiments():
    # Define the experiment configurations
    sequences = list(KITTI_SEQUENCE_MAPS.keys())  # All KITTI sequences
    filter_types = ["EKF", "UKF", "PF", "EnKF", "CKF"]  # Different Kalman Filter variants
    sensor_motion_model_types = [
        "KINEMATICS_IMU_ONLY",
        "KINEMATICS_IMU_GPS",
        "KINEMATICS_IMU_VO_POS",
        "KINEMATICS_IMU_VO_VEL",
        "KINEMATICS_IMU_GPS_VO_POS",
        "KINEMATICS_IMU_GPS_VO_VEL",
        
        "VELOCITY_IMU_ONLY",
        "VELOCITY_IMU_GPS",
        "VELOCITY_IMU_VO_POS",
        "VELOCITY_IMU_VO_VEL",
        "VELOCITY_IMU_GPS_VO_POS",
        "VELOCITY_IMU_GPS_VO_VEL"
    ]

    # Load the base configuration
    base_config = ExtendedConfig.load_from_file(os.path.join("configs", "kitti_00_imu_vo_gps_ekf.json"))
    dataset_config = base_config.dataset
    results = []

    # Run experiments for each combination of sequence, filter type, and sensor-motion model type
    for seq in tqdm(sequences, desc="Sequences"):
        for filter_type in tqdm(filter_types, desc="Filter Types", leave=False):
            for sensor_motion_model_type in tqdm(sensor_motion_model_types, desc="Sensor-Motion Models", leave=False):
                logger.info(f"Running experiment: Sequence={seq}, Filter={filter_type}, Sensor-Motion Model={sensor_motion_model_type}")

                # Update dataset configuration for the current sequence
                new_dataset_config = dataset_config.copy()
                new_dataset_config['sequence'] = seq
                new_dataset_config['sensors'] = SensorMotionModelType.get_sensor_config(sensor_motion_model_type, dataset_config)
                
                # Update the base configuration with the new dataset and filter type
                experiment_config = base_config.copy()
                experiment_config.dataset = ExtendedSensorConfig(**new_dataset_config)
                experiment_config.filter.type = filter_type
                experiment_config.filter.motion_model = SensorMotionModelType.get_motion_model(sensor_motion_model_type)

                # Initialize and run the pipeline
                pipeline = SingleThreadedPipeline(experiment_config)
                result = pipeline.run()

                # Collect results