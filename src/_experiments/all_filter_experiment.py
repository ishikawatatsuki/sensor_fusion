import os
import sys
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from enum import Enum
from typing import List
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from config import (
  Config,
  DatasetConfig,
  SensorConfig
)
from constants import KITTI_SEQUENCE_MAPS

from interfaces import Pose, State

from single_threaded_pipeline import SingleThreadedPipeline


logger = logging.getLogger(__name__)

if __name__ == "__main__":
  logging.basicConfig(format='[%(asctime)s] [%(levelname)5s] > %(message)s (%(filename)s:%(lineno)s)', 
                      datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
  logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
  
  # stdout_handler = logging.StreamHandler(sys.stdout)
  # stdout_handler.setLevel(logging.DEBUG)

  # file_handler = logging.FileHandler('logs.log')
  # file_handler.setLevel(logging.DEBUG)

  # logger.addHandler(file_handler)
  # logger.addHandler(stdout_handler)


class SensorMotionModelType(Enum):
  KINEMATICS_W_GPS = "KINEMATICS_W_GPS"
  KINEMATICS_WO_GPS = "KINEMATICS_WO_GPS"
  VELOCITY_W_GPS = "VELOCITY_W_GPS"
  VELOCITY_WO_GPS = "VELOCITY_WO_GPS"
  
  @staticmethod
  def get_motion_model(type_str: str) -> str:
    try:
      t = SensorMotionModelType(type_str.upper())
      match (t):
        case SensorMotionModelType.KINEMATICS_W_GPS | SensorMotionModelType.KINEMATICS_WO_GPS:
          return "kinematics"
        case SensorMotionModelType.VELOCITY_W_GPS | SensorMotionModelType.VELOCITY_WO_GPS:
          return "velocity"
        case _:
          return "kinematics"
    except:
      return "kinematics"
    
  @staticmethod
  def get_error_folder_name(type_str: str) -> str:
    try:
      t = SensorMotionModelType(type_str.upper())
      match (t):
        case SensorMotionModelType.KINEMATICS_W_GPS:
          return "kinematics_w_gps"
        case SensorMotionModelType.KINEMATICS_WO_GPS:
          return "kinematics_wo_gps"
        case SensorMotionModelType.VELOCITY_W_GPS:
          return "velocity_w_gps"
        case SensorMotionModelType.VELOCITY_WO_GPS:
          return "velocity_wo_gps"
        case _:
          return "warning"
    except:
      return "warning"
    

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

def run_filters_experiment():
  
  def _get_config(variant: str, filter: str, experimental_setting: SensorMotionModelType):
    config_filepath = "../yaml_files/config_kitti_for_thesis.yaml"
    motion_model = SensorMotionModelType.get_motion_model(experimental_setting.value)
    error_folder_name = SensorMotionModelType.get_error_folder_name(experimental_setting.value)
    
    config = Config(config_filepath=config_filepath)
    _new_dataset_config = config.dataset._asdict()
    _new_dataset_config['variant'] = variant
    _new_dataset_config['sensors'] = SensorMotionModelType.get_sensor_config(
      type_str=experimental_setting.value,
      new_dataset_config=_new_dataset_config
    )
    
    _new_filter_config = config.filter._asdict()
    _new_filter_config['type'] = filter
    _new_filter_config['motion_model'] = motion_model
    
    _new_report_config = config.report._asdict()
    _new_report_config['kitti_pose_result_folder'] = f"filter_comparison_final/{filter}/{error_folder_name}/"
    
    _new_visualization_config = config.visualization._asdict()
    _new_visualization_config['output_filepath'] = os.path.join(
      _new_visualization_config['output_filepath'], 
      filter,
      error_folder_name
    )
    
    config.dataset = config.dataset._replace(**_new_dataset_config)
    config.filter = config.filter._replace(**_new_filter_config)
    config.visualization = config.visualization._replace(**_new_visualization_config)
    config.report = config.report._replace(**_new_report_config)
    return config
  
  filters = ["ekf", "ukf", "pf", "enkf", "ckf"]
  variants = ["0067", "0027", "0033"]
  sequences = [KITTI_SEQUENCE_MAPS.get(variant) for variant in variants]
  
  experimental_settings = [
    SensorMotionModelType.KINEMATICS_W_GPS, 
    SensorMotionModelType.KINEMATICS_WO_GPS,
    SensorMotionModelType.VELOCITY_W_GPS,
    SensorMotionModelType.VELOCITY_WO_GPS
  ]
  
  df_root_path = "../../outputs/KITTI/dataframes/experimental_settings/inference_time_final/"
  
  for experimental_setting in tqdm(experimental_settings):
    
    inference_time_matrix = np.empty((len(variants), len(filters)))
    for i, variant in enumerate(variants):
      for j, kf in enumerate(filters):
        config = _get_config(
          variant=variant,
          filter=kf,
          experimental_setting=experimental_setting
        )
        logger.info(f"[Dataset] {experimental_setting.name} {config.dataset.variant}, {[sensor.name for sensor in config.dataset.sensors]}")
        logger.info(f"[Report] {config.report.kitti_pose_result_folder}")
        logger.info(f"[Filter] {config.filter.type}, {config.filter.motion_model}")
        logger.info(f"[Visualization] {config.visualization.output_filepath}")
        logger.info(f"--"*100)
        
        pipeline = SingleThreadedPipeline(
          config=config
        )

        error, total_time = pipeline.demonstrate_with_time()
        inference_time = total_time / len(pipeline.error_reporter.estimated_trajectory)
        
        inference_time_matrix[i, j] = inference_time
        
        logger.info(error)
        logger.info(inference_time)
        
    print(inference_time_matrix)
    average = np.average(inference_time_matrix, axis=0)
    inference_matrix = np.vstack([inference_time_matrix, average])
    print(inference_matrix)
    df = pd.DataFrame(inference_time_matrix, index=sequences, columns=["EKF", "UKF", "PF", "EnKF", "CKF"])
    df_filename = experimental_setting.value.lower() + "_03_07_09.csv"
    df.to_csv(os.path.join(df_root_path, df_filename))



if __name__ == "__main__":
  
  logger.info("Start experiment.")
  run_filters_experiment()
  logger.info("End experiment.")