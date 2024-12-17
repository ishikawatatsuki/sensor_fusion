import os
import sys
import logging
import numpy as np
from typing import Union, List
from multiprocessing import Process, Queue
from queue import PriorityQueue

sys.path.append(os.path.join(os.path.dirname(__file__), 'kalman_filters'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'interfaces'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'dataset'))

from kalman_filters import (
  ExtendedKalmanFilter,
  UnscentedKalmanFilter,
  ParticleFilter,
  EnsembleKalmanFilter,
  CubatureKalmanFilter
)
from custom_types import (
  SensorType,
  DatasetType,
  FilterType
)
from config import (
  Config
)
from dataset import (
  KITTIDataset,
  UAVDataset,
  SensorField
)
from utils import (
  time_reporter,
  ErrorReporter,
  Visualizer, PoseVisualizationField, BaseVisualizationField, VisualizationDataType,
  KITTI_GeometricTransformer, UAV_GeometricTransformer, Viztrack_GeometricTransformer,
)
from interfaces import State, Pose

# NOTE: Referring the github repo: https://github.com/uoip/stereo_msckf

logger = logging.getLogger(__name__)

if __name__ == "__main__":
  logging.basicConfig(format='[%(asctime)s] [%(levelname)5s] > %(message)s (%(filename)s:%(lineno)s)', 
                      datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
  logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)


class Pipeline:
  
  def __init__(
      self,
      config: Config,
    ):
    
    self.config = config
    
    self.error_reporter = ErrorReporter(
      report_config=self.config.report,
      filter_config=self.config.filter,
      dataset_config=self.config.dataset,
    )
    
    self.visualizer = Visualizer(
      config=self.config.visualization,
      filter_config=self.config.filter
    )
    
    self.geo_transformer = self._get_geometric_transformer()

    self.vo_stereo_queue = Queue()
    self.vo_outout_queue = Queue()
    self.filter_buffer_queue = PriorityQueue()
    
    self.dataset = self._get_dataset()
    
    self.filter = self._get_filter_algorithm()
    
    
    # NOTE: Realtime Visual Odometry process is not included in this repo. Only using prepared static data.
    # self.visual_odometry = VisualOdometryV1(
    #   config=self.config.vo_config
    # )
    self.vo_pose = None

  def _get_geometric_transformer(self):
    dataset_type = str(self.config.dataset.type).lower()
    
    kwargs = {
      'dataset_config': self.config.dataset,
    }
    match (dataset_type):
      case "kitti":
        logger.info(f"Setting KITTI geometric transformer.")
        return KITTI_GeometricTransformer(**kwargs)
      case "uav":
        logger.info(f"Setting UAV geometric transformer.")
        return UAV_GeometricTransformer(**kwargs)
      case "viztrack":
        logger.info(f"Setting Viztrack geometric transformer.")
        return Viztrack_GeometricTransformer(**kwargs)
      case _:
        return KITTI_GeometricTransformer(**kwargs)
  
  def _get_filter_algorithm(self):
    filter_type = str(self.config.filter.type).lower()
    initial_state = self.dataset.get_initial_state()
    coordinate_system = DatasetType.get_coordinate_system(self.config.dataset.type)
    
    kwargs = {
      'config': self.config.filter,
      'dataset_config': self.config.dataset,
      'x': initial_state.x,
      'P': initial_state.P,
      'coordinate_system': coordinate_system,
    }
    match (filter_type):
      case "ekf":
        logger.info(f"Configuring Extended Kalman Filter.")
        return ExtendedKalmanFilter(**kwargs)
      case "ukf":
        logger.info(f"Configuring Unscented Kalman Filter.")
        return UnscentedKalmanFilter(**kwargs)
      case "pf":
        logger.info(f"Configuring Particle Filter.")
        return ParticleFilter(**kwargs)
      case "enkf":
        logger.info(f"Configuring Ensemble Kalman Filter.")
        return EnsembleKalmanFilter(**kwargs)
      case "ckf":  
        logger.info(f"Configuring Cubature Kalman Filter.")
        return CubatureKalmanFilter(**kwargs)
      case _:
        # NOTE: Set EKF as a default filter
        logger.warning(f"dataset: {filter_type} is not found. EKF is used instead.")
        return ExtendedKalmanFilter(**kwargs)
      
  def _get_dataset(self) -> Union[UAVDataset, KITTIDataset]:
    dataset_type = str(self.config.dataset.type).lower()
    
    kwargs = {
      'config': self.config.dataset,
      'filter_config': self.config.filter,
      'geo_transformer': self.geo_transformer,
    }
    
    match (dataset_type):
      case "kitti":
        return KITTIDataset(**kwargs)
      case "uav":
        kwargs["uav_sensor_path"] = "./dataset/uav_sensor_path.yaml"
        kwargs["imu_config_path"] = "./dataset/uav_imu_config.yaml"
        return UAVDataset(**kwargs)
      case _:
        # NOTE: Set KITTI dataset as a default dataset
        logger.warning(f"dataset: {dataset_type} is not found. kitti dataset is used instead.")
        return KITTIDataset(**kwargs)
  
  def _get_initial_poses_for_visualization(self) -> List[PoseVisualizationField]:
    gt_pose = Pose.from_state(self.dataset.initial_state)
    estimated_pose = self.filter.get_current_estimate()
    
    initial_poses = [
      PoseVisualizationField(
          type=VisualizationDataType.GROUND_TRUTH, 
          pose=gt_pose,
          lw=2,
          color="black"
        ),
        PoseVisualizationField(
          type=VisualizationDataType.ESTIMATION, 
          pose=estimated_pose,
          lw=2,
          color="red"
        )
    ]
    if self.config.visualization.show_vo_trajectory:
      initial_poses.append(
        PoseVisualizationField(
          type=VisualizationDataType.VO, 
          pose=gt_pose,
          lw=1,
          color="blue"
        )
      )
    return initial_poses
  
  def _get_trajectory_visualization_data(self, sensor_data: SensorField) -> List[PoseVisualizationField]:
    ground_truth = self.geo_transformer.transform_data(
          sensor_type=SensorType.GROUND_TRUTH,
          value=sensor_data.data,
          state=self.filter.x
      )
    self.error_reporter.set_trajectory(
      estimated=self.filter.x.p.flatten(),
      expected=ground_truth
    )
    
    estimated = self.filter.x.p.flatten()
    
    if self.config.filter.type == "pf":
      self.visualizer.set_particles(
        particles=self.filter.particles[:30, :],
        weights=self.filter.weights[:30],
        filter_type=FilterType.PF
      )
    elif self.config.filter.type == "enkf":
      self.visualizer.set_particles(
        particles=self.filter.samples[:30, :],
        filter_type=FilterType.EnKF
      )
    plot_data = [
      PoseVisualizationField(
        type=VisualizationDataType.GROUND_TRUTH,
        pose=Pose(R=np.eye(3), t=ground_truth.reshape(-1, 1)),
        lw=2,
        color='black'
      ),
      PoseVisualizationField(
        type=VisualizationDataType.ESTIMATION,
        pose=Pose(R=np.eye(3), t=estimated.reshape(-1, 1)),
        lw=2,
        color='red'
      ),
    ]
    if self.vo_pose is not None:
      plot_data.append(PoseVisualizationField(
        type=VisualizationDataType.VO,
        pose=self.vo_pose,
        lw=1,
        color='blue'
      ))
    
    return plot_data
    
  def _get_angle_estimation_data(self, sensor_data: SensorField) -> List[BaseVisualizationField]:
    if self.dataset.type is DatasetType.KITTI:
      # NOTE: Consider offset specifically for KITTI dataset
      gt_pose = Pose(R=sensor_data.data[:3, :3], t=sensor_data.data[:3, 3])
      R_cam_to_imu = Pose(
        R=self.geo_transformer.T_from_cam_to_imu[:3, :3],
        t=self.geo_transformer.T_from_cam_to_imu[:3, 3]
      )
      gt_pose_world_frame = R_cam_to_imu * gt_pose
      gt_pose_initial_world_frame = R_cam_to_imu * self.dataset.ground_truth_dataset.initial_pose
      gt_angle = State.get_euler_angle_from_rotation_matrix(gt_pose_world_frame.R).flatten()
      gt_angle_offset = State.get_euler_angle_from_rotation_matrix(gt_pose_initial_world_frame.R).flatten()
      
      estimated_angle = self.filter.x.get_euler_angle_from_quaternion()
      estimated_angle += gt_angle_offset
      
      estimated_angle = [(lambda x: (x + np.pi) % (2 * np.pi) - np.pi)(angle) for angle in estimated_angle]
      
      angle_data = [
        BaseVisualizationField(
          x=gt_angle[0],
          y=gt_angle[1],
          z=gt_angle[2],
          lw=1,
          color='black'
        ),
        BaseVisualizationField(
          x=estimated_angle[0],
          y=estimated_angle[1],
          z=estimated_angle[2],
          lw=1,
          color='red'
        )
      ]
      return angle_data
    else:
      return []
    
  @time_reporter
  def run(
    self
  ):
    
    # NOTE: Start loading data
    self.dataset.start()
    self.visualizer.start(initial_poses=self._get_initial_poses_for_visualization())
    
    error = None
    try:
      
      if self.config.mode.log_sensor_data:
        f = open(self.config.mode.sensor_data_output_filepath, "w")
        
      while True:
        if self.dataset.is_queue_empty():
          break
        
        sensor_data = self.dataset.get_sensor_data(current_state=self.filter.x)
        sensor_name = SensorType.get_sensor_name(self.config.dataset.type, sensor_data.type)
        
        logger.debug(f"[{self.dataset.output_queue.qsize():05}] time: {sensor_data.timestamp}, sensor: {sensor_name}")
        
        if SensorType.is_stereo_image_data(sensor_data.type):
          # NOTE: enqueue stereo data and process vo estimation
          self.vo_stereo_queue.put(sensor_data.data)
        
        elif SensorType.is_time_update(sensor_data.type):
          # NOTE: process time update step
          self.filter.time_update(*sensor_data.data)
          
        elif SensorType.is_measurement_update(sensor_data.type):
          # NOTE: process measurement update step.
          sensor = {
            'z': self.geo_transformer.transform_data(
                sensor_type=sensor_data.type,
                value=sensor_data.data.z,
                state=self.filter.x
            ),
            'R': sensor_data.data.R,
            'sensor_type': sensor_data.type
          }
          if SensorType.is_vo_data(sensor_data.type) and self.config.visualization.show_vo_trajectory:
            self.vo_pose = Pose(R=np.eye(3), t=sensor['z'][:3].reshape(-1, 1))
          
          self.filter.measurement_update(**sensor)
        
        if SensorType.is_reference_data(sensor_data.type):
          # NOTE: append reference trajectory and estimated trajectory in the corresponding list to compare
          
          plot_data = self._get_trajectory_visualization_data(sensor_data)
          self.visualizer.show_realtime_estimation(data=plot_data)
          
          if self.config.visualization.show_angle_estimation:
            angle_data = self._get_angle_estimation_data(sensor_data)
            self.visualizer.set_angle_estimation(data=angle_data)

        if self.config.mode.log_sensor_data:
          f.write(f"{self.dataset.output_queue.qsize():05}: {sensor_name}\n")
    
      error = self.error_reporter.compute_error()
      
      self.visualizer.show_angle_estimation()
      self.visualizer.show_final_estimation()
      self.visualizer.show_innovation(self.filter.innovations)
      
      self.error_reporter.export_all_error_report(filename="estimation_result")
      
    except KeyboardInterrupt:
      logger.info("Interrupted. Closing the plot window(s).")
    # except Exception as e:
    #   logger.error(e)
    finally:
      logger.info("Process finished!")
      self.dataset.stop()
      self.visualizer.stop()
    
    return error

if __name__ == "__main__":
  
  config_filepath = "./yaml_files/config_kitti.yaml"
  
  config = Config(
    config_filepath=config_filepath
  )
  pipeline = Pipeline(
    config=config
  )
  
  error, time = pipeline.run()
  logger.info(error)
  logger.info(time)