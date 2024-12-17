import os
import sys
import logging
import numpy as np
from typing import List

sys.path.append(os.path.join(os.path.dirname(__file__), 'kalman_filters'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'interfaces'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'dataset'))

from custom_types import (
  SensorType,
  FilterType,
  NoiseType,
  DatasetType
)
from config import (
  Config
)
from visual_odometry import (
  VO_Status,
  VisualOdometryV1
)
from dataset import (
  KITTIDataset,
  UAVDataset,
  ExperimentalKITTIDataset
)
from utils import (
  time_reporter,
  BaseVisualizationField, PoseVisualizationField, VisualizationDataType,
  ErrorReportType
)
from interfaces import Pose, State

from main import Pipeline


logger = logging.getLogger(__name__)

if __name__ == "__main__":
  logging.basicConfig(format='[%(asctime)s] [%(levelname)5s] > %(message)s (%(filename)s:%(lineno)s)', 
                      datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
  logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)


class SingleThreadedPipeline(Pipeline):
  
  def __init__(
    self,
    *args,
    **kwargs,
  ):
    super().__init__(*args, **kwargs)
    
    self.visual_odometry = VisualOdometryV1(
      query_current_estimate_cb=self.query_current_estimate_cb,
      initial_pose=self.dataset.ground_truth_dataset.initial_pose,
      vo_config=self.config.vo_config,
      dataset_config=self.config.dataset
    )
    
    self.gt_pose = None
    self.gt_poses = []

  def query_current_estimate_cb(self):
    
    R_cam_to_imu = Pose(
      R=self.geo_transformer.T_from_cam_to_imu[:3, :3],
      t=self.geo_transformer.T_from_cam_to_imu[:3, 3]
    )
    
    gt_pose_w = R_cam_to_imu * self.gt_pose
    gt_pose_initial_world_frame = R_cam_to_imu * self.dataset.ground_truth_dataset.initial_pose
    gt_angle_offset = State.get_euler_angle_from_rotation_matrix(gt_pose_initial_world_frame.R).flatten()
    
    estimated_angle = self.filter.x.get_euler_angle_from_quaternion()
    estimated_angle += gt_angle_offset
    
    estimated_q = State.get_quaternion_from_euler_angle(estimated_angle)
    
    estimated_w = Pose(
      R = State.get_rotation_matrix_from_quaternion_vector(estimated_q.flatten()),
      t=self.filter.x.p,
    )
    
    
    gt_pose_cam = R_cam_to_imu.inverse() * gt_pose_w
    estimated_pose_cam = R_cam_to_imu.inverse() * estimated_w

    logger.info("---"*50)
    logger.info(self.filter.x.p.flatten())
    logger.info(gt_pose_w.t.flatten())
    logger.info("---"*20)
    logger.info(State.get_euler_angle_from_rotation_matrix(gt_pose_cam.R).flatten())
    logger.info(State.get_euler_angle_from_rotation_matrix(estimated_pose_cam.R).flatten())
    logger.info("---"*50)
    return estimated_pose_cam
    
    
  def demonstrate(
      self
    ):
      
      # NOTE: Start loading data
      self.dataset.start()
      self.visualizer.start(initial_poses=self._get_initial_poses_for_visualization())
      
      error = None
      vo_estimates = None
      try:
        if self.config.mode.log_sensor_data:
          f = open(self.config.mode.sensor_data_output_filepath, "w")
        i = 0
        while True:
          if self.dataset.is_queue_empty():
            break
          
          sensor_data = self.dataset.get_sensor_data(current_state=self.filter.x)
          sensor_name = SensorType.get_sensor_name(self.config.dataset.type, sensor_data.type)
          
          logger.debug(f"[{self.dataset.output_queue.qsize():05}] time: {sensor_data.timestamp}, sensor: {sensor_name}")
          
          if SensorType.is_visualization_data(sensor_data.type):
            # NOTE: load image and visualize it frame by frame
            ...
          
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
            
            self.filter.measurement_update(**sensor)
            
            if self.config.visualization.show_vo_trajectory and SensorType.is_vo_data(sensor_data.type):
              vo_estimates = sensor['z'].flatten()
          
          elif SensorType.is_stereo_image_data(sensor_data.type):
            # NOTE: enqueue stereo data and process vo estimation
            estimation = self.visual_odometry.estimate(timestamp=sensor_data.timestamp, data=sensor_data.data)
            if not estimation.success:
              vo_estimates = None
              logger.info("-"*30)
              logger.warning("VO failed estimating trajectory")
              logger.info("-"*30)
              continue
            
            
            velocity_only = self.visual_odometry.vo_status is VO_Status.LOST_TRACKING and\
                              self.config.filter.vo_velocity_only_update_when_failure
            
            value = estimation.velocity if velocity_only else np.hstack([estimation.pose.t, estimation.velocity])
            
            # NOTE: Get measurement noise covariance matrix R
            if self.config.filter.noise_type is NoiseType.DYNAMIC:
              # TODO: Implement dynamic noise tuning
              R = self.dataset.noise_manager.get_vo_measurement_noise(estimation=estimation)
            else:
              R = self.dataset.noise_manager.get_measurement_noise(
                sensor_type=SensorType.KITTI_CUSTOM_VO
              )
            
            # NOTE: transform camera frame to inertial frame
            z = self.geo_transformer.transform_data(
              sensor_type=SensorType.KITTI_CUSTOM_VO,
              value=value,
              state=self.filter.x
            )
            
            # NOTE: Trim if measurement data only contains velocity vector
            R = R[3:, 3:] if z.shape[0] == 3 else R
            
            if not velocity_only:
              vo_estimates = z[:3, 0]
              logger.info("Position+Velocity update")
            else:
              logger.info("Velocity only update")
            
            sensor = {
              'z': z,
              'R': R,
              'sensor_type': SensorType.KITTI_CUSTOM_VO
            }
            self.filter.measurement_update(**sensor)
            
          if SensorType.is_reference_data(sensor_data.type):
            # NOTE: append reference trajectory and estimated trajectory in the corresponding list to compare
            ground_truth = self.geo_transformer.transform_data(
                  sensor_type=SensorType.GROUND_TRUTH,
                  value=sensor_data.data,
                  state=self.filter.x
                )
            estimated = self.filter.x.p.flatten()
            self.error_reporter.set_trajectory(
              estimated=estimated,
              expected=ground_truth
            )
            
            if self.dataset.type is DatasetType.KITTI or\
                self.dataset.type is DatasetType.EXPERIMENT:
              R_cam_to_imu = Pose(
                R=self.geo_transformer.T_from_cam_to_imu[:3, :3],
                t=self.geo_transformer.T_from_cam_to_imu[:3, 3]
              )
              R_imu_to_cam = R_cam_to_imu.inverse()
              self.gt_pose = Pose(R=sensor_data.data[:3, :3], t=sensor_data.data[:3, 3])
              
              gt_pose_world_frame = R_cam_to_imu * self.gt_pose
              gt_pose_initial_world_frame = R_cam_to_imu * self.dataset.ground_truth_dataset.initial_pose
              gt_angle = State.get_euler_angle_from_rotation_matrix(gt_pose_world_frame.R).flatten()
              gt_angle_offset = State.get_euler_angle_from_rotation_matrix(gt_pose_initial_world_frame.R).flatten()
              
              estimated_angle = self.filter.x.get_euler_angle_from_quaternion()
              estimated_angle += gt_angle_offset
              
              # NOTE: Make sure that angles are in range [-pi, pi)
              estimated_angle = np.array([(lambda x: (x + np.pi) % (2 * np.pi) - np.pi)(angle) for angle in estimated_angle])
              t = estimated.reshape(-1, 1)
              q = State.get_quaternion_from_euler_angle(estimated_angle).flatten()
              R = State.get_rotation_matrix_from_quaternion_vector(q)
              estimated_pose = R_imu_to_cam * Pose(R=R, t=t)
              
              self.error_reporter.set_trajectory(
                estimated=estimated_pose.matrix(pose_only=True),
                expected=self.gt_pose.matrix(pose_only=True),
                type=ErrorReportType.POSE_IN_CAM
              )
              
              self.gt_poses.append(self.gt_pose.matrix(pose_only=True))
              
              
            if self.config.filter.type == "pf":
              self.visualizer.set_particles(
                particles=self.filter.particles[:30, :],
                weights=self.filter.weights[:30],
                filter_type=FilterType.PF
              )
            elif self.filter.config.type == "enkf":
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
            if vo_estimates is not None:
              vo_pose = Pose(R=np.eye(3), t=vo_estimates[:3].reshape(-1, 1))
              plot_data.append(PoseVisualizationField(
                type=VisualizationDataType.VO,
                pose=vo_pose,
                lw=1,
                color='blue'
              ))
            
            self.visualizer.show_realtime_estimation(data=plot_data)

            R_cam_to_imu = Pose(
              R=self.geo_transformer.T_from_cam_to_imu[:3, :3],
              t=self.geo_transformer.T_from_cam_to_imu[:3, 3]
            )
            gt_pose_world_frame = R_cam_to_imu * self.gt_pose
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
            
            self.visualizer.set_angle_estimation(data=angle_data)

          if self.config.mode.log_sensor_data:
            f.write(f"{self.dataset.output_queue.qsize():05}: {sensor_name}\n")
      
        self.visualizer.show_angle_estimation()
        
        final_image_filename = f"{self.config.dataset.type}_{self.config.dataset.variant}_result.png"
        self.visualizer.show_final_estimation(filename=final_image_filename)
        self.visualizer.show_innovation(self.filter.innovations)
        error = self.error_reporter.compute_error()
        
        sensor_names = "_".join([sensor.name for sensor in self.config.dataset.sensors])
        filename = f"{self.config.filter.type}_{self.config.filter.motion_model}_{sensor_names}_{self.config.dataset.variant}"
        self.error_reporter.export_all_error_report(filename=filename)
        
        
      except KeyboardInterrupt:
        logger.info("Interrupted. Closing the plot window(s).")
      # except Exception as e:
      #   logger.error(e)
      finally:
        logger.info("Process finished!")
        self.dataset.stop()
        self.visualizer.stop()
      
      return error
  
  @time_reporter
  def demonstrate_with_time(self):
    return self.demonstrate()
  
if __name__ == "__main__":
  
  import matplotlib.pyplot as plt
  
  config_filepath = "./yaml_files/config_kitti_for_thesis_test.yaml"
  
  config = Config(
    config_filepath=config_filepath
  )
  demo_pipeline = SingleThreadedPipeline(
    config=config
  )
  
  error, time = demo_pipeline.demonstrate_with_time()
  
  inference_time = time / len(demo_pipeline.error_reporter.estimated_trajectory)
  
  logger.info(time)
  logger.info(error)
  
  R_cam_to_imu = Pose(
    R=demo_pipeline.geo_transformer.T_from_cam_to_imu[:3, :3],
    t=demo_pipeline.geo_transformer.T_from_cam_to_imu[:3, 3]
  )

  est = np.array(demo_pipeline.error_reporter.estimated_poses_in_cam)
  est = est.reshape(len(est), 3, 4)
  gt = np.array(demo_pipeline.gt_poses)
  
  # est = np.array([(R_cam_to_imu * Pose(R=_est[:3, :3], t=_est[:3, 3])).t for _est in est])
  # gt = np.array([(R_cam_to_imu * Pose(R=_gt[:3, :3], t=_gt[:3, 3])).t for _gt in gt])

