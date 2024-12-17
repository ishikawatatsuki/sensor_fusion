import os
import sys
import yaml
import time
import logging
import numpy as np
from enum import Enum
from typing import Union, Tuple
from queue import PriorityQueue
from collections import namedtuple
from threading import Thread
from typing import List

sys.path.append(os.path.join(os.path.dirname(__file__), '../interfaces'))

from kitti import (
  OXTS_IMUDataReader,
  KITTI_StereoFrameReader,
  KITTI_CustomVisualOdometry,
  KITTI_GroundTruthDataReader,
  KITTI_ColorImageDataReader,
  KITTI_UpwardLeftwardVelocityDataReader
)
from dataset import (
  BaseDataset,
  Sensor,
  ExtendedSensorConfig,
  InitialState,
  SensorField,
  TimeUpdateField,
  MeasurementUpdateField,
  StereoField
)

from config import (
  DatasetConfig,
  SensorConfig,
)
from custom_types import (
  SensorType, 
  KITTI_SensorType,
  FilterType
)
from interfaces import (
  State, Pose
)

from constants import (
  KITTI_DATE_MAPS
)

logger = logging.getLogger(__name__)

class ExperimentalType(Enum):
  VO_DROP="VO_DROP"
  DRIVING_SCENARIO="DRIVING_SCENARIO"
  

ExperimentalArgs = namedtuple('ExperimentalArgs', ['experimental_type', 'frame_drop_in_s', 'drop_start_from', 'vo_data_loss_list', 'gps_data_loss_list'])

class ExperimentalSensor:
  
  def __init__(
      self, 
      dataset, 
      type: SensorType, 
      output_queue: PriorityQueue,
      drop_start_from: int,
      frame_drop_in_s: float=0.0,
    ):
    self.type = type
    self.dataset = dataset
    self.dataset_starttime = dataset.starttime
    self.starttime = None
    self.started = False
    self.stopped = False
    self.output_queue = output_queue
    
    self.field = namedtuple('sensor', ['type', 'data'])
    
    self.frame_drop_in_s = frame_drop_in_s
    self.drop_start_from = drop_start_from
    
    self.publish_thread = Thread(target=self.publish)
    
  def start(self, starttime):
    self.started = True
    self.starttime = starttime
    self.publish_thread.start()
    
  def stop(self):
    self.stopped = True
    if self.started:
      self.publish_thread.join()
    
  def publish(self):
    """This is a publisher only used in experiment to obtain a data that is dropped consecutively.
    Assuming that 0. <= dropout ratio < 0.5
    Maximum sequence of dropout is 30% of total length:
      if 0.5 is set for dropout ratio, it is divided into 0.3 and 0.2
      
    """
    dataset = list(iter(self.dataset))
    iter_length = len(dataset)
    np.random.seed(777)
    
    frames_to_drop = self.frame_drop_in_s * 10 # second + 10Hz
    
    start_dropping_at = self.drop_start_from
    end_dropping_at = start_dropping_at + frames_to_drop
    
    
    logger.info("-"*30)
    logger.info(f"Stereo camera is dropped frames between: {start_dropping_at} and {end_dropping_at}")
    logger.info("-"*30)
    
    i = 0
    while not self.stopped and i < iter_length:
      try:
        data = dataset[i]
        i += 1
      except StopIteration:
        return
      
      data_dropped = start_dropping_at < i <= end_dropping_at
      if not data_dropped:
        self.output_queue.put((data.timestamp, self.field(type=self.type, data=data)))
  
class DrivingScenarioExperimentalSensor:
  
  def __init__(
      self, 
      dataset, 
      type: SensorType, 
      output_queue: PriorityQueue,
      vo_data_loss_list: List
    ):
    self.type = type
    self.dataset = dataset
    self.dataset_starttime = dataset.starttime
    self.starttime = None
    self.started = False
    self.stopped = False
    self.output_queue = output_queue
    
    self.vo_data_loss_list = vo_data_loss_list
    
    self.field = namedtuple('sensor', ['type', 'data'])
    
    
    self.publish_thread = Thread(target=self.publish)
    
  def start(self, starttime):
    self.started = True
    self.starttime = starttime
    self.publish_thread.start()
    
  def stop(self):
    self.stopped = True
    if self.started:
      self.publish_thread.join()
    
  def publish(self):
    """This is a publisher only used in experiment to obtain a data that is dropped consecutively.
    Assuming that 0. <= dropout ratio < 0.5
    Maximum sequence of dropout is 30% of total length:
      if 0.5 is set for dropout ratio, it is divided into 0.3 and 0.2
      
    """
    dataset = list(iter(self.dataset))
    iter_length = len(dataset)
    np.random.seed(777)
    
    logger.info("-"*30)
    logger.info(f"Stereo camera is dropped frames between: ")
    logger.info(f"{[drop for drop in self.vo_data_loss_list]}")
    logger.info("-"*30)
    
    i = 0
    while not self.stopped and i < iter_length:
      try:
        data = dataset[i]
        i += 1
      except StopIteration:
        return
      
      data_dropped = True in (drop[0] <= i < drop[1] for drop in self.vo_data_loss_list)
      if not data_dropped:
        self.output_queue.put((data.timestamp, self.field(type=self.type, data=data)))
  
class ExperimentalKITTIDataset(BaseDataset): 

  sensor_threads = None
  
  def __init__(
      self, 
      experimental_args: ExperimentalArgs,
      **kwargs,
    ):
    super().__init__(**kwargs)
    
    
    date = KITTI_DATE_MAPS.get(self.config.variant)
    assert date is not None, "Please provide proper kitti drive variant."
    
    self.kitti_date = date
    self.sensor_list = self._get_sensor_list(type="kitti", sensors=self.config.sensors)
    
    self.experimental_args = experimental_args
    
    assert len(self.sensor_list) > 0,\
            "Please select sensors"
            
    self._populate_sensor_to_thread()

  def _populate_sensor_to_thread(self) -> List[Sensor]:
    
    def _get_dataset(sensor: ExtendedSensorConfig):
      kwargs = {
        'root_path': self.config.root_path,
        'date': self.kitti_date,
        'drive': self.config.variant
      }
      match (sensor.sensor_type):
        case KITTI_SensorType.OXTS_IMU:
          return OXTS_IMUDataReader(**kwargs)
        case KITTI_SensorType.OXTS_GPS:
          return KITTI_GroundTruthDataReader(**kwargs)
        case KITTI_SensorType.KITTI_STEREO:
          if self.config.variant == "0098":
            iter_offset = 500
          elif self.config.variant == "0099":
            iter_offset = 1100
          elif self.config.variant == "0100":
            iter_offset = 900
          else:
            iter_offset = 0
          kwargs['iter_offset'] = iter_offset
          return KITTI_StereoFrameReader(**kwargs)
        case KITTI_SensorType.KITTI_CUSTOM_VO:
          return KITTI_CustomVisualOdometry(**kwargs)
        case KITTI_SensorType.KITTI_COLOR_IMAGE:
          return KITTI_ColorImageDataReader(**kwargs)
        case KITTI_SensorType.KITTI_UPWARD_LEFTWARD_VELOCITY:
          return KITTI_UpwardLeftwardVelocityDataReader(**kwargs)
        case SensorType.GROUND_TRUTH:
          return KITTI_GroundTruthDataReader(**kwargs)
        case _:
          return None
    
    sensor_threads = []
    
    for sensor in self.sensor_list:
      dataset = _get_dataset(sensor)
      
      if dataset is not None:
        if sensor.sensor_type is KITTI_SensorType.KITTI_STEREO:
          if self.experimental_args.experimental_type is ExperimentalType.VO_DROP:
            s = ExperimentalSensor(
              type=sensor.sensor_type, 
              dataset=dataset, 
              output_queue=self.output_queue, 
              frame_drop_in_s=self.experimental_args.frame_drop_in_s,
              drop_start_from=self.experimental_args.drop_start_from,
            )
          else:
            s = DrivingScenarioExperimentalSensor(
              type=sensor.sensor_type, 
              dataset=dataset, 
              output_queue=self.output_queue, 
              vo_data_loss_list=self.experimental_args.vo_data_loss_list
            )
        else:
          s = Sensor(
            type=sensor.sensor_type, 
            dataset=dataset, 
            output_queue=self.output_queue, 
            dropout_ratio=sensor.dropout_ratio
          )
        
        sensor_threads.append(s)
    
        self.noise_manager._provider.register_process_noise(
          sensor_type=sensor.sensor_type, 
          dataset=dataset
        )
        
    # ADD ground truth
    self.ground_truth_dataset = _get_dataset(self.ground_truth_sensor_config)
    self.initial_state = self.ground_truth_dataset.get_initial_state()
    pose_cam_imu = Pose(
      R=self.geo_transformer.T_from_cam_to_imu[:3, :3], 
      t=self.geo_transformer.T_from_cam_to_imu[:3, 3]
    )
    initial_pose_w = pose_cam_imu * self.ground_truth_dataset.initial_pose
    # TODO: Identify the initialization problem
    # if self.config.variant == "0099":
    #   # euler = State.get_euler_angle_from_rotation_matrix(initial_pose_w.R).flatten()
    #   # q = State.get_quaternion_from_rotation_matrix(self.ground_truth_dataset.initial_pose.R)
    #   # q = State.get_quaternion_from_euler_angle(euler)
    #   euler = State.get_euler_angle_from_rotation_matrix(self.ground_truth_dataset.initial_pose.R).flatten()
    #   q = State.get_quaternion_from_euler_angle(np.array([0., 0., euler[2]]))
    # else:
    #   euler = State.get_euler_angle_from_rotation_matrix(self.ground_truth_dataset.initial_pose.R).flatten()
    #   q = State.get_quaternion_from_euler_angle(np.array([0., 0., euler[2]]))
    #   # q = State.get_quaternion_from_rotation_matrix(initial_pose_w.R)
      
    # euler = State.get_euler_angle_from_rotation_matrix(self.ground_truth_dataset.initial_pose.R).flatten()
    # q = State.get_quaternion_from_euler_angle(np.array([0., 0., euler[2]]))
    # q = State.get_quaternion_from_rotation_matrix(initial_pose_w.R)
    # self.initial_state.q = q
    # self.initial_state.p = initial_pose_w.t.reshape(-1, 1).copy()
    self.initial_state.q = State.get_quaternion_from_rotation_matrix(self.ground_truth_dataset.initial_pose.R)
    self.initial_state.p = initial_pose_w.t.reshape(-1, 1).copy()
    s = Sensor(type=SensorType.GROUND_TRUTH, dataset=self.ground_truth_dataset, output_queue=self.output_queue)
    sensor_threads.append(s)
    
    self.sensor_threads = sensor_threads
  
  def start(self):
    now = time.time()
    
    for sensor_thread in self.sensor_threads:
      sensor_thread.start(starttime=now)
      
    time.sleep(0.5)
    last_timestamp, _ = self.output_queue.get()
    self.last_timestamp = last_timestamp
      
  def stop(self):
    for sensor_thread in self.sensor_threads:
      sensor_thread.stop()
  
    
  def get_sensor_data(self, current_state: State) -> SensorField:
    
    timestamp, sensor_data = self.output_queue.get()
    last_timestamp = self.last_timestamp
    self.last_timestamp = timestamp
    
    Q = self.noise_manager.get_process_noise(sensor_type=sensor_data.type)
    R = self.noise_manager.get_measurement_noise(sensor_type=sensor_data.type)
    
    match(sensor_data.type):
      case KITTI_SensorType.OXTS_IMU:
        u = np.hstack([sensor_data.data.a, sensor_data.data.w])
        dt = (timestamp - last_timestamp)
        return SensorField(
            type=sensor_data.type, 
            timestamp=timestamp, 
            data=TimeUpdateField(u, dt, Q)
          )
      
      case KITTI_SensorType.OXTS_INS:
        u = np.hstack([sensor_data.data.vf, sensor_data.data.wx, sensor_data.data.wy, sensor_data.data.wz])
        dt = (timestamp - last_timestamp)
        return SensorField(
            type=sensor_data.type, 
            timestamp=timestamp, 
            data=TimeUpdateField(u, dt, Q)
          )

      case KITTI_SensorType.OXTS_GPS:
        # z = np.array([sensor_data.data.lon, sensor_data.data.lat, sensor_data.data.alt])
        z = sensor_data.data.t
        return SensorField(
            type=sensor_data.type, 
            timestamp=timestamp, 
            data=MeasurementUpdateField(
              z, 
              R, 
              sensor_data.type
            )
          )
        
      case KITTI_SensorType.KITTI_STEREO:
        return SensorField(
            type=sensor_data.type, 
            timestamp=timestamp, 
            data=StereoField(
              left_frame_id=sensor_data.data.left_frame_id,
              right_frame_id=sensor_data.data.right_frame_id
            )
          )
        
      case KITTI_SensorType.KITTI_CUSTOM_VO:
        # NOTE: VO data is in camera left frame
        z = np.array([
            sensor_data.data.x, 
            sensor_data.data.y,
            sensor_data.data.z,
            sensor_data.data.vx,
            sensor_data.data.vy,
            sensor_data.data.vz,
          ])
        R = R[3:, 3:] if z.shape[0] == 3 else R
        return SensorField(
            type=sensor_data.type, 
            timestamp=timestamp, 
            data=MeasurementUpdateField(
              z, 
              R, 
              sensor_data.type
            )
          )
        
      case KITTI_SensorType.KITTI_COLOR_IMAGE:
        return SensorField(
            type=sensor_data.type, 
            timestamp=timestamp, 
            data=sensor_data.data.image_path
          )
      case KITTI_SensorType.KITTI_UPWARD_LEFTWARD_VELOCITY:
        z = np.array([sensor_data.data.vu, sensor_data.data.vl])
        return SensorField(
          type=sensor_data.type,
          timestamp=timestamp,
          data=MeasurementUpdateField(
            z,
            R,
            sensor_data.type
          )
        )
      case SensorType.GROUND_TRUTH:
        z = np.hstack([sensor_data.data.R, sensor_data.data.t.reshape(-1, 1)])
        z = np.vstack([z, np.array([0., 0., 0., 1.])])
        return SensorField(
            type=sensor_data.type, 
            timestamp=timestamp, 
            data=z
          )
        
      case _:
        return SensorField(
            type=sensor_data.type, 
            timestamp=timestamp, 
            data=MeasurementUpdateField(
              np.zeros(3), 
              np.eye(3), 
              sensor_data.type
            )
          )
  
if __name__ == "__main__":
  
  import time
  
  def _load_dataset():

    dataset_config = DatasetConfig(
      type=dataset,
      mode="stream",
      root_path="../../data/KITTI",
      variant="0033",
      sensors=[
        SensorConfig(name='oxts_imu', dropout_ratio=0), 
        SensorConfig(name='oxts_gps', dropout_ratio=0),
        SensorConfig(name='kitti_stereo', dropout_ratio=0)
      ]
    )
    kwargs = {
      'config': dataset_config,
    }
    dataset = ExperimentalKITTIDataset(**kwargs)
    
    state = State(
      p=np.zeros((3, 1)),
      v=np.zeros((3, 1)),
      q=np.array([1., 0., 0., 0.])
    )
      
    # Start the threads
    dataset.start()
    time.sleep(0.5)
    
    while True:
      if dataset.is_queue_empty():
        break
      
      sensor_data = dataset.get_sensor_data(current_state=state)
      if SensorType.is_time_update(sensor_data.type):
        print(sensor_data.timestamp, sensor_data.type)
        ...
      elif SensorType.is_measurement_update(sensor_data.type):
        ...
      elif SensorType.is_stereo_image_data(sensor_data.type):
        # print(sensor_data.timestamp, sensor_data.type)
        ...
  
  _load_dataset()