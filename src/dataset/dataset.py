import os
import sys
import yaml
import time
import logging
import numpy as np
from typing import Union, Tuple
from queue import PriorityQueue
from collections import namedtuple
from threading import Thread
from typing import List

sys.path.append(os.path.join(os.path.dirname(__file__), '../interfaces'))

from kitti import (
  OXTS_GPSDataReader,
  OXTS_IMUDataReader,
  OXTS_INSDataReader,
  KITTI_StereoFrameReader,
  KITTI_CustomVisualOdometry,
  KITTI_GroundTruthDataReader,
  KITTI_ColorImageDataReader,
  KITTI_UpwardLeftwardVelocityDataReader
)
from uav import (
  VOXL_IMUDataReader,
  VOXL_StereoFrameReader,
  VOXL_QVIOOverlayDataReader,
  PX4_GPSDataReader,
  PX4_IMUDataReader,
  PX4_MagnetometerDataReader,
  PX4_VisualOdometryDataReader,
  
  UAVCustomVisualOdometryDataReader
)

from config import (
  FilterConfig,
  DatasetConfig,
  SensorConfig,
  GyroSpecification,
  AccelSpecification
)
from custom_types import (
  SensorType, 
  UAV_SensorType, 
  KITTI_SensorType,
  MotionModel,
  FilterType,
  DatasetType
)
from interfaces import (
  State, Pose
)
from utils import NoiseManager, KITTI_GeometricTransformer, UAV_GeometricTransformer

from constants import (
  KITTI_DATE_MAPS,
  MAX_CONSECUTIVE_DROPOUT_RATIO
)

logger = logging.getLogger(__name__)

ExtendedSensorConfig = namedtuple('LookupElement', SensorConfig._fields + ('sensor_type', ))

InitialState = namedtuple('InitialState', ['x', 'P'])
SensorField = namedtuple('SensorField', ['type', 'timestamp', 'data'])

TimeUpdateField = namedtuple('TimeUpdateField', ('u', 'dt', 'Q'))
MeasurementUpdateField = namedtuple('MeasurementUpdateField', ('z', 'R', 'sensor_type'))
StereoField = namedtuple('StereoField', ['left_frame_id', 'right_frame_id'])

class Sensor:
  
  def __init__(
      self, 
      dataset, 
      type: SensorType, 
      output_queue: PriorityQueue,
      dropout_ratio: float=0.0,
    ):
    self.type = type
    self.dataset = dataset
    self.dataset_starttime = dataset.starttime
    self.starttime = None
    self.started = False
    self.stopped = False
    self.output_queue = output_queue
    
    self.field = namedtuple('sensor', ['type', 'data'])
    
    assert 0.0 <= dropout_ratio and dropout_ratio <= 1.0, "Please set dropout ratio 0. <= ratio <= 1."
    self.dropout_ratio = dropout_ratio
    
    publisher = self.publish_consecutive_drp if self.type is SensorType.KITTI_STEREO and\
                                              self.dropout_ratio > 0.0 else self.publish
                                              
    self.publish_thread = Thread(target=publisher)
    
  def start(self, starttime):
    self.started = True
    self.starttime = starttime
    self.publish_thread.start()
    
  def stop(self):
    self.stopped = True
    if self.started:
      self.publish_thread.join()
    
  def publish(self):
    dataset = iter(self.dataset)
    while not self.stopped:
      try:
        data = next(dataset)
      except StopIteration:
        return

      data_dropped = np.random.uniform() < self.dropout_ratio
      if not data_dropped:
        self.output_queue.put((data.timestamp, self.field(type=self.type, data=data)))
  
  def publish_consecutive_drp(self):
    """This is a publisher only used in experiment to obtain a data that is dropped consecutively.
    Assuming that 0. <= dropout ratio < 0.5
    Maximum sequence of dropout is 30% of total length:
      if 0.5 is set for dropout ratio, it is divided into 0.3 and 0.2
      
    """
    dataset = list(iter(self.dataset))
    iter_length = len(dataset)
    np.random.seed(777)
    
    dp_list = [MAX_CONSECUTIVE_DROPOUT_RATIO for i in range(int(self.dropout_ratio // MAX_CONSECUTIVE_DROPOUT_RATIO))]
  
    if self.dropout_ratio % MAX_CONSECUTIVE_DROPOUT_RATIO != 0.0 and 0.001 < self.dropout_ratio % MAX_CONSECUTIVE_DROPOUT_RATIO < MAX_CONSECUTIVE_DROPOUT_RATIO:
      dp_list.append(round(self.dropout_ratio % MAX_CONSECUTIVE_DROPOUT_RATIO, 3))
      
    logger.info("-"*30)
    logger.info(f"Stereo camera dropout ratio list: {dp_list}")
    logger.info("-"*30)
    
    dp_list.reverse()
    dropout_length = int(iter_length * dp_list.pop())
    start_dropping_at = np.random.randint(int(iter_length * 0.2))
    end_dropping_at = start_dropping_at + dropout_length
    
    i = 0
    while not self.stopped and i < iter_length:
      try:
        data = dataset[i]
        i += 1
      except StopIteration:
        return
      # FIXME: recover the comment out
      # data_dropped = start_dropping_at < i <= end_dropping_at
      data_dropped = False
      if 100 < i <= 200:
        data_dropped = True
      elif 500 < i <= 600:
        data_dropped = True
        
      if not data_dropped:
        self.output_queue.put((data.timestamp, self.field(type=self.type, data=data)))
  
      # if end_dropping_at < i and len(dp_list):
      #   dropout_length = int(iter_length * dp_list.pop())
      #   rest_iter = iter_length - end_dropping_at
      #   start_dropping_at = end_dropping_at + np.random.randint(int(rest_iter * 0.2))
      #   end_dropping_at = start_dropping_at + dropout_length

class BaseDataset:
  
  def __init__(
    self,
    config: DatasetConfig,
    filter_config: FilterConfig,
    geo_transformer: KITTI_GeometricTransformer|UAV_GeometricTransformer
  ):
    self.config = config
    
    self.type = DatasetType.get_type_from_str(self.config.type)
    
    self.output_queue = PriorityQueue()
    self.last_timestamp = None
    
    self.noise_manager = NoiseManager(
      filter_config=filter_config
    )
    
    self.geo_transformer = geo_transformer
    self.ground_truth_sensor_config = ExtendedSensorConfig(
      sensor_type=SensorType.GROUND_TRUTH, 
      name="ground_truth", 
      dropout_ratio=0., 
      window_size=1
    )
    self.ground_truth_dataset = None
    self.initial_state = None
    
  
  def get_initial_state(self):
    assert self.initial_state is not None, "Error happened while requesting initial pose."
    
    return InitialState(x=self.initial_state, P=np.eye(self.initial_state.get_vector_size()) * 0.1)
  
  def _get_sensor_list(self, type: str, sensors: List[SensorConfig]) -> List[ExtendedSensorConfig]:
    sensor_list: List[ExtendedSensorConfig] = []
    get_sensor_from_str = SensorType.get_sensor_from_str_func(type)
    
    logger.debug("Configured sensors are: ")
    for s in sensors:
      sensor_type = get_sensor_from_str(s.name)
      if sensor_type is None:
        logger.warning(f"The sensor: {s} does not exist in {type} dataset.")
        continue
      sensor_list.append(ExtendedSensorConfig(
        name=s.name,
        dropout_ratio=s.dropout_ratio,
        window_size=s.window_size,
        sensor_type=sensor_type
      ))
      logger.debug(sensor_type)
      
    return sensor_list
  
  def is_queue_empty(self) -> bool:
    return self.output_queue.empty()
  
class UAVDataset(BaseDataset):

  sensor_threads: List[Sensor] = []
  imu_noise_vector = None
  
  def __init__(
      self, 
      uav_sensor_path="./uav_sensor_path.yaml",
      imu_config_path="./uav_imu_config.yaml",
      **kwargs,
    ):
    
    super().__init__(**kwargs)
    
    assert self.config.variant in ["log0001", "log0002", "log0003"],\
            "Please provide proper uav variant."
    
    self.root_path = os.path.join(self.config.root_path, self.config.variant)
    self.sensor_list = self._get_sensor_list(type="uav", sensors=self.config.sensors)

    assert len(self.sensor_list) > 0,\
            "Please select sensors"
    
    filepath = None
    with open(uav_sensor_path, "r") as f:
      filepath = yaml.safe_load(f)
      f.close()
      
    self.uav_sensor = namedtuple('uav_sensor', ['type', 'px4', 'voxl'])(**filepath[self.config.variant])
    
    self.px4_path = namedtuple('px4', [
      'imu0_gyro', 'imu0_acc', 'imu1_gyro', 'imu1_acc',
      'gps', 'visual_odometry', 'actuator_motors', 'actuator_outputs', 'mag'
      ])(**self.uav_sensor.px4)
    
    self.voxl_path = namedtuple('voxl', [
      'imu0', 'imu1', 'stereo', 'qvio_overlay'
      ])(**self.uav_sensor.voxl)
    
    imu_configs = None
    with open(imu_config_path, "r") as f:
      imu_configs = yaml.safe_load(f)
      f.close()
    
    self.imu_config = namedtuple('IMU_Configs', ["icm_42688_p", "icm_20948", "icm_20602", "icm_42688"])(**imu_configs)
      
    
    self._populate_sensor_to_thread()

  def _populate_sensor_to_thread(self) -> List[Sensor]:
    
    px4_path = os.path.join(self.root_path, "px4")
    voxl_path = os.path.join(self.root_path, "run/mpa")
    
    def _get_dataset(sensor: ExtendedSensorConfig):
      match (sensor.sensor_type):
        case UAV_SensorType.VOXL_IMU0:
          data_reader = VOXL_IMUDataReader(
              path=os.path.join(voxl_path, self.voxl_path.imu0),
              gyro_spec=GyroSpecification(**self.imu_config.icm_42688_p["gyroscope"]),
              acc_spec=AccelSpecification(**self.imu_config.icm_42688_p["accelerometer"]),
              window_size=sensor.window_size
            )
          data_reader.set_starttime(603023850.803)
          return data_reader
        case UAV_SensorType.VOXL_IMU1:
          data_reader = VOXL_IMUDataReader(
              path=os.path.join(voxl_path, self.voxl_path.imu1),
              gyro_spec=GyroSpecification(**self.imu_config.icm_20948["gyroscope"]),
              acc_spec=AccelSpecification(**self.imu_config.icm_20948["accelerometer"]),
              window_size=sensor.window_size
            )
          data_reader.set_starttime(603023850.803)
          return data_reader
        case UAV_SensorType.VOXL_STEREO:
          return VOXL_StereoFrameReader(
            path=os.path.join(voxl_path, self.voxl_path.stereo),
            image_root_path=os.path.join(voxl_path, self.voxl_path.stereo.split("/")[0])
          )
        case UAV_SensorType.VOXL_QVIO_OVERLAY:
          return VOXL_QVIOOverlayDataReader(
            path=os.path.join(voxl_path, self.voxl_path.qvio_overlay)
          )
        case UAV_SensorType.PX4_IMU0:
          data_reader = PX4_IMUDataReader(
              gyro_path=os.path.join(px4_path, self.px4_path.imu0_gyro),
              acc_path=os.path.join(px4_path, self.px4_path.imu0_acc),
              gyro_spec=GyroSpecification(**self.imu_config.icm_20602["gyroscope"]),
              acc_spec=AccelSpecification(**self.imu_config.icm_20602["accelerometer"]),
              window_size=sensor.window_size,
            )
          return data_reader
        case UAV_SensorType.PX4_IMU1:
          data_reader = PX4_IMUDataReader(
              gyro_path=os.path.join(px4_path, self.px4_path.imu1_gyro),
              acc_path=os.path.join(px4_path, self.px4_path.imu1_acc),
              gyro_spec=GyroSpecification(**self.imu_config.icm_42688["gyroscope"]),
              acc_spec=AccelSpecification(**self.imu_config.icm_42688["accelerometer"]),
              window_size=sensor.window_size
            )
          return data_reader
        case UAV_SensorType.PX4_GPS:
          return PX4_GPSDataReader(
              path=os.path.join(px4_path, self.px4_path.gps),
              divider=1
            )
        case UAV_SensorType.PX4_VO:
          return PX4_VisualOdometryDataReader(
            path=os.path.join(px4_path, self.px4_path.visual_odometry),
            divider=1,
          )
        case UAV_SensorType.PX4_MAG:
          return PX4_MagnetometerDataReader(
            path=os.path.join(px4_path, self.px4_path.mag),
            divider=1
          )
        case SensorType.GROUND_TRUTH:
          # NOTE: Currently, GPS data is set as a ground truth in UAV dataset
          return PX4_GPSDataReader(
              path=os.path.join(px4_path, self.px4_path.gps),
              divider=1
            )
        case UAV_SensorType.UAV_CUSTOM_VO:
          return UAVCustomVisualOdometryDataReader(
            path=os.path.join(px4_path, self.px4_path.visual_odometry),
            divider=1,
          )
        # case UAV_SensorType.PX4_ACTUATOR_MOTORS:
        #   return PX4_IMUDataReader() # FIXME: Create data reader for PX4 actuator motor speed
        # case UAV_SensorType.PX4_ACTUATOR_OUTPUTS:
        #   return PX4_IMUDataReader() # FIXME: Create data reader for PX4 actuator output
        case _:
          return None
    
    sensor_threads = []
    
    for sensor in self.sensor_list:
      dataset = _get_dataset(sensor)
      if dataset is not None:
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
      case UAV_SensorType.VOXL_IMU0 | UAV_SensorType.VOXL_IMU1\
          | UAV_SensorType.PX4_IMU0 | UAV_SensorType.PX4_IMU1:
        
        u = np.hstack([sensor_data.data.a, sensor_data.data.w])
        dt = (timestamp - last_timestamp) / 1e6
        return SensorField(
            type=sensor_data.type, 
            timestamp=timestamp, 
            data=TimeUpdateField(u, dt, Q)
          )
      
      case UAV_SensorType.PX4_VO:
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
              z=z, 
              R=R, 
              sensor_type=sensor_data.type
            )
          )
        
      case UAV_SensorType.PX4_GPS:
        z = np.array([sensor_data.data.lon, sensor_data.data.lat, sensor_data.data.alt])
        return SensorField(
            type=sensor_data.type, 
            timestamp=timestamp, 
            data=MeasurementUpdateField(
              z=z, 
              R=R, 
              sensor_type=sensor_data.type
            )
          )
        
      case UAV_SensorType.PX4_MAG:
        z = np.array([sensor_data.data.x, sensor_data.data.y, sensor_data.data.z])
        return SensorField(
            type=sensor_data.type, 
            timestamp=timestamp, 
            data=MeasurementUpdateField(
              z=z, 
              R=R, 
              sensor_type=sensor_data.type
            )
          )
        
      case UAV_SensorType.VOXL_STEREO:
        return SensorField(
            type=sensor_data.type, 
            timestamp=timestamp, 
            data=StereoField(
              left_frame_id=sensor_data.data.left_frame_id,
              right_frame_id=sensor_data.data.right_frame_id,
            )
          )
      
      case UAV_SensorType.UAV_CUSTOM_VO:
        return SensorField(
            type=sensor_data.type, 
            timestamp=timestamp, 
            data=MeasurementUpdateField(
              z=sensor_data.data.delta_pose, 
              R=R, 
              sensor_type=sensor_data.type
            )
          )
      case UAV_SensorType.UWB:
        # TODO: run uwb triangulation and rotate
        z = np.zeros(3)
        return SensorField(
            type=sensor_data.type, 
            timestamp=timestamp, 
            data=MeasurementUpdateField(
              z=z, 
              R=R, 
              sensor_type=sensor_data.type
            )
          )
      
      case SensorType.GROUND_TRUTH:
        z = np.array([sensor_data.data.lon, sensor_data.data.lat, sensor_data.data.alt])
        return SensorField(
            type=sensor_data.type, 
            timestamp=timestamp, 
            data=z
          )
        
      case UAV_SensorType.VOXL_QVIO_OVERLAY:
        return SensorField(
            type=sensor_data.type, 
            timestamp=timestamp, 
            data=sensor_data.data.image_path
          )
      case _:
        return SensorField(
            type=sensor_data.type, 
            timestamp=timestamp, 
            data=MeasurementUpdateField(
              z=np.zeros(3), 
              R=np.eye(3), 
              sensor_type=sensor_data.type
            )
          )


class KITTIDataset(BaseDataset): 

  sensor_threads = None
  
  def __init__(
      self, 
      **kwargs,
    ):
    super().__init__(**kwargs)
    
    
    date = KITTI_DATE_MAPS.get(self.config.variant)
    assert date is not None, "Please provide proper kitti drive variant."
    
    self.kitti_date = date
    self.sensor_list = self._get_sensor_list(type="kitti", sensors=self.config.sensors)

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
        # case KITTI_SensorType.OXTS_INS:
        #   return OXTS_INSDataReader(**kwargs)
        case KITTI_SensorType.OXTS_GPS:
          return KITTI_GroundTruthDataReader(**kwargs)
          # return OXTS_GPSDataReader(**kwargs)
        case KITTI_SensorType.KITTI_STEREO:
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
        if z.shape[0] == 3:
          R = self.noise_manager.get_measurement_noise(sensor_type=SensorType.KITTI_CUSTOM_VO_VELOCITY_ONLY)
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
  
  def _load_dataset(dataset="uav"):
    
    if dataset == "uav":
      dataset_config = DatasetConfig(
        type=dataset,
        mode="stream",
        root_path="../../data/UAV",
        variant="log0001",
        sensors=[
            SensorConfig(name='voxl_imu0', dropout_ratio=0), 
            SensorConfig(name='px4_mag', dropout_ratio=0),
            SensorConfig(name='px4_gps', dropout_ratio=0),
            SensorConfig(name='voxl_stereo', dropout_ratio=0)
          ]
      )
      kwargs = {
        'config': dataset_config,
      }
      dataset = UAVDataset(**kwargs)
    else:
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
      dataset = KITTIDataset(**kwargs)
      
    # Start the threads
    dataset.start()
    time.sleep(0.5)
    
    while True:
      if dataset.is_queue_empty():
        break
      
      sensor_data = dataset.get_sensor_data()
      if SensorType.is_time_update(sensor_data.type):
        print(sensor_data.timestamp, sensor_data.type)
        ...
      elif SensorType.is_measurement_update(sensor_data.type):
        ...
      elif SensorType.is_stereo_image_data(sensor_data.type):
        # print(sensor_data.timestamp, sensor_data.type)
        ...
  
  dataset = "uav"
  
  _load_dataset(dataset=dataset)