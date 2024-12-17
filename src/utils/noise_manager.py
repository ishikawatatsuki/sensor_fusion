import os
import sys
import logging
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../interfaces'))

from custom_types import FilterType, SensorType, NoiseType, MotionModel
from config import FilterConfig

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s > %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG)
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
    
    
class BaseNoise:
  
  def __init__(
    self,
    motion_model: MotionModel,
  ):
    
    self.Q = dict()
    self.R = dict()
    self.motion_model = motion_model
    
    
  def register_process_noise(self, sensor_type: SensorType, dataset):
    
    if SensorType.is_time_update(sensor_type):
      if self.Q.get(sensor_type) is None:
        self.Q[sensor_type] = np.hstack([np.tile(dataset.acc_noise, 3), np.tile(dataset.gyro_noise, 3)])

class DynamicNoise(BaseNoise):
  
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
  
  def get_process_noise(self, sensor_type: SensorType):
    ...
  def get_measurement_noise(self, sensor_type: SensorType):
    ...
    
  def get_vo_measurement_noise(self, vo_estimation, z_dim: int) -> np.ndarray:
    # TODO: 1. find relation between estimated delta translation and delta rotation with those in ground truth. calculate the translation error and rotation error at each time t (frame by frame estimation error).
    # TODO: 2. find relation between those errors and n_keypoints and good_matches. visualize each combinations: n_keypoints vs translation_error, n_keypoints vs rotation_error, good_matches vs translation_error, good_matches vs rotation_error, n_keypoints vs good_matches vs translation_error, and rotation_error, n_keypoints vs good_matches vs rotation_error.
    # TODO: 3. find regression params: a and b in y = ax + b.
    
    alpha = 2
    
    h = 0. / 1000
    y = h if h < 1. else 1.
    scale_factor = 10 ** (alpha * y)
    
    if z_dim == 6:
      return np.eye(6) * np.array([1., 1., 1., .1, .1, .1]) * scale_factor # in range [1; 1000) when alpha=2
    return np.eye(3) * np.array([.1, .1, .1]) * scale_factor # in range [0.1; 100) when alpha=2
    
class DefaultNoise(BaseNoise):
  
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
  
  def get_process_noise(self, sensor_type: SensorType):
    # Q = self.Q.get(sensor_type)
    # if Q is None:
    #   return None
    
    return np.eye(10) * np.array([2., 2., 2., .5, .5, .5, 0.01, 0.01, 0.01, 0.01]) ** 2
    
  def get_measurement_noise(self, sensor_type: SensorType):
    
    if not SensorType.is_measurement_update(sensor_type):
      return None
    
    R = self.R.get(sensor_type)
    if R is not None:
      return R
    
    match (sensor_type.name):
      case SensorType.PX4_VO.name:
        R = np.eye(6) * np.array([3.0, 3.0, 3.0, .5, .5, .5]) ** 2
      case SensorType.UAV_CUSTOM_VO.name:
        R = np.eye(3) * np.array([3.0, 3.0, 3.0]) ** 2
      case SensorType.PX4_MAG.name:
        R = np.array([[0.001]]) ** 2
      case SensorType.PX4_GPS.name:
        R = np.eye(3) * np.array([1.0, 1.0, 1.0]) ** 2
      case SensorType.UWB.name:
        R = np.eye(3) * np.array([0.2, 0.2, 0.2]) ** 2
      case SensorType.OXTS_GPS.name:
        R = np.eye(3) * np.array([1., 1., 2.]) ** 2
      case SensorType.KITTI_CUSTOM_VO.name:
        R = np.eye(6) * np.array([3., 3., 10., .15, .15, .15]) ** 2
      case SensorType.KITTI_CUSTOM_VO_VELOCITY_ONLY.name:
        R = np.eye(3) * np.array([1.5, 1.5, 1.5]) ** 2
      case SensorType.KITTI_UPWARD_LEFTWARD_VELOCITY.name | SensorType.VIZTRACK_UPWARD_LEFTWARD_VELOCITY.name:
        R = np.eye(2) * np.array([0.1, 0.1]) ** 2
      case _:
        logger.warning(f"Not registered sensor type appeared {sensor_type}")
    
    self.R[sensor_type] = R
    return self.R[sensor_type]
  
class OptimalNoise(BaseNoise):
  
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    
  def get_process_noise(self, sensor_type: SensorType):
    # Q = self.Q.get(sensor_type)
    # if Q is None:
    #   return None
    
    if self.motion_model is MotionModel.KINEMATICS:    
      match (self.filter_type):
        case FilterType.EKF:
          return np.eye(10) * np.array([0.01, 67.41922287280015, 0.7494152064351436, 1000.0, 1000.0, 714.0250583141932, 1e-05, 1e-05, 0.0022092728926040926, 0.00011590354305331916])
        case FilterType.UKF:
          return np.eye(10) * np.array([1000.0, 1000.0, 1000.0, 1.301367123522085, 1000.0, 1000.0, 1.0, 1.0, 1.0, 0.00017544970595799445])
        case FilterType.PF:
          return np.eye(10) * np.array([249.21237219227166, 12.20399908738481, 0.03588744104725751, 1.5586037141903673, 1.1038565658724064, 1.594112300226227, 0.28418988590663163, 0.007908521415237886, 0.0003909634307197494, 0.0059353348767415695])
        case FilterType.EnKF:
          return np.eye(10) * np.array([1000.0, 1000.0, 0.10317833736637362, 0.6439568019875933, 1000.0, 1000.0, 0.0006217822248112819, 1.0, 1.0, 0.008578923104614692])
        case FilterType.CKF:
          return np.eye(10) * np.array([1000.0, 1000.0, 0.01, 0.043934344271553, 0.01, 0.11738175104229587, 1.0, 1e-05, 0.00025238068108866514, 1.0])
        case _:
          return np.eye(10) * np.ones((10, ))
    else:
      match (self.filter_type):
        case FilterType.EKF:
          return np.eye(10) * np.array([0.3888959891501419, 1000.0, 1000.0, 0.01, 0.3198469561118217, 3.6432061727018694, 1.0, 1e-05, 1.0, 1.0])
        case FilterType.UKF:
          return np.eye(10) * np.array([1000.0, 1000.0, 1000.0, 0.01, 0.3274837690187921, 116.41851222130477, 1.0, 0.032318702269830316, 1.0, 1e-05])
        case FilterType.PF:
          return np.eye(10) * np.array([10.653489422380202, 184.78967451984616, 0.04932205932140033, 2.5572037508247285, 0.2582531190873234, 0.6955689270455638, 0.00024734525835096166, 0.004926351709833528, 1e-05, 0.029480469530866072])
        case FilterType.EnKF:
          return np.eye(10) * np.array([1.5092861316509083, 94.57508790908766, 1000.0, 0.01, 0.013653713642398915, 1000.0, 1e-05, 1e-05, 1.0, 0.8075225140378695])
        case FilterType.CKF:
          return np.eye(10) * np.array([1000.0, 0.09333978491852399, 322.1905268661857, 522.2906967100026, 3.9780547358029654, 9.589890855632401, 0.06131895160471703, 0.0011148945211455002, 0.0007544812203883254, 0.013600363065037927])
        case _:
          return np.eye(10) * np.ones((10, ))

  def get_measurement_noise(self, sensor_type: SensorType):
    if not SensorType.is_measurement_update(sensor_type):
      return None
    
    R = self.R.get(sensor_type)
    if R is not None:
      return R
    match (sensor_type.name):
      case SensorType.PX4_VO.name:
        R = np.eye(6) * np.array([10.0, 10.0, 10.0, .5, .5, .5]) ** 2
      case SensorType.UAV_CUSTOM_VO.name:
        R = np.eye(6) * np.array([10.0, 10.0, 10.0, .5, .5, .5]) ** 2
      case SensorType.PX4_MAG.name:
        R = np.array([[0.001]]) ** 2
      case SensorType.PX4_GPS.name:
        R = np.eye(3) * np.array([1.0, 1.0, 1.0]) ** 2
      case SensorType.UWB.name:
        R = np.eye(3) * np.array([0.2, 0.2, 0.2]) ** 2
      case SensorType.OXTS_GPS.name:
        R = np.eye(3) * np.array([1., 1., 2.]) ** 2
      case SensorType.KITTI_CUSTOM_VO.name:
        R = np.eye(6) * np.array([3., 3., 10., .15, .15, .15]) ** 2
      case SensorType.KITTI_CUSTOM_VO_VELOCITY_ONLY.name:
        R = np.eye(3) * np.array([1., 1., 1.]) ** 2
      case SensorType.KITTI_UPWARD_LEFTWARD_VELOCITY.name | SensorType.VIZTRACK_UPWARD_LEFTWARD_VELOCITY.name:
        R = np.eye(2) * np.array([0.1, 0.1]) ** 2
      case _:
        logger.warning(f"Not registered sensor type appeared {sensor_type}")
    
    self.R[sensor_type] = R
    return self.R[sensor_type]
    

class NoiseManager:
  # TODO: If variants of dataset or motion model increases, save the noise parameters to numpy file and load them when noise manager is initialized.
  
  def __init__(
    self,
    filter_config: FilterConfig
  ):
    self.filter_type = FilterType.get_filter_type_from_str(filter_type=filter_config.type)
    self.motion_model = MotionModel.get_motion_model(filter_config.motion_model)
    self.noise_type = NoiseType.get_noise_type_from_str(filter_config.noise_type)
    
    self._provider = self._get_noise_provider()
    
  def _get_noise_provider(self):
    
    kwargs = {
      "motion_model": self.motion_model
    }
    match (self.noise_type):
      case NoiseType.DEFAULT:
        return DefaultNoise(**kwargs)
      case NoiseType.OPTIMAL:
        return OptimalNoise(**kwargs)
      case NoiseType.DYNAMIC:
        return DynamicNoise(**kwargs)
      case _:
        return DefaultNoise(**kwargs)
      
  def get_process_noise(self, **kwargs):
    return self._provider.get_process_noise(**kwargs)
  
  def get_measurement_noise(self, **kwargs):
    return self._provider.get_measurement_noise(**kwargs)