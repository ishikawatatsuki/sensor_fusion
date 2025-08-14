import os
import sys
import abc
import logging
import numpy as np

from ..common import (
    FilterConfig,
    HardwareConfig,
    MotionModel,
    NoiseType,
    SensorType,
    SensorDataField,
    FilterType
)


class BaseNoise(abc.ABC):
  
    def __init__(
        self,
        motion_model: MotionModel,
        hardware_config: HardwareConfig
    ):
      
        self.Q = dict()
        self.R = dict()
        self.motion_model = motion_model

        self.imu_hardware_config = hardware_config.imu_config


        self.Rs, self.alpha = self._initialize_R()
        self.Qs = self._initialize_Q()
      
      
    def register_process_noise(self, sensor_type: SensorType):
        pass
        # if SensorType.is_time_update(sensor_type):
        #   if self.Q.get(sensor_type) is None:
        #     self.Q[sensor_type] = np.hstack([np.tile(dataset.acc_noise, 3), np.tile(dataset.gyro_noise, 3)])

    def _initialize_R(self):
        def _get_initial_measurement_noise(sensor_type: SensorType):
            match (sensor_type.name):
                case SensorType.OXTS_GPS.name | SensorType.OXTS_GPS_UNSYNCED.name:
                    noise_vector = np.array([1.0, 1.0, 1.0])
                    return np.eye(noise_vector.shape[0]) * noise_vector**2

                case SensorType.KITTI_VO.name:
                    noise_vector = np.array(
                        [3., 3., 10., .15, .15, .15, 0.5, 0.5, 0.5, 0.5])
                    return np.eye(noise_vector.shape[0]) * noise_vector**2
                
                case SensorType.KITTI_UPWARD_LEFTWARD_VELOCITY.name:
                    return np.eye(2) * np.array([0.1, 0.1])**2

                case _:
                    logging.warning(
                        f"Not registered sensor type appeared {sensor_type.name}")
                    return np.eye(0)
        logging.debug("Initializing measurement covariance matrix R")
        
        Rs = dict()
        alphas = dict()
        for sensor_type in SensorType.get_all_sensors():
            if SensorType.is_measurement_update(sensor_type):
                Rs[sensor_type] = _get_initial_measurement_noise(sensor_type=sensor_type)
                alphas[sensor_type] = 0.3
        
        return Rs, alphas
    
    def _initialize_Q(self):
        """Initialize process noise covariance matrix Q for each sensor type."""
        p_noise = np.repeat(2.0, 3)
        v_noise = np.repeat(0.5, 3)
        q_noise = np.repeat(0.01, 4)
        b_w_noise = np.repeat(self.imu_hardware_config.gyroscope_random_walk, 3)
        b_a_noise = np.repeat(self.imu_hardware_config.accelerometer_random_walk, 3)


        q = np.hstack([p_noise, v_noise, q_noise, b_w_noise, b_a_noise])

        Q = np.eye(q.shape[0]) * q ** 2


        Qs = {
            SensorType.OXTS_IMU: Q,
            SensorType.OXTS_IMU_UNSYNCED: Q,
        }
        return Qs

    @abc.abstractmethod
    def get_process_noise(self, sensor_data: SensorDataField):
        """Returns process noise covariance matrix Q for the given sensor type.

        Args:
            sensor_type (SensorType): Sensor type used for time update step.

        Returns:
            _type_: _description_
        """
        pass
      
    @abc.abstractmethod
    def get_measurement_noise(self, sensor_data: SensorDataField):
        """Returns measurement noise covariance matrix R for the given sensor type.

        Args:
            sensor_type (SensorType): Sensor type used for measurement update step.

        Returns:
            _type_: _description_
        """
        pass


class DynamicNoise(BaseNoise):
  
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
  
  def get_process_noise(self, sensor_data: SensorDataField):
    ...
  def get_measurement_noise(self, sensor_data: SensorDataField):
    ...
    
  def get_vo_measurement_noise(self, vo_estimation, z_dim: int) -> np.ndarray:
    ...
    
class OptimalNoise(BaseNoise):
    
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
  
  def get_process_noise(self, sensor_data: SensorDataField):
    ...
  
  def get_measurement_noise(self, sensor_data: SensorDataField):
    ...
    
  def get_vo_measurement_noise(self, vo_estimation, z_dim: int) -> np.ndarray:
    ...


class DefaultNoise(BaseNoise):
  
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
  
  def get_process_noise(self, sensor_data: SensorDataField):

    return self.Qs.get(sensor_data.type, None)

  def get_measurement_noise(self, sensor_data: SensorDataField):
      if not SensorType.is_measurement_update(sensor_data.type):
          return None

      return self.Rs.get(sensor_data.type, None)
  

class NoiseManager:
    def __init__(
        self,
        filter_config: FilterConfig,
        hardware_config: HardwareConfig
    ):
        self.filter_type = FilterType.get_filter_type_from_str(filter_type=filter_config.type)
        self.motion_model = MotionModel.get_motion_model(filter_config.motion_model)
        self.noise_type = NoiseType.get_noise_type_from_str(filter_config.noise_type)
        self.hardware_config = hardware_config

        self._provider = self._get_noise_provider()
      
    def _get_noise_provider(self):
      
        kwargs = {
            "motion_model": self.motion_model,
            "hardware_config": self.hardware_config
        }
        match (self.noise_type):
            case NoiseType.DEFAULT:
                return DefaultNoise(**kwargs)
            case NoiseType.DYNAMIC:
                return DynamicNoise(**kwargs)
            case _:
                return DefaultNoise(**kwargs)

    def get_process_noise(self, **kwargs) -> np.ndarray:
        return self._provider.get_process_noise(**kwargs)
    
    def get_measurement_noise(self, **kwargs) -> np.ndarray:
        return self._provider.get_measurement_noise(**kwargs)