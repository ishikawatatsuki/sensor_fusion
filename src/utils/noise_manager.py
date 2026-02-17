import os
import sys
import abc
import logging
import numpy as np
from sqlalchemy import case

from ..common import (
    FusionData,
    FilterConfig,
    HardwareConfig,
    MotionModel,
    NoiseType,
    SensorType,
    SensorDataField,
    FilterType,
    SensorNoiseConfig
)
from .noise_transforms import get_transform_function


class BaseNoise(abc.ABC):
    
    def __init__(
        self,
        filter_config: FilterConfig,
        hardware_config: HardwareConfig
    ):
    
        self.Q = dict()
        self.R = dict()
        self.filter_config = filter_config
        self.motion_model = MotionModel.get_motion_model(filter_config.motion_model)

        self.imu_hardware_config = hardware_config.imu_config


        self.Rs, self.alpha = self._initialize_R()
        self.Qs = self._initialize_Q()

    def register_process_noise(self, sensor_type: SensorType):
        pass
        # if SensorType.is_time_update(sensor_type):
        #   if self.Q.get(sensor_type) is None:
        #     self.Q[sensor_type] = np.hstack([np.tile(dataset.acc_noise, 3), np.tile(dataset.gyro_noise, 3)])

    def _initialize_R(self):
        """Initialize measurement noise covariance matrix R using config transformation functions."""
        def _get_initial_measurement_noise(sensor_type: SensorType):
            # Check if sensor has noise config with transformation function
            sensor_config = self.filter_config.sensors.get(sensor_type, None)
            
            if sensor_config and sensor_config.get('noise'):
                noise_config: SensorNoiseConfig = sensor_config['noise']
                
                # Only process measurement noise here
                if noise_config.type == 'measurement' and noise_config.transformation:
                    try:
                        transform_fn = get_transform_function(noise_config.transformation)
                        
                        # Get fusion fields to pass to transformation if needed
                        fusion_fields = sensor_config.get('fields', [])
                        field_names = [field.name.lower() for field in fusion_fields]
                        
                        # Add fields to params if the transformation needs it
                        params = noise_config.params.copy()
                        if 'fields' not in params and transform_fn.__name__ == 'vo_noise_to_measurement_noise':
                            params['fields'] = field_names
                        
                        noise_variance = transform_fn(**params)
                        logging.debug(f"Using transformation '{noise_config.transformation}' for {sensor_type.name} measurement noise with scale: {noise_config.scale}")
                        return np.diag(noise_variance) * noise_config.scale
                    except Exception as e:
                        logging.error(f"Failed to apply transformation '{noise_config.transformation}' for {sensor_type.name}: {e}")
            
            # Fallback to default hard-coded values if no transformation configured
            fusion_fields = sensor_config.get('fields', []) if sensor_config else []
            return self._get_default_measurement_noise(sensor_type, fusion_fields)
        
        logging.debug("Initializing measurement covariance matrix R")
        
        Rs = dict()
        alphas = dict()
        for sensor_type in SensorType.get_all_sensors():
            if SensorType.is_measurement_update(sensor_type):
                Rs[sensor_type] = _get_initial_measurement_noise(sensor_type=sensor_type)
                alphas[sensor_type] = 0.3
        
        return Rs, alphas
    
    def _get_default_measurement_noise(self, sensor_type: SensorType, fusion_fields: list) -> np.ndarray:
        """Fallback default measurement noise when no transformation is configured."""
        match (sensor_type.name):
            case SensorType.OXTS_GPS.name | SensorType.OXTS_GPS_UNSYNCED.name:
                noise_vector = np.array([1.0, 1.0, 1.0])
                return np.eye(noise_vector.shape[0]) * noise_vector**2
            
            case SensorType.KITTI_VO.name:
                if len(fusion_fields) == 0:
                    logging.warning(f"Fusion field is not set for sensor: {sensor_type.name}")
                noise_vector = np.empty(0)
                if FusionData.POSITION in fusion_fields:
                    noise_vector = np.append(noise_vector, np.array([3., 3., 10]))
                if FusionData.LINEAR_VELOCITY in fusion_fields:
                    noise_vector = np.append(noise_vector, np.array([.5, .5, .5]))
                if FusionData.ORIENTATION in fusion_fields:
                    noise_vector = np.append(noise_vector, np.array([0.01, 0.01, 0.01, 0.01]))
                return np.eye(noise_vector.shape[0]) * noise_vector**2
            
            case SensorType.KITTI_UPWARD_LEFTWARD_VELOCITY.name:
                return np.eye(2) * np.array([.5, .5])**2
            
            case SensorType.PX4_GPS.name:
                return np.eye(3) * np.array([1.0, 1.0, 1.0])**2
            
            case SensorType.PX4_VO.name | SensorType.UAV_VO.name:
                noise_vector = np.empty(0)
                if FusionData.POSITION in fusion_fields:
                    noise_vector = np.append(noise_vector, np.array([2.5, 2.5, 5]))
                if FusionData.LINEAR_VELOCITY in fusion_fields:
                    noise_vector = np.append(noise_vector, np.array([.5, .5, .5]))
                if FusionData.ORIENTATION in fusion_fields:
                    noise_vector = np.append(noise_vector, np.array([0.01, 0.01, 0.01, 0.01]))
                return np.eye(noise_vector.shape[0]) * noise_vector**2
            
            case SensorType.PX4_MAG.name:
                return np.eye(1) * np.array([0.1])**2
            
            case SensorType.PX4_CUSTOM_IMU.name:
                return np.eye(4) * np.array([0.01, 0.01, 0.01, 0.01])**2
            
            case SensorType.EuRoC_LEICA.name:
                return np.eye(3) * np.array([1., 1.0, 1.0])**2
            
            case SensorType.EuRoC_VO.name:
                noise_vector = np.empty(0)
                if FusionData.POSITION in fusion_fields:
                    noise_vector = np.append(noise_vector, np.array([3., 3., 3]))
                if FusionData.LINEAR_VELOCITY in fusion_fields:
                    noise_vector = np.append(noise_vector, np.array([.5, .5, .5]))
                if FusionData.ORIENTATION in fusion_fields:
                    noise_vector = np.append(noise_vector, np.array([0.01, 0.01, 0.01, 0.01]))
                return np.eye(noise_vector.shape[0]) * noise_vector**2
            
            case _:
                logging.warning(f"No default measurement noise for sensor: {sensor_type.name}")
                return np.eye(0)
    
    def _initialize_Q(self):
        """Initialize process noise covariance matrix Q using config transformation functions."""
        def _get_initial_process_noise(sensor_type: SensorType):
            """Returns process noise covariance matrix Q for the given sensor type."""
            # Check if sensor has noise config with transformation function
            sensor_config = self.filter_config.sensors.get(sensor_type, None)
            
            if sensor_config and sensor_config.get('noise'):
                noise_config: SensorNoiseConfig = sensor_config['noise']
                
                # Only process process noise here
                if noise_config.type == 'process' and noise_config.transformation:
                    try:
                        transform_fn = get_transform_function(noise_config.transformation)
                        noise_variance = transform_fn(**noise_config.params)
                        logging.debug(f"Using transformation '{noise_config.transformation}' for {sensor_type.name} process noise")
                        
                        # Handle drone kinematics extension if needed
                        if self.motion_model is MotionModel.DRONE_KINEMATICS:
                            drone_noise = np.repeat(0.01**2, 3)
                            noise_variance = np.concatenate([noise_variance, drone_noise])
                        
                        logging.debug(f"Using transformation '{noise_config.transformation}' for {sensor_type.name} process noise with scale: {noise_config.scale}")
                        return np.diag(noise_variance) * noise_config.scale
                    except Exception as e:
                        logging.error(f"Failed to apply transformation '{noise_config.transformation}' for {sensor_type.name}: {e}")
            
            # Fallback to default hard-coded values
            return self._get_default_process_noise(sensor_type)
        
        Qs = dict()
        for sensor_type in SensorType.get_all_sensors():
            if SensorType.is_time_update(sensor_type):
                Qs[sensor_type] = _get_initial_process_noise(sensor_type=sensor_type)
        
        return Qs
    
    def _get_default_process_noise(self, sensor_type: SensorType) -> np.ndarray:
        """Fallback default process noise when no transformation is configured."""
        q_noise = np.repeat(0.01, 4)
        b_w_noise = np.repeat(self.imu_hardware_config.gyroscope_random_walk, 3)
        b_a_noise = np.repeat(self.imu_hardware_config.accelerometer_random_walk, 3)
        
        # Set position and velocity noise based on sensor type
        match (sensor_type.name):
            case SensorType.OXTS_IMU.name | SensorType.OXTS_IMU_UNSYNCED.name:
                p_noise = np.repeat(1.5, 3)
                v_noise = np.repeat(0.5, 3)
            case SensorType.EuRoC_IMU.name:
                p_noise = np.repeat(1.75, 3)
                v_noise = np.repeat(0.6, 3)
            case SensorType.PX4_IMU0.name | SensorType.PX4_IMU1.name |\
                    SensorType.VOXL_IMU0.name | SensorType.VOXL_IMU1.name:
                p_noise = np.repeat(2.5, 3)
                v_noise = np.repeat(0.7, 3)
            case SensorType.PX4_ACTUATOR_MOTORS.name | SensorType.PX4_ACTUATOR_OUTPUTS.name:
                p_noise = np.repeat(2., 3)
                v_noise = np.repeat(0.7, 3)
            case _:
                p_noise = np.repeat(3., 3)
                v_noise = np.repeat(1., 3)
        
        q = np.hstack([p_noise, v_noise, q_noise, b_w_noise, b_a_noise])
        
        if self.motion_model is MotionModel.DRONE_KINEMATICS:
            q = np.hstack([q, np.repeat(0.01, 3)])
        
        return np.eye(q.shape[0]) * q ** 2

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

class AdaptiveNoise(BaseNoise):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        filter_config: FilterConfig = kwargs.get("filter_config", None)

        assert filter_config is not None, "Filter config must be provided"

        self.process_data_type = None
        self.alpha = filter_config.noise_management.params.get("alpha", 0.9)
    
    def get_process_noise(self, sensor_data: SensorDataField):
        self.process_data_type = sensor_data.type
        return self.Qs.get(sensor_data.type, None)

    def get_measurement_noise(self, sensor_data: SensorDataField):
        if not SensorType.is_measurement_update(sensor_data.type):
            return None
        
        return self.Rs.get(sensor_data.type, None)
        
    def update_noise_matrix(
            self,
            sensor_data: SensorDataField,
            innovation: np.ndarray,
            residual: np.ndarray,
            P: np.ndarray,
            H: np.ndarray,
            K: np.ndarray
        ):
        if not SensorType.is_measurement_update(sensor_data.type):
            return None
        
        R = self.Rs.get(sensor_data.type, None)
        Q = self.Qs.get(self.process_data_type, None)
        if R is None or Q is None:
            return None

        logging.debug(f"Updating noise matrix for sensor: {sensor_data.type.name}")
        logging.debug(f"[Shape] residual: {residual.shape}, innovation: {innovation.shape}, P: {P.shape}, H: {H.shape}, K: {K.shape}")
        # Update R using the residual
        # alpha is called forgetting factor
        """
        The paper introduces a forgetting factor (alpha), which takes a value between 0 and 1 to adaptively estimate Rk. A higher alpha puts more weights on previous estimates and therefore incurs less fluctuation of Rk, and longer time delay to catch up with changes. The paper set alpha to 0.3.
        """
        self.Rs[sensor_data.type] = self.alpha * R + (1 - self.alpha) * (residual @ residual.T + H @ P @ H.T)
        self.Qs[self.process_data_type] = self.alpha * Q + (1 - self.alpha) * (K @ innovation @ innovation.T @ K.T)[:Q.shape[0], :Q.shape[0]]


class DynamicNoise(BaseNoise):
    # This is supposed to be machine learning based noise management
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def get_process_noise(self, sensor_data: SensorDataField):
        ...
    def get_measurement_noise(self, sensor_data: SensorDataField):
        ...
        
    def get_vo_measurement_noise(self, vo_estimation, z_dim: int) -> np.ndarray:
        ...
    
class OptimalNoise(BaseNoise):
    # This is supposed to be optimization based noise management
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
        self.noise_type = NoiseType.get_noise_type_from_str(filter_config.noise_management.type)
        self.hardware_config = hardware_config
        self.filter_config = filter_config

        self._provider = self._get_noise_provider()

    def _get_noise_provider(self):
        
        kwargs = {
            "filter_config": self.filter_config,
            "hardware_config": self.hardware_config
        }
        match (self.noise_type):
            case NoiseType.DEFAULT:
                logging.info("Using Default Noise Provider")
                return DefaultNoise(**kwargs)
            case NoiseType.DYNAMIC:
                logging.info("Using Dynamic Noise Provider")
                return DynamicNoise(**kwargs)
            case NoiseType.ADAPTIVE:
                logging.info("Using Adaptive Noise Provider")
                return AdaptiveNoise(**kwargs)
            case _:
                return DefaultNoise(**kwargs)

    def get_process_noise(self, **kwargs) -> np.ndarray:
        return self._provider.get_process_noise(**kwargs)
    
    def get_measurement_noise(self, **kwargs) -> np.ndarray:
        return self._provider.get_measurement_noise(**kwargs)
    
    def update_noise_matrix(self, **kwargs):
        if self.noise_type != NoiseType.ADAPTIVE:
            return None
        
        self._provider.update_noise_matrix(**kwargs)
