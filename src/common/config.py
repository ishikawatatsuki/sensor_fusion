import logging
import yaml
import numpy as np
import pandas as pd
from enum import Enum
from ahrs import DCM, RAD2DEG
from typing import Dict, List, Union
from collections import namedtuple
from dataclasses import dataclass
from shapely.geometry import LineString


from .datatypes import SensorConfig, VisualizationDataType, State, Pose, SensorType


@dataclass
class FilterConfig:
    type: str
    dimension: int
    motion_model: str
    noise_type: str
    innovation_masking: bool
    params: dict
    is_imu_preintegrated: bool
    compensate_gravity: bool
    use_imu_preprocessing: bool

    def __init__(
        self,
        type: str = "ckf",
        dimension: int = 2,
        motion_model: str = "velocity",
        noise_type: str = "default",
        innovation_masking: bool = False,
        params: dict = {},
        is_imu_preintegrated: bool = False,
        compensate_gravity: bool = False,
        use_imu_preprocessing: bool = False,
        ):
        self.type = type
        self.dimension = dimension
        self.motion_model = motion_model
        self.noise_type = noise_type
        self.innovation_masking = innovation_masking
        self.params = params
        self.is_imu_preintegrated = is_imu_preintegrated
        self.compensate_gravity = compensate_gravity
        self.use_imu_preprocessing = use_imu_preprocessing

    def __str__(self):
        return \
            f"FilterConfig(\n"\
            f"\ttype={self.type}\n" \
            f"\tdimension={self.dimension}\n" \
            f"\tmotion_model={self.motion_model}\n" \
            f"\tnoise_type={self.noise_type}\n" \
            f"\tinnovation_masking={self.innovation_masking}\n" \
            f"\tparams={self.params}\n" \
            f"\tis_imu_preintegrated={self.is_imu_preintegrated}\n" \
            f"\tcompensate_gravity={self.compensate_gravity}\n" \
            f"\tuse_imu_preprocessing={self.use_imu_preprocessing}\n" \
            f")"


@dataclass
class ImuConfig:
    frequency: float
    target_frequency: float

    sigma_gyro: float
    sigma_accel: float
    sigma_gyro_bias: float
    sigma_accel_bias: float

    gyroscope_random_walk: float
    accelerometer_random_walk: float

    def __init__(
            self,
            frequency: float = 100.0,
            target_frequency: float = 10.0,
            gyroscope_noise_density: float = 2.44e-4,
            accelerometer_noise_density: float = 1.86e-3,
            gyroscope_random_walk: float = 1.9393e-05,
            accelerometer_random_walk: float = 3.0000e-3
    ):
        """Initialize IMU configuration.
        This class contains the IMU noise parameters and the frequency of the IMU.

        Args:
            frequency (float, optional): _description_. Defaults to 100.0.
            gyroscope_noise_density (float, optional): [ rad / s / sqrt(Hz) ]   ( gyro "white noise" ). Defaults to 2.44e-4.
            accelerometer_noise_density (float, optional): [ m / s^2 / sqrt(Hz) ]   ( accel "white noise" ). Defaults to 1.86e-3.
            gyroscope_random_walk (float, optional): [ rad / s^2 / sqrt(Hz) ] ( gyro bias diffusion ). Defaults to 1.9393e-05.
            accelerometer_random_walk (float, optional): [ m / s^3 / sqrt(Hz) ].  ( accel bias diffusion ). Defaults to 3.0000e-3.
        """
        self.frequency = frequency
        self.target_frequency = target_frequency

        # TODO: Set IMU noise parameters from config file
        
        dt = 1 / frequency
        self.sigma_gyro = gyroscope_noise_density / np.sqrt(dt)  
        self.sigma_accel = accelerometer_noise_density / np.sqrt(dt)   
        self.sigma_gyro_bias = gyroscope_random_walk * np.sqrt(dt) 
        self.sigma_accel_bias = accelerometer_random_walk * np.sqrt(dt)

        self.gyroscope_random_walk = gyroscope_random_walk
        self.accelerometer_random_walk = accelerometer_random_walk

    @classmethod
    def from_json(cls, json_data: dict):
        """Load IMU config from json data.

        Args:
            json_data (dict): JSON data containing IMU configuration.
        """
        if json_data is None:
            return None

        frequency = json_data.get("frequency", 100.0)
        gyroscope_noise_density = json_data["gyroscope_noise_density"]
        accelerometer_noise_density = json_data["accelerometer_noise_density"]
        gyroscope_random_walk = json_data["gyroscope_random_walk"]
        accelerometer_random_walk = json_data["accelerometer_random_walk"]

        return cls(
            frequency=frequency,
            gyroscope_noise_density=gyroscope_noise_density,
            accelerometer_noise_density=accelerometer_noise_density,
            gyroscope_random_walk=gyroscope_random_walk,
            accelerometer_random_walk=accelerometer_random_walk
        )

@dataclass
class TransformationConfig:
    T_from_cam_to_imu: np.ndarray
    T_from_imu_to_cam: np.ndarray
    T_imu_body_to_inertial: np.ndarray
    
    T_imu_to_virtual_imu: Dict[SensorType, np.ndarray]

    def __init__(self,
                    T_from_cam_to_imu: np.ndarray = None,
                    T_from_imu_to_cam: np.ndarray = None,
                    T_imu_body_to_inertial: np.ndarray = None,
                    T_imu_to_virtual_imu: Dict[SensorType, np.ndarray] = {}):
            """Initialize transformation configuration.
            Args:
                T_from_cam_to_imu (np.ndarray): Transformation matrix from camera to IMU.
                T_from_imu_to_cam (np.ndarray): Transformation matrix from IMU to camera.
                T_imu_body_to_inertial (np.ndarray): Transformation matrix from IMU body frame to inertial frame.
                T_imu_body_to_leica (np.ndarray, optional): Transformation matrix from IMU body frame to Leica frame.
                T_imu_to_virtual_imu (Dict[SensorType, np.ndarray], optional): Transformation matrices for virtual IMUs.
            """
            self.T_from_cam_to_imu = T_from_cam_to_imu if T_from_cam_to_imu is not None else np.eye(4)
            self.T_from_imu_to_cam = T_from_imu_to_cam if T_from_imu_to_cam is not None else np.eye(4)
            self.T_imu_body_to_inertial = T_imu_body_to_inertial if T_imu_body_to_inertial is not None else np.eye(4)
            
            self.T_imu_to_virtual_imu = T_imu_to_virtual_imu
            

    @classmethod
    def from_kitti_config(cls, T_calib_velo_to_cam: np.ndarray, T_calib_imu_to_velo: np.ndarray):
        """Get transformation matrix for the sensor hardware configuration.
        Args:
            dataset_type (str): Type of the dataset (e.g., "kitti", "paldiski", "vlaardingan").
            calibration_root_path (str): Path to the folder that stores calibration files.
        """

        T_from_imu_to_cam = T_calib_imu_to_velo @ T_calib_velo_to_cam
        T_from_cam_to_imu = np.linalg.inv(T_from_imu_to_cam)
        
        return cls(
            T_from_cam_to_imu=T_from_cam_to_imu,
            T_from_imu_to_cam=T_from_imu_to_cam,
            T_imu_body_to_inertial=np.eye(4),
        )

    @classmethod
    def from_uav_config(cls, yaml_config: dict):
        r = np.eye(4)
        return cls(
            T_from_cam_to_imu=r,
            T_from_imu_to_cam=r,
            T_imu_body_to_inertial=r,
            T_imu_to_virtual_imu={},
        )

        
    def __str__(self):
        return \
            f"TransformationConfig(\n" \
            f"\tT_from_cam_to_imu={self.T_from_cam_to_imu}\n" \
            f"\tT_from_imu_to_cam={self.T_from_imu_to_cam}\n" \
            f"\tT_imu_body_to_inertial={self.T_imu_body_to_inertial}\n" \
            f")"


@dataclass
class HardwareConfig:
    type: str
    imu_config: ImuConfig
    transformation: TransformationConfig
    
    def __init__(
            self,
            type: str,
            imu_config: ImuConfig,
            transformation: TransformationConfig,
    ):
        self.type = type
        self.imu_config = imu_config
        self.transformation = transformation

    def __str__(self):
        return \
            f"HardwareConfig(\n"\
            f"\ttype={self.type}\n" \
            f"\timu_config={self.imu_config}\n" \
            f"\ttransformation={self.transformation}\n" \
            f")"

GyroSpecification = namedtuple("GyroSpecification", ['noise', 'offset'])
AccelSpecification = namedtuple("AccelSpecification",
                                ['noise', 'offset', 'scale'])
    

@dataclass
class VisualOdometryConfig:
    def __init__(
        self,
        type: str,
        estimator: str = "2d2d",
        camera_id: str = "left",
        feature_detector: str = "sift",
        feature_matcher: str = "bf",
        depth_estimator: str = None,
        use_advanced_detector: bool = False,
        params: Dict[str, Union[int, float]] = {},
    ):
        self.type = type
        self.estimator = estimator
        self.camera_id = camera_id
        self.feature_detector = feature_detector
        self.feature_matcher = feature_matcher
        self.depth_estimator = depth_estimator
        self.use_advanced_detector = use_advanced_detector
        self.params = params

        logging.debug(f"VisualOdometryConfig initialized with: {self}")

    @classmethod
    def from_json(cls, json_data: dict):
        """Load VO config from json data.

        Args:
            json_data (dict): JSON data containing VO configuration.
        """
        if json_data is None:
            return None

        type = json_data.get("type", "monocular")
        estimator = json_data.get("estimator", "2d2d")
        camera_id = json_data.get("camera_id", "left")
        feature_detector = json_data.get("feature_detector", "sift")
        feature_matcher = json_data.get("feature_matcher", "bf")
        depth_estimator = json_data.get("depth_estimator", None)
        use_advanced_detector = json_data.get("use_advanced_detector", False)
        params = json_data.get("params", {})

        return cls(
            type=type,
            estimator=estimator,
            camera_id=camera_id,
            feature_detector=feature_detector,
            feature_matcher=feature_matcher,
            depth_estimator=depth_estimator,
            use_advanced_detector=use_advanced_detector,
            params=params
        )
    def __str__(self):
        return \
            f"VisualOdometryConfig(\n"\
            f"\ttype={self.type}\n" \
            f"\testimator={self.estimator}\n" \
            f"\tfeature_detector={self.feature_detector}\n" \
            f"\tfeature_matcher={self.feature_matcher}\n" \
            f"\tcamera_id={self.camera_id}\n" \
            f"\tdepth_estimator={self.depth_estimator}\n" \
            f"\tuse_advanced_detector={self.use_advanced_detector}\n" \
            f"\tparams={self.params}\n" \
            f")"

class Config:

    def __init__(
        self,
        config_filepath,
    ):
        """
      - config_filepath: yaml file path that stores system's configurations
    """
        self.config_filepath = config_filepath

        config = None
        with open(config_filepath, "r") as f:
            config = yaml.safe_load(f)
            f.close()

        self.filter = FilterConfig(**config["filter"])
        self.vo_config = VisualOdometryConfig(**config['visual_odometry'])