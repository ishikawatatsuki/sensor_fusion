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


from .datatypes import SensorConfig, VisualizationDataType, State, Pose, SensorType, FusionData

@dataclass
class FilterNoise:
    type: str
    params: dict
    
    @classmethod
    def from_json(cls, json_data: dict):
        _type = json_data.get("type", "default")
        _params = json_data.get("params", {})
        return cls(type=_type, params=_params)
    
    def __str__(self):
        return f"FilterNoise(type={self.type}, params={self.params})"

@dataclass
class FilterConfig:
    type: str
    dimension: int
    motion_model: str
    innovation_masking: bool
    params: dict
    is_imu_preintegrated: bool
    compensate_gravity: bool
    use_imu_preprocessing: bool
    sensors: dict
    noise: FilterNoise

    def __init__(
        self,
        type: str = "ekf",
        dimension: int = 2,
        motion_model: str = "velocity",
        noise_type: str = "default",
        innovation_masking: bool = False,
        params: dict = {},
        is_imu_preintegrated: bool = False,
        compensate_gravity: bool = False,
        use_imu_preprocessing: bool = False,
        sensors: dict = {},
        noise: dict = {},
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
        self.sensors = sensors
        self.noise = FilterNoise.from_json(noise)

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
            f"\tsensors={self.sensors}\n" \
            f"\tnoise={self.noise}\n" \
            f")"

    def set_sensor_fields(self, dataset_type: str):
        get_sensor_type_fn = SensorType.get_sensor_from_str_func(dataset_type)
        sensors = {}
        fusion_data_fields = FusionData.get_enum_name_list()
        for sensor_type_str, values in self.sensors.items():
            sensor_type = get_sensor_type_fn(sensor_type_str)
            fields = values.get("fields", [])
            if sensor_type is not None:
                sensors[sensor_type] = [FusionData.get_type(field) for field in fields if field in fusion_data_fields]

        self.sensors = sensors

    def to_dict(self):
        return {
            "type": self.type,
            "dimension": self.dimension,
            "motion_model": self.motion_model,
            "innovation_masking": self.innovation_masking,
            "params": self.params,
            "is_imu_preintegrated": self.is_imu_preintegrated,
            "compensate_gravity": self.compensate_gravity,
            "use_imu_preprocessing": self.use_imu_preprocessing,
            "sensors": {sensor_type.name: [field.name for field in fields] for sensor_type, fields in self.sensors.items()},
            "noise": {
                "type": self.noise.type,
                "params": self.noise.params
            }
        }

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
    T_leica_to_inertial: np.ndarray

    T_angle_compensation: np.ndarray

    def __init__(self,
                    T_from_cam_to_imu: np.ndarray = None,
                    T_from_imu_to_cam: np.ndarray = None,
                    T_imu_body_to_inertial: np.ndarray = None,
                    T_leica_to_inertial: np.ndarray = None,
                    T_imu_to_virtual_imu: Dict[SensorType, np.ndarray] = {},
                    T_angle_compensation: np.ndarray = np.eye(4)):
            """Initialize transformation configuration.
            Args:
                T_from_cam_to_imu (np.ndarray): Transformation matrix from camera to IMU.
                T_from_imu_to_cam (np.ndarray): Transformation matrix from IMU to camera.
                T_imu_body_to_inertial (np.ndarray): Transformation matrix from IMU body frame to inertial frame.
                T_leica_to_inertial (np.ndarray, optional): Transformation matrix from IMU body frame to Leica frame.
                T_imu_to_virtual_imu (Dict[SensorType, np.ndarray], optional): Transformation matrices for virtual IMUs.
            """
            self.T_from_cam_to_imu = T_from_cam_to_imu if T_from_cam_to_imu is not None else np.eye(4)
            self.T_from_imu_to_cam = T_from_imu_to_cam if T_from_imu_to_cam is not None else np.eye(4)
            self.T_imu_body_to_inertial = T_imu_body_to_inertial if T_imu_body_to_inertial is not None else np.eye(4)
            
            self.T_leica_to_inertial = T_leica_to_inertial
            self.T_imu_to_virtual_imu = T_imu_to_virtual_imu
            self.T_angle_compensation = T_angle_compensation
            

    @classmethod
    def from_kitti_config(cls, T_from_imu_to_cam: np.ndarray, T_angle_compensation: np.ndarray = np.eye(4)):
        """Get transformation matrix for the sensor hardware configuration.
        Args:
            dataset_type (str): Type of the dataset (e.g., "kitti", "paldiski", "vlaardingan").
            calibration_root_path (str): Path to the folder that stores calibration files.
        """
        T_from_cam_to_imu = np.linalg.inv(T_from_imu_to_cam)
        
        return cls(
            T_from_cam_to_imu=T_from_cam_to_imu,
            T_from_imu_to_cam=T_from_imu_to_cam,
            T_imu_body_to_inertial=np.eye(4),
            T_angle_compensation=T_angle_compensation
        )
    
    @classmethod
    def from_euroc_config(
        cls, 
        T_from_cam_to_imu: np.ndarray,
        T_from_imu_to_cam: np.ndarray, 
        T_from_imu_to_inertial: np.ndarray,
        T_leica_to_inertial: np.ndarray
    ):
        T_from_cam_to_imu = np.linalg.inv(T_from_imu_to_cam)
        
        return cls(
            T_from_cam_to_imu=T_from_cam_to_imu,
            T_from_imu_to_cam=T_from_imu_to_cam,
            T_imu_body_to_inertial=T_from_imu_to_inertial,
            T_leica_to_inertial=T_leica_to_inertial
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
        export_vo_data: bool = False,
        export_vo_data_path: str = "./data/",
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
        self.export_vo_data = export_vo_data
        self.export_vo_data_path = export_vo_data_path

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
        export_vo_data = json_data.get("export_vo_data", False)
        export_vo_data_path = json_data.get("export_vo_data_path", "./data/")
        params = json_data.get("params", {})

        return cls(
            type=type,
            estimator=estimator,
            camera_id=camera_id,
            feature_detector=feature_detector,
            feature_matcher=feature_matcher,
            depth_estimator=depth_estimator,
            use_advanced_detector=use_advanced_detector,
            export_vo_data=export_vo_data,
            export_vo_data_path=export_vo_data_path,
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
            f"\texport_vo_data={self.export_vo_data}\n" \
            f"\texport_vo_data_path={self.export_vo_data_path}\n" \
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


if __name__ == "__main__":
    config_file = "configs/kitti_config.yaml"
    config = Config(config_file)
    
    config.filter.set_sensor_fields("kitti")
    print(config.filter)