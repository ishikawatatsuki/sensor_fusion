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
class DroneHardwareConfig:
    moment_of_inertia: float  # kg.m2
    arm_length: float  # in meter
    inertia_of_rotor: float  # kg.m2
    thrust_coefficient: np.ndarray  # Ns2
    moment_coefficient: np.ndarray  # Nms2
    mass_of_drone: float  # kg
    aerodynamic_thrust_drag_coefficient: float  # Ns/m
    aerodynamic_moment_drag_coefficient: float  # Nm.s

    M: np.ndarray

    def __init__(self,
                 moment_of_inertia_x: float = 7.5e-3,
                 moment_of_inertia_y: float = 7.5e-3,
                 moment_of_inertia_z: float = 1.3e-2,
                 arm_length: float = 0.25,
                 inertia_of_rotor: float = 6e-5,
                 thrust_coefficient: float = 3.90153837e-6,
                 moment_coefficient: float = 7.5e-7,
                 mass_of_drone: float = 1.075,
                 aerodynamic_thrust_drag_coefficient: float = 0.1,
                 aerodynamic_moment_drag_coefficient: float = 0.1):
        self.Ix = moment_of_inertia_x
        self.Iy = moment_of_inertia_y
        self.Iz = moment_of_inertia_z
        self.moment_of_inertia = np.eye(3) * np.array(
            [moment_of_inertia_x, moment_of_inertia_y, moment_of_inertia_z])
        self.arm_length = arm_length
        self.inertia_of_rotor = inertia_of_rotor

        rpm_to_rad = (60 / (2 * np.pi)) ** 2
        if False:
            self.thrust_coefficient = np.repeat(thrust_coefficient, 4) * rpm_to_rad # to rad/s
            self.moment_coefficient = np.array([-1, 1, -1, 1]) * np.repeat(moment_coefficient, 4) * rpm_to_rad # to rad/s
        else:
            self.thrust_coefficient = np.array([2.6890e-7, 2.8190e-7, 2.7263e-7, 2.7741e-7]) * rpm_to_rad # to rad/s
            self.moment_coefficient = np.array([-5.6343e-9, 4.7180e-9, -5.7012e-9, 4.8260e-9]) * rpm_to_rad # to rad/s
            # self.moment_coefficient = np.array([-4.7180e-9, 5.6343e-9, -4.8260e-9, 5.7012e-9]) * rpm_to_rad # to rad/s

        self.mass_of_drone = mass_of_drone
        self.aerodynamic_thrust_drag_coefficient = np.eye(3) * np.repeat(
            aerodynamic_thrust_drag_coefficient, 3)
        self.aerodynamic_moment_drag_coefficient = np.eye(3) * np.repeat(
            aerodynamic_moment_drag_coefficient, 3)

        M = np.array([
            [1., 1., 1., 1.],
            [-1., 1., 1., -1.],
            [1., -1., 1., -1.],
            [1., 1., -1., -1.]
        ])
        M[0] *= self.thrust_coefficient
        # M[0] = self.thrust_coefficient
        M[1] *= self.thrust_coefficient*np.repeat(arm_length, 4)
        M[2] *= self.thrust_coefficient*np.repeat(arm_length, 4)
        M[3] *= self.moment_coefficient
        self.M = M # Control allocation matrix
        self.Jr = np.array([-9.8316e-6, 8.5648e-6, -9.7166e-6, 9.8131e-6]) # Rotor's moment of inertia


    def __str__(self):
        return \
            f"Arm length of drone: {self.arm_length}m\n" \
            f"The weight of drone: {self.mass_of_drone}kg\n" \
            f"Rotor's inertia of drone: {self.inertia_of_rotor}\n" \
            f"Thrust coefficient of drone: {self.thrust_coefficient}\n" \
            f"Moment coefficient of drone: {self.moment_coefficient}\n" \
            f"Moment of inertia of drone: {self.moment_of_inertia}\n" \
            f"Aerodynamic thrust drag coefficient of drone: {self.aerodynamic_thrust_drag_coefficient}\n" \
            f"Aerodynamic moment drag coefficient  of drone: {self.aerodynamic_moment_drag_coefficient}\n" \
            f"Control allocation matrix M: {self.M}\n"


@dataclass
class HardwareConfig:
    type: str
    imu_config: ImuConfig
    transformation: TransformationConfig
    drone_hardware_config: DroneHardwareConfig
    
    def __init__(
            self,
            type: str,
            imu_config: ImuConfig,
            transformation: TransformationConfig,
            drone_hardware_config: DroneHardwareConfig = None
    ):
        self.type = type
        self.imu_config = imu_config
        self.transformation = transformation
        self.drone_hardware_config = drone_hardware_config

    def __str__(self):
        return \
            f"HardwareConfig(\n"\
            f"\ttype={self.type}\n" \
            f"\timu_config={self.imu_config}\n" \
            f"\ttransformation={self.transformation}\n" \
            f"\tdrone_hardware_config={self.drone_hardware_config}\n" \
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
        object_detector: str = None,
        use_advanced_detector: bool = False,
        export_vo_data: bool = False,
        export_vo_data_path: str = "./data/",
        params: Dict[str, Union[int, float]] = {},
        dynamic_objects: List[str] = [],
    ):
        self.type = type
        self.estimator = estimator
        self.camera_id = camera_id
        self.feature_detector = feature_detector
        self.feature_matcher = feature_matcher
        self.depth_estimator = depth_estimator
        self.object_detector = object_detector
        self.use_advanced_detector = use_advanced_detector
        self.params = params
        self.export_vo_data = export_vo_data
        self.export_vo_data_path = export_vo_data_path
        self.dynamic_objects = dynamic_objects

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
        object_detector = json_data.get("object_detector", None)
        use_advanced_detector = json_data.get("use_advanced_detector", False)
        export_vo_data = json_data.get("export_vo_data", False)
        export_vo_data_path = json_data.get("export_vo_data_path", "./data/")
        params = json_data.get("params", {})
        dynamic_objects = json_data.get("dynamic_objects", [])

        return cls(
            type=type,
            estimator=estimator,
            camera_id=camera_id,
            feature_detector=feature_detector,
            feature_matcher=feature_matcher,
            depth_estimator=depth_estimator,
            object_detector=object_detector,
            use_advanced_detector=use_advanced_detector,
            export_vo_data=export_vo_data,
            export_vo_data_path=export_vo_data_path,
            params=params,
            dynamic_objects=dynamic_objects
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
            f"\tobject_detector={self.object_detector}\n" \
            f"\tuse_advanced_detector={self.use_advanced_detector}\n" \
            f"\texport_vo_data={self.export_vo_data}\n" \
            f"\texport_vo_data_path={self.export_vo_data_path}\n" \
            f"\tparams={self.params}\n" \
            f"\tdynamic_objects={self.dynamic_objects}\n" \
            f")"

    def to_dict(self):
        return {
            "type": self.type,
            "estimator": self.estimator,
            "camera_id": self.camera_id,
            "feature_detector": self.feature_detector,
            "feature_matcher": self.feature_matcher,
            "depth_estimator": self.depth_estimator,
            "object_detector": self.object_detector,
            "use_advanced_detector": self.use_advanced_detector,
            "export_vo_data": self.export_vo_data,
            "export_vo_data_path": self.export_vo_data_path,
            "params": self.params,
            "dynamic_objects": self.dynamic_objects
        }


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