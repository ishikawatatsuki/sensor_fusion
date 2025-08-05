import logging
import cv2
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
    vo_velocity_only_update_when_failure: bool
    params: dict
    estimation_publishing_interval: float
    is_imu_preintegrated: bool
    multi_threading: bool
    compensate_gravity: bool
    use_imu_preprocessing: bool
    imu_dead_reckoning: bool
    use_vo_relative_pose: bool

    def __init__(
        self,
        type: str = "ckf",
        dimension: int = 2,
        motion_model: str = "velocity",
        noise_type: str = "default",
        innovation_masking: bool = False,
        vo_velocity_only_update_when_failure: bool = False,
        params: dict = {},
        estimation_publishing_interval: float = 1.0,
        is_imu_preintegrated: bool = False,
        multi_threading: bool = False,
        compensate_gravity: bool = False,
        use_imu_preprocessing: bool = False,
        imu_dead_reckoning: bool = False,
        use_vo_relative_pose: bool = False,
        ):
        self.type = type
        self.dimension = dimension
        self.motion_model = motion_model
        self.noise_type = noise_type
        self.innovation_masking = innovation_masking
        self.vo_velocity_only_update_when_failure = vo_velocity_only_update_when_failure
        self.params = params
        self.estimation_publishing_interval = estimation_publishing_interval
        self.is_imu_preintegrated = is_imu_preintegrated
        self.multi_threading = multi_threading
        self.compensate_gravity = compensate_gravity
        self.use_imu_preprocessing = use_imu_preprocessing
        self.imu_dead_reckoning = imu_dead_reckoning
        self.use_vo_relative_pose = use_vo_relative_pose

    def __str__(self):
        return \
            f"FilterConfig(\n"\
            f"\ttype={self.type}\n" \
            f"\tdimension={self.dimension}\n" \
            f"\tmotion_model={self.motion_model}\n" \
            f"\tnoise_type={self.noise_type}\n" \
            f"\tinnovation_masking={self.innovation_masking}\n" \
            f"\tvo_velocity_only_update_when_failure={self.vo_velocity_only_update_when_failure}\n" \
            f"\tparams={self.params}\n" \
            f"\testimation_publishing_interval={self.estimation_publishing_interval}\n" \
            f"\tis_imu_preintegrated={self.is_imu_preintegrated}\n" \
            f"\tmulti_threading={self.multi_threading}\n" \
            f"\tcompensate_gravity={self.compensate_gravity}\n" \
            f"\tuse_imu_preprocessing={self.use_imu_preprocessing}\n" \
            f"\timu_dead_reckoning={self.imu_dead_reckoning}\n" \
            f"\tuse_vo_relative_pose={self.use_vo_relative_pose}\n" \
            f")"


@dataclass
class BeaconInfo:
    position: List[float]
    lla: List[float]
    timestamp: int

@dataclass
class BeaconReference:
    rssi_ref: float
    rssi_max: float
    rssi_min: float
    n: int
    emission_interval: int

@dataclass
class BeaconConfig:
    device_info: Dict[str, BeaconInfo]
    configs: BeaconReference
    vehicle_state_df: pd.DataFrame

    @classmethod
    def from_data(
        cls,
        json_data: dict,
        vehicle_state_df: pd.DataFrame = None
    ):
        """Load beacon config from json data.

        Args:
            json_data (dict): JSON data containing beacon configuration.
        """
        if json_data is None:
            return None

        device_info = {
            key: BeaconInfo(**value) for key, value in json_data["device_info"].items()
        }
        configs = BeaconReference(**json_data["configs"])
        
        return cls(
            device_info=device_info,
            configs=configs,
            vehicle_state_df=vehicle_state_df
        )

    def __str__(self):
        return \
            f"BeaconConfig(\n"\
            f"\device_info={self.device_info}\n" \
            f"\configs={self.configs}\n" \
            f"\tvehicle_state_df={self.vehicle_state_df}\n" \
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
    T_imu_body_to_leica: np.ndarray = None

    def __init__(self,
                    T_from_cam_to_imu: np.ndarray = None,
                    T_from_imu_to_cam: np.ndarray = None,
                    T_imu_body_to_inertial: np.ndarray = None,
                    T_imu_body_to_leica: np.ndarray = None,
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
            self.T_imu_body_to_leica = T_imu_body_to_leica
            
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
            T_imu_body_to_leica=None,
        )

    @classmethod
    def from_uav_config(cls, yaml_config: dict):
        r = np.eye(4)
        return cls(
            T_from_cam_to_imu=r,
            T_from_imu_to_cam=r,
            T_imu_body_to_inertial=r,
            T_imu_to_virtual_imu={},
            T_imu_body_to_leica=None,
        )

        
    def __str__(self):
        return \
            f"TransformationConfig(\n" \
            f"\tT_from_cam_to_imu={self.T_from_cam_to_imu}\n" \
            f"\tT_from_imu_to_cam={self.T_from_imu_to_cam}\n" \
            f"\tT_imu_body_to_inertial={self.T_imu_body_to_inertial}\n" \
            f")"

@dataclass
class GeofencingConfig:
    fencing_lines: Dict[str, List[LineString]]

    def __init__(self, fencing_lines: Dict[str, List[LineString]]):
        self.fencing_lines = fencing_lines

    @staticmethod
    def get_geofencing(geo: pd.DataFrame) -> Dict[str, List[LineString]]:
        """Get geofencing config from dataframe.
        Args:
            df (pd.DataFrame): Dataframe containing geofencing information.
        """
        geofencing_lines = {}
        geo_ids = geo['id'].unique().tolist()
        for geo_id in geo_ids:
            geo_coord = geo[geo['id'] == geo_id][['X', 'Y']].values
            next_coord = np.roll(geo_coord, -1, axis=0)[: geo_coord.shape[0]-1]
            geo_coord = geo_coord[:geo_coord.shape[0]-1]
            lines = []
            for (coord1, coord2) in zip(geo_coord, next_coord):
                x1 = (coord1[0], coord1[1])
                x2 = (coord2[0], coord2[1])
                line = LineString([x1, x2])
                lines.append(line)

            geofencing_lines[geo_id] = lines
        return geofencing_lines
    
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame):
        """Create geofencing config from dataframe."""
        if df is None:
            return cls(fencing_lines={})

        fencing_lines = cls.get_geofencing(df)
        return cls(fencing_lines=fencing_lines)
    
    def __str__(self):
        return \
            f"GeofencingConfig(\n"\
            f"\tfencing_lines={self.fencing_lines}\n" \
            f")"

@dataclass
class DroneHardwareConfig:
    moment_of_inertia: float  # kg.m2
    arm_length: float  # in meter
    inertia_of_rotor: float  # kg.m2
    thrust_coefficient: float  # Ns2
    moment_coefficient: float  # Nms2
    mass_of_drone: float  # kg
    aerodynamic_thrust_drag_coefficient: float  # Ns/m
    aerodynamic_moment_drag_coefficient: float  # Nm.s

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
        self.thrust_coefficient = thrust_coefficient
        self.moment_coefficient = moment_coefficient
        self.mass_of_drone = mass_of_drone
        self.aerodynamic_thrust_drag_coefficient = np.eye(3) * np.repeat(
            aerodynamic_thrust_drag_coefficient, 3)
        self.aerodynamic_moment_drag_coefficient = np.eye(3) * np.repeat(
            aerodynamic_moment_drag_coefficient, 3)

    def __str__(self):
        return \
          f"Arm length of drone: {self.arm_length}m\n" \
          f"The weight of drone: {self.mass_of_drone}kg\n" \
          f"Rotor's inertia of drone: {self.inertia_of_rotor}\n" \
          f"Thrust coefficient of drone: {self.thrust_coefficient}\n" \
          f"Moment coefficient of drone: {self.moment_coefficient}\n" \
          f"Moment of inertia of drone: {self.moment_of_inertia}\n" \
          f"Aerodynamic thrust drag coefficient of drone: {self.aerodynamic_thrust_drag_coefficient}\n" \
          f"Aerodynamic moment drag coefficient  of drone: {self.aerodynamic_moment_drag_coefficient}"


@dataclass
class HardwareConfig:
    type: str
    imu_config: ImuConfig
    beacon_config: BeaconConfig
    transformation: TransformationConfig
    geofencing_config: GeofencingConfig
    drone_hardware_config: DroneHardwareConfig
    
    def __init__(
            self,
            type: str,
            imu_config: ImuConfig,
            beacon_config: BeaconConfig,
            transformation: TransformationConfig,
            geofencing_config: GeofencingConfig,
            drone_hardware_config: DroneHardwareConfig = None
    ):
        self.type = type
        self.imu_config = imu_config
        self.beacon_config = beacon_config
        self.transformation = transformation
        self.geofencing_config = geofencing_config
        self.drone_hardware_config = drone_hardware_config

    def __str__(self):
        return \
            f"HardwareConfig(\n"\
            f"\ttype={self.type}\n" \
            f"\timu_config={self.imu_config}\n" \
            f"\tbeacon_config={self.beacon_config}\n" \
            f"\ttransformation={self.transformation}\n" \
            f"\tgeofencing_config={self.geofencing_config}\n" \
            f"\drone_hardware_config={self.drone_hardware_config}\n" \
            f")"

GyroSpecification = namedtuple("GyroSpecification", ['noise', 'offset'])
AccelSpecification = namedtuple("AccelSpecification",
                                ['noise', 'offset', 'scale'])
    

@dataclass
class VO_Config:
    def __init__(
        self,
        type: str,
        estimator: str,
        camera_id: str,
        feature_detector: str = "sift",
        feature_matcher: str = "bf",
        depth_estimator: str = None,
        use_advanced_detector: bool = False,
        solve_pose_max_iterations: int = 5,
        params: Dict[str, Union[int, float]] = {},
    ):
        self.type = type
        self.estimator = estimator
        self.camera_id = camera_id
        self.feature_detector = feature_detector
        self.feature_matcher = feature_matcher
        self.depth_estimator = depth_estimator
        self.use_advanced_detector = use_advanced_detector
        self.solve_pose_max_iterations = solve_pose_max_iterations
        self.params = params

        logging.debug(f"VO_Config initialized with: {self}")

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
        solve_pose_max_iterations = json_data.get("solve_pose_max_iterations", 10)
        params = json_data.get("params", {})

        return cls(
            type=type,
            estimator=estimator,
            camera_id=camera_id,
            feature_detector=feature_detector,
            feature_matcher=feature_matcher,
            depth_estimator=depth_estimator,
            use_advanced_detector=use_advanced_detector,
            solve_pose_max_iterations=solve_pose_max_iterations,
            params=params
        )
    def __str__(self):
        return \
            f"VO_Config(\n"\
            f"\ttype={self.type}\n" \
            f"\testimator={self.estimator}\n" \
            f"\tfeature_detector={self.feature_detector}\n" \
            f"\tfeature_matcher={self.feature_matcher}\n" \
            f"\tcamera_id={self.camera_id}\n" \
            f"\tdepth_estimator={self.depth_estimator}\n" \
            f"\tuse_advanced_detector={self.use_advanced_detector}\n" \
            f"\tsolve_pose_max_iterations={self.solve_pose_max_iterations}\n" \
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

        kwargs = config['drone_config'] if config.get(
            'drone_config') is not None else {}

        self.hardware_config = DroneHardwareConfig(**kwargs)

        self.vo_config = VO_Config(**config['visual_odometry'])