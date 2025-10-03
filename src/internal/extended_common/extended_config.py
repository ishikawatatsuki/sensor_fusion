import os
import yaml
import logging
import numpy as np
import pandas as pd
from collections import namedtuple
from dataclasses import dataclass
from typing import List, Union, Tuple
from scipy.spatial.transform import Rotation

from ...utils.geometric_transformer.base_geometric_transformer import BaseGeometryTransformer
from ...common.constants import KITTI_SEQUENCE_TO_DATE, KITTI_ANGLE_COMPENSATION_CAMERA_TO_INERTIAL, EUROC_SEQUENCE_MAPS
from ...common.datatypes import SensorConfig, VisualizationDataType, SensorType, DatasetType, Pose
from ...common.config import FilterConfig, HardwareConfig, ImuConfig, TransformationConfig, VisualOdometryConfig

@dataclass
class GeneralConfig:
    log_level: str
    log_sensor_data: float
    save_sensor_data: bool
    save_estimation: int
    save_output_debug_frames: bool
    sensor_data_output_filepath: str
    sensor_data_save_path: str

    def __init__(
            self,
            log_level: str = 'debug',
            log_sensor_data: bool = True,
            save_sensor_data: bool = False,
            save_estimation: bool = False,
            save_output_debug_frames: bool = False,
            sensor_data_output_filepath: str = './_debugging/sensor_output.txt',
            sensor_data_save_path: str = './outputs/sensor_data'
    ):
        self.log_level = log_level
        self.log_sensor_data = log_sensor_data
        self.save_sensor_data = save_sensor_data
        self.save_estimation = save_estimation
        self.save_output_debug_frames = save_output_debug_frames
        self.sensor_data_output_filepath = sensor_data_output_filepath
        self.sensor_data_save_path = sensor_data_save_path
    
    def __str__(self):
        return \
            f"GeneralConfig(\n"\
            f"\tlog_level={self.log_level}\n" \
            f"\tlog_sensor_data={self.log_sensor_data}\n" \
            f"\tsave_sensor_data={self.save_sensor_data}\n" \
            f"\tsave_estimation={self.save_estimation}\n" \
            f"\tsave_output_debug_frames={self.save_output_debug_frames}\n" \
            f"\tsensor_data_output_filepath={self.sensor_data_output_filepath}\n" \
            f"\tsensor_data_save_path={self.sensor_data_save_path})"


@dataclass
class DatasetConfig:
    type: str
    mode: float
    root_path: int
    variant: bool
    sensors: List[SensorConfig]
    imu_config_path: str
    sensor_config_path: str

    run_visual_odometry: bool

    def __init__(self,
                 type: str = 'kitti',
                 mode: str = 'stream',
                 root_path: str = '../data/KITTI',
                 variant: str = '0033',
                 sensors: List[SensorConfig] = [],
                 imu_config_path: str = None,
                 sensor_config_path: str = None
                 ):
        self.type = type
        self.mode = mode
        self.root_path = root_path
        self.variant = variant
        self.sensors = sensors
        self.imu_config_path = imu_config_path
        self.sensor_config_path = sensor_config_path

        self.run_visual_odometry = False

    def __str__(self):
        return \
            f"DatasetConfig(\n"\
            f"\ttype={self.type}\n" \
            f"\tmode={self.mode}\n" \
            f"\troot_path={self.root_path}\n" \
            f"\tvariant={self.variant}\n" \
            f"\tsensors={self.sensors}\n" \
            f"\timu_config_path={self.imu_config_path}\n" \
            f"\tsensor_config_path={self.sensor_config_path}\n" \
            f"\t)"
    
    def to_dict(self):
        return {
            "type": self.type,
            "mode": self.mode,
            "root_path": self.root_path,
            "variant": self.variant,
            "sensors": [sensor.__dict__ for sensor in self.sensors],
            "imu_config_path": self.imu_config_path,
            "sensor_config_path": self.sensor_config_path
        }

    def set_run_visual_odometry(self):
        self.run_visual_odometry = True

    @property
    def should_run_visual_odometry(self) -> bool:
        return self.run_visual_odometry

GeometricLimit = namedtuple('GeometricLimit', ['min', 'max'])


@dataclass
class VisualizationConfig:
    realtime: bool
    output_filepath: str
    save_frames: bool
    save_trajectory: bool
    show_vo_trajectory: bool
    show_angle_estimation: bool
    show_end_result: bool
    show_vio_frame: bool
    show_particles: bool
    set_lim_in_plot: bool
    show_innovation_history: bool
    limits: GeometricLimit
    fields: List[VisualizationDataType]

    def __init__(
        self,
        realtime: bool = False,
        output_filepath: str = './',
        save_frames: bool = False,
        save_trajectory: bool = False,
        show_vo_trajectory: bool = False,
        show_angle_estimation: bool = False,
        show_end_result: bool = False,
        show_vio_frame: bool = False,
        show_particles: bool = False,
        set_lim_in_plot: bool = False,
        show_innovation_history: bool = False,
        limits: Union[GeometricLimit | None] = None,
        fields: List[VisualizationDataType] = []
    ):
        self.realtime = realtime
        self.output_filepath = output_filepath
        self.save_frames = save_frames
        self.save_trajectory = save_trajectory
        self.show_vo_trajectory = show_vo_trajectory
        self.show_angle_estimation = show_angle_estimation
        self.show_end_result = show_end_result
        self.show_vio_frame = show_vio_frame
        self.show_particles = show_particles
        self.set_lim_in_plot = set_lim_in_plot
        self.show_innovation_history = show_innovation_history
        self.limits = limits
        self.fields = fields


@dataclass
class ReportConfig:
    export_error: bool
    error_output_root_path: str
    pose_result_dir: str
    location_only: bool

    def __init__(
        self,
        export_error: bool = False,
        error_output_root_path: str = '../outputs/KITTI',
        pose_result_dir: str = '',
        location_only: bool = False,
    ):
        self.export_error = export_error
        self.error_output_root_path = error_output_root_path
        self.pose_result_dir = pose_result_dir
        self.location_only = location_only

    def __str__(self):
        return \
            f"ReportConfig(\n"\
            f"\texport_error={self.export_error}\n" \
            f"\terror_output_root_path={self.error_output_root_path}\n" \
            f"\tpose_result_dir={self.pose_result_dir}\n" \
            f"\tlocation_only={self.location_only})"


GyroSpecification = namedtuple("GyroSpecification", ['noise', 'offset'])
AccelSpecification = namedtuple("AccelSpecification",
                                ['noise', 'offset', 'scale'])


def get_geometric_limitations(dataset: str, variant: str) -> GeometricLimit:
    if dataset == "kitti" or dataset == "experiment":
        match (variant):
            case "0016":
                return GeometricLimit(min=[], max=[])
            case "0033":
                return GeometricLimit(min=[-100, -400, -30],
                                      max=[600, 200, 15])
            case _:
                return GeometricLimit(min=[0, 0, 0], max=[100, 100, 20])
    else:
        match (variant):
            case "log0001":
                return GeometricLimit(min=[-5, -20, -8], max=[15, 15, 10])
            case "log0002":
                return GeometricLimit(min=[], max=[])
            case _:
                return GeometricLimit(min=[0, 0, 0], max=[100, 100, 20])


class ExtendedConfig:

    def __init__(
        self,
        config_filepath,
    ):
        """
      - config_filepath: yaml file path that stores system's configurations
    """
        self.config_filepath = config_filepath

        self.parsed_config = None
        with open(config_filepath, "r") as f:
            self.parsed_config = yaml.safe_load(f)
            f.close()

        self.general = GeneralConfig(**self.parsed_config["general"])
        self.report = ReportConfig(**self.parsed_config["report"])
        self.dataset = DatasetConfig(**self.parsed_config["dataset"])
        
        sensors = []
        for sensor, value in self.dataset.sensors.items():
            if value.get("selected", False):
                sensors.append(
                    SensorConfig(
                        name=sensor,
                        dropout_ratio=value["dropout_ratio"],
                        window_size=value["window_size"],
                        args=value.get("args", {}),
                    )
                )
                if "vo" in sensor.lower():
                    self.dataset.set_run_visual_odometry()

        self.dataset.sensors = sensors
        self.filter = FilterConfig(**self.parsed_config["filter"])
        self.filter.set_sensor_fields(self.dataset.type)

        self.report = ReportConfig(**self.parsed_config["report"])
        self.visual_odometry = VisualOdometryConfig(**self.parsed_config['visual_odometry'])

        self.visualization = VisualizationConfig(**self.parsed_config['visualization'])
        fields = [
            VisualizationDataType.get_type(key)
            for (key, value) in self.visualization.fields.items()
            if value and VisualizationDataType.get_type(key) is not None
        ]
        self.visualization.fields = fields

        limits = get_geometric_limitations(self.dataset.type,
                                           self.dataset.variant)
        self.visualization.limits = limits
        
        self.hardware = self._get_sensor_hardware_config()

        logging.info(f"Dataset sensors: {self.dataset.sensors}")



    def _get_sensor_hardware_config(self) -> HardwareConfig:
        # NOTE: This is only internal usage.

        def _get_transformation_config() -> TransformationConfig:
            """Get transformation matrix for the sensor hardware configuration."""

            def _get_kitti_transformation_config():
                def _get_rigid_transformation(calib_path: str) -> Tuple[np.ndarray, np.ndarray]:
                    logging.debug(f"Reading calibration file: {calib_path}")
                    with open(calib_path, 'r') as f:
                        calib = f.readlines()
                    R = np.array([float(x) for x in calib[1].strip().split(' ')[1:]]).reshape((3, 3))
                    t = np.array([float(x) for x in calib[2].strip().split(' ')[1:]])[:, None]
                    T = np.vstack((np.hstack((R, t)), np.array([0, 0, 0, 1])))
                    return T

                kitti_date = KITTI_SEQUENCE_TO_DATE.get(self.dataset.variant, None)
                root_calibration_path = os.path.join(self.dataset.root_path, kitti_date)
                T_calib_velo_to_cam = _get_rigid_transformation(os.path.join(root_calibration_path, "calib_velo_to_cam.txt"))
                T_calib_imu_to_velo = _get_rigid_transformation(os.path.join(root_calibration_path, "calib_imu_to_velo.txt"))

                return T_calib_velo_to_cam, T_calib_imu_to_velo
            
            def _get_euroc_transformation_config():
                def _get_calibration_data(calib_path: str) -> Pose:
                    with open(calib_path, 'r') as f:
                        calib = yaml.safe_load(f)

                    data = np.array(calib["T_BS"]["data"]).reshape(4, 4)
                    return Pose(R=data[:3, :3], t=data[:3, 3])

                variant = EUROC_SEQUENCE_MAPS.get(self.dataset.variant, "MH_01_easy")
                imu_calibration_path = os.path.join(self.dataset.root_path, variant, 'imu0/sensor.yaml')
                leica_calibration_path = os.path.join(self.dataset.root_path, variant, 'leica0/sensor.yaml')
                camera_calibration_path = os.path.join(self.dataset.root_path, variant, 'cam0/sensor.yaml')
                T_calib_imu_to_inertial = _get_calibration_data(imu_calibration_path)
                T_calib_leica_to_inertial = _get_calibration_data(leica_calibration_path)
                T_calib_cam_to_inertial = _get_calibration_data(camera_calibration_path)

                # Align the inertial frame such that x is forward, y is left, z is up
                # R = BaseGeometryTransformer.Ry(np.radians(90)) @ BaseGeometryTransformer.Rx(np.radians(180))
                # T_calib_imu_to_inertial = Pose(R=R, t=np.zeros(3)) * T_calib_imu_to_inertial

                return T_calib_imu_to_inertial, T_calib_leica_to_inertial, T_calib_cam_to_inertial

            dataset_type = DatasetType.get_type_from_str(self.dataset.type)
            if dataset_type == DatasetType.KITTI:
                angle_compensation = KITTI_ANGLE_COMPENSATION_CAMERA_TO_INERTIAL.get(self.dataset.variant, 0.0)
                T_angle_compensation = np.vstack(
                    (np.hstack((BaseGeometryTransformer.Rz(angle_compensation), np.zeros((3, 1)))), 
                     np.array([0, 0, 0, 1])))
                T_angle_compensation = np.linalg.inv(T_angle_compensation)
                
                T_calib_velo_to_cam, T_calib_imu_to_velo = _get_kitti_transformation_config()
                T_from_imu_to_cam = T_calib_imu_to_velo @ T_calib_velo_to_cam

                return TransformationConfig.from_kitti_config(
                    T_from_imu_to_cam=T_from_imu_to_cam,
                    T_angle_compensation=T_angle_compensation
                )
            elif dataset_type == DatasetType.EUROC:
                T_calib_imu_to_inertial, T_calib_leica_to_inertial, T_calib_cam_to_inertial = _get_euroc_transformation_config()

                T_from_cam_to_imu = T_calib_cam_to_inertial.matrix()
                T_from_imu_to_cam = T_calib_cam_to_inertial.inverse().matrix()
                T_from_imu_to_inertial = T_calib_imu_to_inertial.matrix()
                T_leica_to_inertial = T_calib_leica_to_inertial.matrix()

                return TransformationConfig.from_euroc_config(
                    T_from_cam_to_imu=T_from_cam_to_imu,
                    T_from_imu_to_cam=T_from_imu_to_cam,
                    T_from_imu_to_inertial=T_from_imu_to_inertial,
                    T_leica_to_inertial=T_leica_to_inertial
                )
            
            else:
                logging.error(f"Dataset type {self.dataset.type} is not supported for transformation configuration.")

        def _get_imu_config() -> ImuConfig:
            
            get_sensor_from_str = SensorType.get_sensor_from_str_func(d=self.dataset.type)
            for s in self.dataset.sensors:
                print(s.name)
            imu = [sensor for sensor in self.dataset.sensors if SensorType.is_imu_data(get_sensor_from_str(sensor.name))]
            frequency = imu[0].args.get("frequency", 100) if len(imu) != 0 else 100
            if frequency < 100:
                logging.warning(f"IMU frequency is set to {frequency} Hz. This is lower than the default value of 100 Hz.")
                frequency = 100
            
            default_value = 0.001
            if len(imu) == 0:
                gyroscope_noise_density = default_value
                accelerometer_noise_density = default_value
                gyroscope_random_walk = default_value
                accelerometer_random_walk = default_value
            else:
                gyroscope_noise_density = imu[0].args.get("gyroscope_noise_density", default_value)
                accelerometer_noise_density = imu[0].args.get("accelerometer_noise_density", default_value)
                gyroscope_random_walk = imu[0].args.get("gyroscope_random_walk", default_value)
                accelerometer_random_walk = imu[0].args.get("accelerometer_random_walk", default_value)

            if self.dataset.type == DatasetType.KITTI.name:
                return ImuConfig(
                    frequency=frequency,
                    target_frequency=frequency,
                    gyroscope_noise_density=gyroscope_noise_density,
                    accelerometer_noise_density=accelerometer_noise_density,
                    gyroscope_random_walk=gyroscope_random_walk,
                    accelerometer_random_walk=accelerometer_random_walk,
                )
            elif self.dataset.type == DatasetType.EUROC.name:
                return ImuConfig(
                    frequency=frequency,
                    target_frequency=frequency,
                    gyroscope_noise_density=gyroscope_noise_density,
                    accelerometer_noise_density=accelerometer_noise_density,
                    gyroscope_random_walk=gyroscope_random_walk,
                    accelerometer_random_walk=accelerometer_random_walk,
                )
            else:
                return ImuConfig(frequency=frequency)
        

        imu_config = _get_imu_config()
        transformation = _get_transformation_config()
        return HardwareConfig(
            type=self.dataset.type,
            imu_config=imu_config,
            transformation=transformation,
        )

def dump_config(filter_config: FilterConfig, dataset_config: DatasetConfig, vo_config: VisualOdometryConfig, output_filepath: str):
    """ Dump the configuration to a YAML file.

    Args:
        config (ExtendedConfig): Configuration object.
        output_filepath (str): Path to the output YAML file.
    """
    import yaml
    with open(output_filepath, 'w') as f:
        yaml.dump({"filter": filter_config.to_dict(), "dataset": dataset_config.to_dict(), "visual_odometry": vo_config.to_dict()}, f)
    logging.info(f"Configuration dumped to {output_filepath}")

if __name__ == "__main__":
    config_file = "configs/kitti_config.yaml"
    config = ExtendedConfig(config_file)
    
    print(config.filter)