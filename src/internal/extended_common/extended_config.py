import os
import re
import yaml
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from collections import namedtuple
from dataclasses import dataclass
from typing import Dict, List, Union, Tuple
from scipy.spatial.transform import Rotation

from ...common.datatypes import SensorConfig, VisualizationDataType, SensorType, DatasetType, Pose
from ...common.config import FilterConfig, HardwareConfig, DroneHardwareConfig, ImuConfig, BeaconConfig, TransformationConfig, GeofencingConfig
from ...utils import BaseGeometryTransformer

@dataclass
class GeneralConfig:
    log_level: str
    log_sensor_data: float
    save_estimation: int
    save_output_debug_frames: bool
    sensor_data_output_filepath: str

    def __init__(
            self,
            log_level: str = 'debug',
            log_sensor_data: bool = True,
            save_estimation: bool = False,
            save_output_debug_frames: bool = False,
            sensor_data_output_filepath: str = './_debugging/sensor_output.txt'
    ):
        self.log_level = log_level
        self.log_sensor_data = log_sensor_data
        self.save_estimation = save_estimation
        self.save_output_debug_frames = save_output_debug_frames
        self.sensor_data_output_filepath = sensor_data_output_filepath
    
    def __str__(self):
        return \
            f"GeneralConfig(\n"\
            f"\tlog_level={self.log_level}\n" \
            f"\tlog_sensor_data={self.log_sensor_data}\n" \
            f"\tsave_estimation={self.save_estimation}\n" \
            f"\tsave_output_debug_frames={self.save_output_debug_frames}\n" \
            f"\tsensor_data_output_filepath={self.sensor_data_output_filepath})"


@dataclass
class DatasetConfig:
    type: str
    mode: float
    root_path: int
    variant: bool
    sensors: List[SensorConfig]
    imu_config_path: str
    sensor_config_path: str

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
    kitti_pose_result_folder: str
    location_only: bool

    def __init__(
        self,
        export_error: bool = False,
        error_output_root_path: str = '../outputs/KITTI',
        kitti_pose_result_folder: str = '',
        location_only: bool = False,
    ):
        self.export_error = export_error
        self.error_output_root_path = error_output_root_path
        self.kitti_pose_result_folder = kitti_pose_result_folder
        self.location_only = location_only

    def __str__(self):
        return \
            f"ReportConfig(\n"\
            f"\texport_error={self.export_error}\n" \
            f"\terror_output_root_path={self.error_output_root_path}\n" \
            f"\tkitti_pose_result_folder={self.kitti_pose_result_folder}\n" \
            f"\tlocation_only={self.location_only})"


@dataclass
class VO_Mode:
    type: str
    save_traj: bool
    save_poses_kitti_format: bool
    save_output_debug_frames: bool
    output_directory: str
    log_level: str

    def __init__(
        self,
        type: str = 'stereo',
        save_traj: bool = True,
        save_poses_kitti_format: bool = True,
        save_output_debug_frames: bool = True,
        output_directory: str = 'results/seq_09_with_car_median',
        log_level: str = 'debug',
    ):

        self.type = type
        self.save_traj = save_traj
        self.save_poses_kitti_format = save_poses_kitti_format
        self.save_output_debug_frames = save_output_debug_frames
        self.output_directory = output_directory
        self.log_level = log_level


@dataclass
class VO_Stereo:
    algorithm: str
    speckle_window_size: int
    median_filter: int
    wsl_filter: bool
    dynamic_depth: bool

    def __init__(
        self,
        algorithm: str = 'SGBM',
        speckle_window_size: int = 100,
        median_filter: int = 3,
        wsl_filter: bool = False,
        dynamic_depth: bool = False,
    ):

        self.algorithm = algorithm
        self.speckle_window_size = speckle_window_size
        self.median_filter = median_filter
        self.wsl_filter = wsl_filter
        self.dynamic_depth = dynamic_depth


@dataclass
class VO_Detector:
    algorithm: str
    no_of_keypoints: str
    homography: bool
    error_threshold: int
    circular_matching: bool

    def __init__(
        self,
        algorithm: str = 'sift',
        no_of_keypoints: str = 'max',
        homography: bool = True,
        error_threshold: int = 30,
        circular_matching: bool = False,
    ):

        self.algorithm = algorithm
        self.no_of_keypoints = no_of_keypoints
        self.homography = homography
        self.error_threshold = error_threshold
        self.circular_matching = circular_matching


@dataclass
class VO_Matcher:
    algorithm: str
    ratio_test: float
    dynamic_ratio: bool

    def __init__(
        self,
        algorithm: str = 'bf',
        ratio_test: float = 0.45,
        dynamic_ratio: bool = True,
    ):

        self.algorithm = algorithm
        self.ratio_test = ratio_test
        self.dynamic_ratio = dynamic_ratio


@dataclass
class VO_MotionEstimation:
    algorithm: str
    depth_threshold: float
    depth_limit: bool
    x_correspondence_threshold: int
    invalidate_cars: bool

    def __init__(self,
                 algorithm: str = 'iterative',
                 depth_threshold: int = 125,
                 depth_limit: int = 500,
                 x_correspondence_threshold: int = 200,
                 invalidate_cars: bool = False):

        self.algorithm = algorithm
        self.depth_threshold = depth_threshold
        self.depth_limit = depth_limit
        self.x_correspondence_threshold = x_correspondence_threshold
        self.invalidate_cars = invalidate_cars


@dataclass
class VisualOdometryConfig:
    mode: VO_Mode
    stereo: VO_Stereo
    detector: VO_Detector
    matcher: VO_Matcher
    motion_estimation: VO_MotionEstimation

    def __init__(
        self,
        mode: VO_Mode = VO_Mode(),
        stereo: VO_Stereo = VO_Stereo(),
        detector: VO_Detector = VO_Detector(),
        matcher: VO_Matcher = VO_Matcher(),
        motion_estimation: VO_MotionEstimation = VO_MotionEstimation()):

        self.mode = mode
        self.stereo = stereo
        self.detector = detector
        self.matcher = matcher
        self.motion_estimation = motion_estimation


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

        self.filter = FilterConfig(**self.parsed_config["filter"])
        self.general = GeneralConfig(**self.parsed_config["general"])
        self.report = ReportConfig(**self.parsed_config["report"])
        self.dataset = DatasetConfig(**self.parsed_config["dataset"])
        sensors = [
            SensorConfig(
                name=sensor,
                dropout_ratio=value["dropout_ratio"],
                window_size=value["window_size"],
                args=value.get("args", {}),
            ) for sensor, value in self.dataset.sensors.items()
            if value.get("selected", False)
        ]
        self.dataset.sensors = sensors
        self.report = ReportConfig(**self.parsed_config["report"])
        self.vo_config = VisualOdometryConfig(
            mode=VO_Mode(**self.parsed_config["visual_odometry"]["mode"]),
            stereo=VO_Stereo(**self.parsed_config["visual_odometry"]["stereo"]),
            detector=VO_Detector(**self.parsed_config["visual_odometry"]["detector"]),
            matcher=VO_Matcher(**self.parsed_config["visual_odometry"]["matcher"]),
            motion_estimation=VO_MotionEstimation(
                **self.parsed_config["visual_odometry"]["motion_estimation"]))

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


    def _get_sensor_hardware_config(self) -> HardwareConfig:
        # NOTE: This is only internal usage.

        def _get_transformation_config() -> TransformationConfig:
            """Get transformation matrix for the sensor hardware configuration."""

            def _get_kitti_transformation_config():
                def _get_rigid_transformation(calib_path: str) -> Tuple[np.ndarray, np.ndarray]:
                    with open(calib_path, 'r') as f:
                        calib = f.readlines()
                    R = np.array([float(x) for x in calib[1].strip().split(' ')[1:]]).reshape((3, 3))
                    t = np.array([float(x) for x in calib[2].strip().split(' ')[1:]])[:, None]
                    T = np.vstack((np.hstack((R, t)), np.array([0, 0, 0, 1])))
                    return T

                # kitti_variant = KITTI_SEQUENCE_MAPS.get(self.dataset.variant)
                root_calibration_path = os.path.join(self.dataset.root_path, f'seq_{self.dataset.variant}', 'calibration')
                T_calib_velo_to_cam = _get_rigid_transformation(os.path.join(root_calibration_path, "calib_velo_to_cam.txt"))
                T_calib_imu_to_velo = _get_rigid_transformation(os.path.join(root_calibration_path, "calib_imu_to_velo.txt"))

                return T_calib_velo_to_cam, T_calib_imu_to_velo

            def _get_uav_transformation_config():

                def load_voxl_extrinsics(conf_path):
                    text = Path(conf_path).read_text()
                    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
                    data = json.loads(text)
                    extrinsics = []
                    for entry in data["extrinsics"]:
                        rpy_deg = entry["RPY_parent_to_child"]
                        r = Rotation.from_euler('xyz', rpy_deg, degrees=True).as_matrix()
                        extrinsics.append({
                            "parent": entry["parent"],
                            "child": entry["child"],
                            "translation": entry["T_child_wrt_parent"],
                            "rpy_degrees": rpy_deg,
                            "rotation_matrix": r
                        })
                    return extrinsics

                def _get_imu_transformations(imu_calib: dict) -> Dict[SensorType, np.ndarray]:
                    
                    # voxl_imu0 = imu_calib.get("voxl_imu0", {})
                    voxl_imu1 = imu_calib.get("voxl_imu1", {})
                    px4_imu0 = imu_calib.get("px4_imu0", {})
                    px4_imu1 = imu_calib.get("px4_imu1", {})

                    voxl_imu0_to_imu0 = Pose(
                        R=np.eye(3),
                        t=np.array([0, 0, 0])
                    )
                    voxl_imu1_to_imu0 = Pose(
                        R=np.array(voxl_imu1.get("R", np.eye(3))),
                        t=np.array(voxl_imu1.get("t", [0, 0, 0]))
                    )
                    px4_imu0_to_imu0 = Pose(
                        R=np.array(px4_imu0.get("R", np.eye(3))),
                        t=np.array(px4_imu0.get("t", [0, 0, 0]))
                    )
                    px4_imu1_to_imu0 = px4_imu0_to_imu0 * Pose(
                        R=np.array(px4_imu1.get("R", np.eye(3))),
                        t=np.array(px4_imu1.get("t", [0, 0, 0]))
                    )
                    
                    
                    return {
                        SensorType.VOXL_IMU0: voxl_imu0_to_imu0.matrix(),
                        SensorType.VOXL_IMU1: voxl_imu1_to_imu0.matrix(),
                        SensorType.PX4_IMU0: px4_imu0_to_imu0.matrix(),
                        SensorType.PX4_IMU1: px4_imu1_to_imu0.matrix(),
                    }

                imu_calib_file = os.path.join(self.dataset.root_path, f'configs', 'imu_extrinsic.yaml')
                extrinsic_file = os.path.join(self.dataset.root_path, self.dataset.variant, 'etc/modalai/extrinsics.conf')
                imu_calib = None

                with open(imu_calib_file, 'r') as f:
                    imu_calib = yaml.safe_load(f)
                
                system_calib = load_voxl_extrinsics(extrinsic_file)

                body_to_stereo_left = [e for e in system_calib if e["parent"] == "body" and e["child"] == "stereo_l"]
                body_to_imu0 = [e for e in system_calib if e["parent"] == "body" and e["child"] == "stereo_l"]
                if len(body_to_stereo_left) == 0 or len(body_to_imu0) == 0:
                    raise ValueError("No stereo left camera found in the UAV calibration file.")
                
                body_to_stereo_left = body_to_stereo_left[0]
                body_to_imu0 = body_to_imu0[0]
                R = np.array(body_to_stereo_left["rotation_matrix"])
                t = np.array(body_to_stereo_left["translation"])
                R_imu = np.array(body_to_imu0["rotation_matrix"])
                t_imu = np.array(body_to_imu0["translation"])

                T_stereo_wrt_body = Pose(R=R, t=t)
                T_body_to_stereo_left = T_stereo_wrt_body.inverse()
                T_imu_to_body = Pose(R=R_imu, t=t_imu)
                T_body_to_imu0 = T_imu_to_body.inverse()
                T_from_cam_to_imu = T_body_to_imu0 * T_body_to_stereo_left
                
                T_imu_to_virtual_imu = _get_imu_transformations(imu_calib)

                return T_from_cam_to_imu, T_imu_to_body, T_imu_to_virtual_imu

            def _get_euroc_transformation_config():
                def _get_calibration_data(calib_path: str) -> Pose:
                    with open(calib_path, 'r') as f:
                        calib = yaml.safe_load(f)

                    data = np.array(calib["T_BS"]["data"]).reshape(4, 4)
                    return Pose(R=data[:3, :3], t=data[:3, 3])

                variant = f"mav_{self.dataset.variant}"
                imu_calibration_path = os.path.join(self.dataset.root_path, variant, 'imu0/sensor.yaml')
                leica_calibration_path = os.path.join(self.dataset.root_path, variant, 'leica0/sensor.yaml')
                camera_calibration_path = os.path.join(self.dataset.root_path, variant, 'cam0/sensor.yaml')
                T_calib_imu_to_inertial = _get_calibration_data(imu_calibration_path)
                T_calib_leica_to_inertial = _get_calibration_data(leica_calibration_path)
                T_calib_cam_to_inertial = _get_calibration_data(camera_calibration_path)

                # R = BaseGeometryTransformer.Rx(np.radians(180)) @ BaseGeometryTransformer.Ry(np.radians(-90))
                # T_calib_imu_to_inertial = Pose(R=R, t=np.zeros(3))

                return T_calib_imu_to_inertial, T_calib_leica_to_inertial, T_calib_cam_to_inertial

            dataset_type = DatasetType.get_type_from_str(self.dataset.type)
            if dataset_type == DatasetType.KITTI:
                T_calib_velo_to_cam, T_calib_imu_to_velo = _get_kitti_transformation_config()

                return TransformationConfig.from_kitti_config(
                    T_calib_velo_to_cam=T_calib_velo_to_cam,
                    T_calib_imu_to_velo=T_calib_imu_to_velo,
                )
            elif dataset_type == DatasetType.UAV:
                T_from_cam_to_imu, T_imu_to_body, T_imu_to_virtual_imu = _get_uav_transformation_config()
                
                return TransformationConfig(
                    T_from_cam_to_imu=T_from_cam_to_imu.matrix(),
                    T_from_imu_to_cam=T_from_cam_to_imu.inverse().matrix(),
                    T_imu_body_to_inertial=T_imu_to_body.matrix(),
                    T_imu_to_virtual_imu=T_imu_to_virtual_imu
                )
            elif dataset_type == DatasetType.EUROC:
                T_calib_imu_to_inertial, T_calib_leica_to_inertial, T_calib_cam_to_inertial = _get_euroc_transformation_config()

                T_from_cam_to_imu = T_calib_cam_to_inertial.matrix()
                T_from_imu_to_cam = T_calib_cam_to_inertial.inverse().matrix()
                T_imu_body_to_inertial = T_calib_imu_to_inertial.matrix()
                T_imu_body_to_leica = T_calib_leica_to_inertial.inverse().matrix()


                return TransformationConfig(
                    T_from_cam_to_imu=T_from_cam_to_imu,
                    T_from_imu_to_cam=T_from_imu_to_cam,
                    T_imu_body_to_inertial=T_imu_body_to_inertial,
                    T_imu_body_to_leica=T_imu_body_to_leica,
                )

        def _get_imu_config() -> ImuConfig:
            
            get_sensor_from_str = SensorType.get_sensor_from_str_func(d=self.dataset.type)
            imu = [sensor for sensor in self.dataset.sensors if SensorType.is_imu_data(get_sensor_from_str(sensor.name))]
            frequency = imu[0].args.get("frequency", 100) if len(imu) != 0 else 100
            if frequency < 100:
                logging.warning(f"IMU frequency is set to {frequency} Hz. This is lower than the default value of 100 Hz.")
                frequency = 100
            return ImuConfig(frequency=frequency)
        
        def _get_beacon_config() -> BeaconConfig:
            _ = DatasetType.get_type_from_str(self.dataset.type)

            return BeaconConfig.from_data(json_data=None, vehicle_state_df=None)

        def _get_geofencing_config() -> GeofencingConfig:
            _ = DatasetType.get_type_from_str(self.dataset.type)
            return GeofencingConfig(fencing_lines={})

        def _get_drone_config() -> DroneHardwareConfig:
            if self.filter.motion_model == "drone_kinematics":
                try:
                    return DroneHardwareConfig(**self.parsed_config["drone_config"])
                except:
                    logging.error(f"Error in loading drone hardware configuration.")
                    return None
            return None


        imu_config = _get_imu_config()
        beacon_config = _get_beacon_config()
        transformation = _get_transformation_config()
        geofencing_config = _get_geofencing_config()
        drone_hardware_config = _get_drone_config()
        return HardwareConfig(
            type=self.dataset.type,
            imu_config=imu_config,
            beacon_config=beacon_config,
            transformation=transformation,
            geofencing_config=geofencing_config,
            drone_hardware_config=drone_hardware_config,
        )

