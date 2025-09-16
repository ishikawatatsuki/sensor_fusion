import logging
import numpy as np
from collections import namedtuple

from ...common import (
    HardwareConfig,
    CoordinateFrame,
    SensorDataField,
)
from .base_geometric_transformer import BaseGeometryTransformer, TransformationField


class EuRoC_GeometricTransformer(BaseGeometryTransformer):

    def __init__(
            self,
            hardware_config: HardwareConfig
        ):
        super().__init__()
        
        self.hardware_config = hardware_config

        self.T_from_cam_to_imu = hardware_config.transformation.T_from_cam_to_imu
        self.T_from_imu_to_cam = hardware_config.transformation.T_from_imu_to_cam
        self.T_imu_body_to_inertial = hardware_config.transformation.T_imu_body_to_inertial
        self.T_leica_to_inertial = hardware_config.transformation.T_leica_to_inertial
        self.T_leica_to_imu = np.linalg.inv(self.T_imu_body_to_inertial) @ self.T_leica_to_inertial
        
        self.origin = None
        self.vo_origin = None
    
    def transform_data(self, fields: TransformationField) -> np.ndarray:
        match (fields.coord_from):
            case CoordinateFrame.LEICA:
                return self._transform_leica_data(fields).reshape(-1, 1)
            case CoordinateFrame.STEREO_LEFT:
                return self._transform_vo_data(fields)
            case CoordinateFrame.IMU:
                return self._transform_imu_data(fields).reshape(-1, 1)
            case _:
                logging.warning(f"Data is not transformed. {fields.coord_from} -> {fields.coord_to}")
                return np.array(fields.value).reshape(-1, 1)
            
    def _decimal_place_shift(self, values: np.ndarray) -> float:
        return np.array([float(value / 10**(len(str(value)) - 2)) for value in values])

    def _transform_leica_to_inertial(self, leica_data: np.ndarray) -> np.ndarray:
        """Transform LEICA data into inertial coordinate frame."""
        leica_inertial = self.T_leica_to_inertial @ np.array([leica_data[0], leica_data[1], leica_data[2], 1])
        return leica_inertial.flatten()[:3]

    def _transform_leica_to_imu(self, leica_data: np.ndarray) -> np.ndarray:
        """Transform LEICA data into IMU coordinate frame."""
        leica_imu_body = self.T_leica_to_imu @ np.array([leica_data[0], leica_data[1], leica_data[2], 1])
        return leica_imu_body[:3].flatten()

    def _transform_leica_data(self, fields: TransformationField) -> np.ndarray:
        """Transform data in GPS coordinate into other coordinate frames."""
        match (fields.coord_to):
            case CoordinateFrame.INERTIAL:
                return self._transform_leica_to_inertial(fields.value)
            case CoordinateFrame.IMU:
                return self._transform_leica_to_imu(fields.value)
            case _:
                logging.warning("GPS data is not transformed.")
                return fields.value

    def _process_vo_data(self, data: np.ndarray) -> np.ndarray:
        """Transform VO data into ENU coordinate frame.
            data: 3x4 matrix
        """
        pose = np.eye(4)
        pose[:3, :] = data
        pose = self.T_from_cam_to_imu @ pose @ np.linalg.inv(self.T_from_cam_to_imu)

        return pose
    
    def _transform_vo_data(self, fields: TransformationField) -> np.ndarray:
        """Transform data in camera coordinate into other coordinate frame."""
        match (fields.coord_to):
            case CoordinateFrame.INERTIAL | CoordinateFrame.IMU:
                vo_in_imu = self._process_vo_data(fields.value)
                return vo_in_imu
            case _:
                logging.warning("VO data is not transformed.")
                return fields.value

    def _process_imu_into_camera_coord(self, imu_data: np.ndarray) -> np.ndarray:
        """Transform IMU data in KITTI dataset into ENU coordinate frame."""
        imu = imu_data.flatten()
        a = imu[:3]
        w = imu[3:6]
        a = self.T_from_imu_to_cam @ np.array([a[0], a[1], a[2], 0])
        w = self.T_from_imu_to_cam @ np.array([w[0], w[1], w[2], 0])
        a = a.flatten()
        w = w.flatten()
        return np.hstack([a[:3], w[:3]]).flatten()
        
    def _process_imu_into_inertial_coord(self, imu_data: np.ndarray) -> np.ndarray:
        """Transform IMU data in KITTI dataset into ENU coordinate frame."""
        imu = imu_data.flatten()
        a = imu[:3]
        w = imu[3:6]
        a = self.T_imu_body_to_inertial @ np.array([a[0], a[1], a[2], 0])
        w = self.T_imu_body_to_inertial @ np.array([w[0], w[1], w[2], 0])
        a = a.flatten()
        w = w.flatten()
        return np.hstack([a[:3], w[:3]]).flatten()
        
    def _transform_imu_data(self, fields: TransformationField) -> np.ndarray:
        """Transform data in IMU coordinate into other coordinate frames."""
        match (fields.coord_to):
            case CoordinateFrame.STEREO_LEFT:
                return self._process_imu_into_camera_coord(fields.value)
            case CoordinateFrame.INERTIAL:
                return self._process_imu_into_inertial_coord(fields.value)
            case _:
                logging.warning("IMU data is not transformed.")
                return fields.value

    def _transform_virtual_imu_data(self, data: SensorDataField) -> np.ndarray:
        """Transform IMU data into virtual IMU coordinate frame."""
        return data.data
