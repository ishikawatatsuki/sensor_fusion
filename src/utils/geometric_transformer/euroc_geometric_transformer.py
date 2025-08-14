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
        self.T_imu_body_to_leica = hardware_config.transformation.T_imu_body_to_leica
        
        self.origin = None
        self.vo_origin = None
    
    def transform_data(self, fields: TransformationField) -> np.ndarray:
        match (fields.coord_from):
            case CoordinateFrame.LEICA:
                return self._transform_leica_data(fields).reshape(-1, 1)
            case CoordinateFrame.STEREO_LEFT:
                return self._transform_vo_data(fields).reshape(-1, 1)
            case CoordinateFrame.IMU:
                return self._transform_imu_data(fields).reshape(-1, 1)
            case _:
                logging.warning(f"Data is not transformed. {fields.coord_from} -> {fields.coord_to}")
                return np.array(fields.value).reshape(-1, 1)
            
    def _decimal_place_shift(self, values: np.ndarray) -> float:
        return np.array([float(value / 10**(len(str(value)) - 2)) for value in values])

    def _process_leica_data(self, leica_data: np.ndarray) -> np.ndarray:
        """Transform GPS data in KITTI dataset into ENU coordinate frame."""
        pose = np.eye(4)
        pose[:3, 3] = leica_data[:3]
        pose = self.T_imu_body_to_leica @ pose
        return pose[:3, 3].flatten()

    def _transform_leica_data(self, fields: TransformationField) -> np.ndarray:
        """Transform data in GPS coordinate into other coordinate frames."""
        match (fields.coord_to):
            case CoordinateFrame.IMU | CoordinateFrame.INERTIAL:
                return self._process_leica_data(fields.value)
            case _:
                logging.warning("GPS data is not transformed.")
                return fields.value

    def _process_vo_data(self, data: np.ndarray) -> np.ndarray:
        """Transform VO data in KITTI dataset into ENU coordinate frame."""
        pose = np.eye(4)
        pose[:3, 3] = data[:3]
        pose = self.T_from_cam_to_imu @ pose

        return pose[:3, 3].flatten()
    
    def _transform_vo_data(self, fields: TransformationField) -> np.ndarray:
        """Transform data in camera coordinate into other coordinate frame."""
        match (fields.coord_to):
            case CoordinateFrame.INERTIAL | CoordinateFrame.IMU:
                vo_in_imu = self._process_vo_data(fields.value)
                if self.vo_origin is None:
                    self.vo_origin = vo_in_imu[:3]
                vo_in_imu[:3] -= self.vo_origin
                return vo_in_imu
            case _:
                logging.warning("VO data is not transformed.")
                return fields.value

    def _process_px4_custom_vo_data(self, vo_data: np.ndarray) -> np.ndarray:
        position = BaseGeometryTransformer.Rz(np.radians(180)) @ vo_data[:3]
        position = position.flatten()
        if vo_data.shape[0] > 3:
            velocity = BaseGeometryTransformer.Rz(np.radians(180)) @ vo_data[3:]
            velocity = velocity.flatten()
            return np.hstack([position, velocity])
        return position
    

    def _process_imu_into_camera_coord(self, imu_data: np.ndarray) -> np.ndarray:
        """Transform IMU data in KITTI dataset into ENU coordinate frame."""
        # pose = np.eye(4)
        # pose[:3, 3] = imu_data[:3]
        # pose = self.T_from_imu_to_cam @ pose
        # a = pose[:3, 3].flatten()

        a = imu_data.flatten()[:3]
        w = imu_data.flatten()[3:6]
        a = self.T_from_imu_to_cam[:3, :3] @ a
        w = self.T_from_imu_to_cam[:3, :3] @ w
        a = a.flatten()
        w = w.flatten()
        return np.hstack([a, w]).flatten()
        
    def _process_imu_into_inertial_coord(self, imu_data: np.ndarray) -> np.ndarray:
        """Transform IMU data in KITTI dataset into ENU coordinate frame."""
        # pose = np.eye(4)
        # pose[:3, 3] = imu_data[:3]
        # pose = self.T_imu_body_to_inertial @ pose
        # a = pose[:3, 3].flatten()

        
        a = imu_data.flatten()[:3]
        w = imu_data.flatten()[3:6]
        a = self.T_imu_body_to_inertial[:3, :3] @ a
        w = self.T_imu_body_to_inertial[:3, :3] @ w
        a = a.flatten()
        w = w.flatten()
        return np.hstack([a, w]).flatten()
        
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
