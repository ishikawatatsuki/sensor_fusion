import logging
import numpy as np

from ...common import (
    State,
    HardwareConfig,
    CoordinateFrame,
    SensorDataField,
)
from .base_geometric_transformer import BaseGeometryTransformer, TransformationField

class KITTI_GeometricTransformer(BaseGeometryTransformer):

    def __init__(
            self,
            hardware_config: HardwareConfig
        ):
        super().__init__()
        
        self.hardware_config = hardware_config
        
        self.T_from_imu_to_cam = hardware_config.transformation.T_from_imu_to_cam
        self.T_from_cam_to_imu = hardware_config.transformation.T_from_cam_to_imu
        self.T_imu_body_to_inertial = hardware_config.transformation.T_imu_body_to_inertial
        self.T_angle_compensation = hardware_config.transformation.T_angle_compensation

        self.origin = None
        self.vo_origin = None

    def transform_data(self, fields: TransformationField) -> np.ndarray:
        match (fields.coord_from):
            case CoordinateFrame.IMU:
                return self._transform_imu_data(fields).reshape(-1, 1)
            case CoordinateFrame.GPS:
                return self._transform_gps_data(fields).reshape(-1, 1)
            case CoordinateFrame.STEREO_LEFT:
                return self._transform_vo_data(fields)
            case CoordinateFrame.INERTIAL:
                return self._transform_inertial_data(fields)
            case _:
                return np.array(fields.value).reshape(-1, 1)
            
    def _process_kitti_gps_data(self, gps_data: np.ndarray) -> np.ndarray:
        """Transform GPS data in KITTI dataset into ENU coordinate frame."""
        if self.origin is None:
            self.origin = gps_data
            return np.array([0., 0., 0.])
        
        # NOTE: lla to enu coord
        gps_data = self.lla_to_enu(gps_data.reshape(-1, 1), self.origin).flatten()
        gps_data = self.T_angle_compensation @ np.array([gps_data[0], gps_data[1], gps_data[2], 0.])
        gps_data = gps_data[:3]

        return gps_data

    def _transform_gps_data(self, fields: TransformationField) -> np.ndarray:
        """Transform data in GPS coordinate into other coordinate frames."""
        match (fields.coord_to):
            case CoordinateFrame.IMU | CoordinateFrame.INERTIAL:
                return self._process_kitti_gps_data(fields.value)
            case _:
                logging.warning("GPS data is not transformed.")
                return fields.value

    def _transform_kitti_imu_into_cam_coord(self, imu_data: np.ndarray) -> np.ndarray:
        values = np.array([imu_data[0], imu_data[1], imu_data[2], 1])
        transformed_values = self.T_from_imu_to_cam @ values
        return np.array([transformed_values[0], transformed_values[1], transformed_values[2]])
    
    def _transform_imu_data(self, fields: TransformationField) -> np.ndarray:
        """Transform data in IMU coordinate into other coordinate frames."""
        match (fields.coord_to):
            case CoordinateFrame.STEREO_LEFT:
                return self._transform_kitti_imu_into_cam_coord(fields.value)
            case _:
                return fields.value

    def _transform_inertial_data_into_camera_coord(self, data: np.ndarray) -> np.ndarray:
        """Transform inertial data into camera coordinate frame."""

        if data.ndim == 2:
            pose = np.eye(4)
            pose[:3, :] = data.copy()[:3, :]
            pose_in_camera_coord = self.T_from_imu_to_cam @ pose

            return pose_in_camera_coord[:3, :]


        if data.shape[0] == 3:
            return (self.T_from_imu_to_cam @ np.hstack([data, np.array([1.])]))[:3]
        elif data.shape[0] == 6:
            arr = np.array([1.])
            p = self.T_from_imu_to_cam @ np.hstack([data[:3], arr])
            v = self.T_from_imu_to_cam @ np.hstack([data[3:], arr])
            p = p[:3]
            v = v[:3]
            return np.hstack([p, v])
        elif data.shape[0] == 7:
            mat = np.eye(4)
            t = data[:3]
            R = State.get_rotation_matrix_from_quaternion_vector(data[3:])
            mat[:3, :3] = R
            mat[:3, 3] = t
            pose_in_camera_coord = self.T_from_imu_to_cam @ mat
            q = State.get_quaternion_from_rotation_matrix(pose_in_camera_coord[:3, :3]).flatten()
            t = pose_in_camera_coord[:3, 3].flatten()
            return np.hstack([t, q])
        else:
            arr = np.array([1.])
            v = self.T_from_imu_to_cam @ np.hstack([data[3:], arr])
            v = v[:3]
            mat = np.eye(4)
            t = data[:3]
            R = State.get_rotation_matrix_from_quaternion_vector(data[6:])
            mat[:3, :3] = R
            mat[:3, 3] = t
            pose_in_camera_coord = self.T_from_imu_to_cam @ mat
            q = State.get_quaternion_from_rotation_matrix(pose_in_camera_coord[:3, :3]).flatten()
            t = pose_in_camera_coord[:3, 3].flatten()
            return np.hstack([t, v, q])
        
    def _transform_inertial_data_into_gps_coord(self, data: np.ndarray) -> np.ndarray:
        """Transform inertial data into GPS coordinate frame."""
        if self.origin is None:
            return None
        
        position_in_gps_coord = self.enu_to_lla(data.reshape(-1, 1), self.origin).flatten()
        return position_in_gps_coord

    def _transform_inertial_data(self, fields: TransformationField) -> np.ndarray:
        """Transform data in inertial frame into other coordinate frames."""
        match (fields.coord_to):
            case CoordinateFrame.IMU:
                # NOTE: In KITTI dataset, IMU Coordinate is aligned with the inertial frame. So, no transformation is required.
                return fields.value.reshape(-1, 1)
            case CoordinateFrame.STEREO_LEFT:
                return self._transform_inertial_data_into_camera_coord(fields.value)
            case CoordinateFrame.GPS:
                # TODO: implement transformation into the GPS coordinate frame (lon, lat, alt)
                return self._transform_inertial_data_into_gps_coord(fields.value).reshape(-1, 1)
            case _:
                logging.warning("Inertial data is not transformed.")
                return fields.value
    
    def _transform_kitti_vo_data_into_imu_coord(self, vo_data: np.ndarray) -> np.ndarray:

        # assuming 3x4 relative pose matrix.
        relative_pose = np.eye(4)
        relative_pose[:3, :] = vo_data.copy()
        relative_pose_inertial_coord = self.T_from_cam_to_imu @ relative_pose @ np.linalg.inv(self.T_from_cam_to_imu)

        return relative_pose_inertial_coord[:3, :]

    def _transform_vo_data(self, fields: TransformationField) -> np.ndarray:
        """Transform data in camera coordinate into other coordinate frame."""
        match (fields.coord_to):
            case CoordinateFrame.INERTIAL | CoordinateFrame.IMU:
                vo_in_imu = self._transform_kitti_vo_data_into_imu_coord(fields.value)
                return vo_in_imu
            case _:
                logging.warning("VO data is not transformed.")
                return fields.value

    def _transform_virtual_imu_data(self, data: SensorDataField) -> np.ndarray:
        """Transform IMU data into virtual IMU coordinate frame."""
        return data.data
