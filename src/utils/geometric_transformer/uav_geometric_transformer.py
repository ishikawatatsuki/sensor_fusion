import logging
import numpy as np

from ...common import (
    State,
    HardwareConfig,
    CoordinateFrame,
    SensorDataField,

    DECLINATION_OFFSET_RADIAN_IN_ESTONIA
)
from .base_geometric_transformer import BaseGeometryTransformer, TransformationField


class UAV_GeometricTransformer(BaseGeometryTransformer):

    def __init__(
            self,
            hardware_config: HardwareConfig
        ):
        super().__init__()
        
        self.hardware_config = hardware_config
        

        self.T_from_imu_to_cam = hardware_config.transformation.T_from_imu_to_cam
        self.T_from_cam_to_imu = hardware_config.transformation.T_from_cam_to_imu
        self.T_imu_body_to_inertial = hardware_config.transformation.T_imu_body_to_inertial

        self.T_imu_to_virtual_imu = hardware_config.transformation.T_imu_to_virtual_imu
        
        self.origin = None
        self.vo_origin = None
    

    def transform_data(self, fields: TransformationField) -> np.ndarray:
        match (fields.coord_from):
            case CoordinateFrame.GPS:
                return self._transform_gps_data(fields).reshape(-1, 1)
            case CoordinateFrame.STEREO_LEFT:
                return self._transform_vo_data(fields)
            case CoordinateFrame.MAGNETOMETER:
                return self._transform_magnetometer_data(fields).reshape(-1, 1)
            case CoordinateFrame.IMU:
                return self._transform_imu_data(fields).reshape(-1, 1)
            case _:
                logging.warning(f"Data is not transformed. {fields.coord_from} -> {fields.coord_to}")
                return np.array(fields.value).reshape(-1, 1)
            
    def _decimal_place_shift(self, values: np.ndarray) -> float:
        return np.array([float(value / 10**(len(str(value)) - 2)) for value in values])

    def _process_uav_gps_data(self, gps_data: np.ndarray) -> np.ndarray:
        """Transform GPS data in KITTI dataset into ENU coordinate frame."""
        if self.origin is None:
            self.origin = self._decimal_place_shift(gps_data)
            return np.array([0., 0., 0.])
        
        # NOTE: lla to ned coord
        gps_data = self.lla_to_ned(self._decimal_place_shift(gps_data).reshape(-1, 1), self.origin).flatten()
        return gps_data

    def _transform_gps_data(self, fields: TransformationField) -> np.ndarray:
        """Transform data in GPS coordinate into other coordinate frames."""
        match (fields.coord_to):
            case CoordinateFrame.IMU | CoordinateFrame.INERTIAL:
                return self._process_uav_gps_data(fields.value)
            case _:
                logging.warning("GPS data is not transformed.")
                return fields.value

    def _process_vo_data(self, data: np.ndarray) -> np.ndarray:
        relative_pose = np.eye(4)
        relative_pose[:3, :] = data.copy()
        relative_pose_inertial_coord = self.T_from_cam_to_imu @ relative_pose @ np.linalg.inv(self.T_from_cam_to_imu)

        return relative_pose_inertial_coord[:3, :]
    
    def _transform_vo_data(self, fields: TransformationField) -> np.ndarray:
        """Transform data in camera coordinate into other coordinate frame."""
        match (fields.coord_to):
            case CoordinateFrame.INERTIAL | CoordinateFrame.IMU:
                vo_in_imu = self._process_vo_data(fields.value)
                # if self.vo_origin is None:
                #     self.vo_origin = vo_in_imu[:3]
                # vo_in_imu[:3] -= self.vo_origin
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
    
    def _process_mag_data(self, mag_data: np.ndarray, state: State) -> np.ndarray:
        """Convert magnetometer data measured in Body frame into inertial frame using the rotation matrix obtained from the quaternion.

        Args:
            position (np.ndarray): A earth's magnetic field reading in body frame.

        Returns:
            np.ndarray: The Earth's magnetic field in inertial frame.
        """
        mag_data = mag_data.flatten()
        # TODO: convert lla to NED coordinate
        Rot = state.get_rotation_matrix()
        mag_data = (Rot @ mag_data).flatten()
        
        z_m = np.arctan2(mag_data[0], mag_data[1])
        z_m += DECLINATION_OFFSET_RADIAN_IN_ESTONIA
        
        return z_m

    def _transform_magnetometer_data(self, fields: TransformationField) -> np.ndarray:
        """Transform data in IMU coordinate into other coordinate frames."""
        match (fields.coord_to):
            case CoordinateFrame.INERTIAL:
                return self._process_mag_data(fields.value, fields.state)
            case _:
                logging.warning("Magnetometer data is not transformed.")
                return fields.value

    def _process_imu_into_camera_coord(self, imu_data: np.ndarray) -> np.ndarray:
        # values = np.array([imu_data[0], imu_data[1], imu_data[2], 1])
        values = imu_data.flatten()[:3]
        transformed_values = self.T_from_imu_to_cam[:3, :3] @ values
        return np.array([transformed_values[0], transformed_values[1], transformed_values[2]])

    def _process_imu_into_inertial_coord(self, imu_data: np.ndarray) -> np.ndarray:
        return imu_data
        
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

        T_imu_to_imu0 = self.T_imu_to_virtual_imu.get(data.type, None)
        if len(self.T_imu_to_virtual_imu.keys()) == 0:
            return data.data
        R = T_imu_to_imu0[:3, :3]
        transformed_value = []

        x = data.data.flatten()
        a = x[:3]
        a = R @ a
        transformed_value.extend(a)
        if x.shape[0] > 3:
            w = x[3:6]
            w = R @ w
            transformed_value.extend(w)
        if x.shape[0] > 6:
            m = x[6:9]
            m = R @ m        
            transformed_value.extend(m)

        
        return np.array(transformed_value)
