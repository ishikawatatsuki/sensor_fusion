import os
import sys
import abc
import logging
import numpy as np
from collections import namedtuple

from ...common import (
    FilterConfig, HardwareConfig,
    State,
    CoordinateFrame,
    SensorType,
    DatasetType,
    SensorDataField,
    DECLINATION_OFFSET_RADIAN_IN_ESTONIA
)

TransformationField = namedtuple('TransformationField', ['state', 'value', 'coord_from', 'coord_to'])


class BaseGeometryTransformer(abc.ABC):
    
    def __init__(self):
        
        self._a = 6378137.0
        self._f = 1. / 298.257223563
        self._b = (1. - self._f) * self._a
        self._e = np.sqrt(self._a ** 2. - self._b ** 2.) / self._a
        self._e_prime = np.sqrt(self._a ** 2. - self._b ** 2.) / self._b

    def _lla_to_ecef(self, points_lla: np.ndarray) -> np.ndarray:
        """transform N x [longitude(deg), latitude(deg), altitude(m)] coords into
        N x [x, y, z] coords measured in Earth-Centered-Earth-Fixed frame.
        """
        lon = np.radians(points_lla[0])  # [N,]
        lat = np.radians(points_lla[1])  # [N,]
        alt = points_lla[2]  # [N,]

        N = self._a / np.sqrt(1. - (self._e * np.sin(lat)) ** 2.)  # [N,]
        x = (N + alt) * np.cos(lat) * np.cos(lon)
        y = (N + alt) * np.cos(lat) * np.sin(lon)
        z = (N * (1. - self._e ** 2.) + alt) * np.sin(lat)

        points_ecef = np.stack([x, y, z], axis=0)  # [3, N]
        return points_ecef


    def _ecef_to_enu(self, points_ecef: np.ndarray, ref_lla: np.ndarray) -> np.ndarray:
        """transform N x [x, y, z] coords measured in Earth-Centered-Earth-Fixed frame into
        N x [x, y, z] coords measured in a local East-North-Up frame.
        """
        lon = np.radians(ref_lla[0])
        lat = np.radians(ref_lla[1])

        ref_ecef = self._lla_to_ecef(ref_lla)  # [3,]

        relative = points_ecef - ref_ecef[:, np.newaxis]  # [3, N]
        # R = Rz(np.pi / 2.0) @ Ry(np.pi / 2.0 - lat) @ Rz(lon)  # [3, 3]
        R = np.array([
            [-np.sin(lon), np.cos(lon), 0],
            [-np.sin(lat)*np.cos(lon), -np.sin(lat)*np.sin(lon), np.cos(lat)],
            [np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)]
        ])
        points_enu = R @ relative  # [3, N]
        return points_enu

    def _ecef_to_ned(self, points_ecef: np.ndarray, ref_lla: np.ndarray) -> np.ndarray:
        """transform N x [x, y, z] coords measured in Earth-Centered-Earth-Fixed frame into
        N x [x, y, z] coords measured in a local North-East-Down frame.
        """
        lon = np.radians(ref_lla[0])
        lat = np.radians(ref_lla[1])

        ref_ecef = self._lla_to_ecef(ref_lla)  # [3,]

        relative = points_ecef - ref_ecef[:, np.newaxis]  # [3, N]

        # R = Rz(np.pi / 2.0) @ Ry(np.pi / 2.0 - lat) @ Rz(lon)  # [3, 3]
        R = np.array([
            [-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)],
            [-np.sin(lon), np.cos(lon), 0],
            [-np.cos(lat) * np.cos(lon), -np.cos(lat) * np.sin(lon), -np.sin(lat)]
        ])
        points_ned = R @ relative  # [3, N]
        return points_ned


    def lla_to_enu(self, points_lla: np.ndarray, ref_lla: np.ndarray) -> np.ndarray:
        """transform N x [longitude(deg), latitude(deg), altitude(m)] coords into
        N x [x, y, z] coords measured in a local East-North-Up frame.
        """
        points_ecef = self._lla_to_ecef(points_lla)
        points_enu = self._ecef_to_enu(points_ecef, ref_lla)
        return points_enu

    def lla_to_ned(self, points_lla: np.ndarray, ref_lla: np.ndarray) -> np.ndarray:
        """transform N x [longitude(deg), latitude(deg), altitude(m)] coords into
        N x [x, y, z] coords measured in a local North-East-Down frame.
        """
        points_ecef = self._lla_to_ecef(points_lla)
        points_ned = self._ecef_to_ned(points_ecef, ref_lla)
        return points_ned

    def enu_to_ecef(self, points_enu: np.ndarray, ref_lla: np.ndarray) -> np.ndarray:
        """transform N x [x, y, z] coords measured in a local East-North-Up frame into
        N x [x, y, z] coords measured in Earth-Centered-Earth-Fixed frame.
        """
        # inverse transformation of `ecef_to_enu`

        lon = np.radians(ref_lla[0])
        lat = np.radians(ref_lla[1])
        alt = ref_lla[2]

        ref_ecef = self._lla_to_ecef(ref_lla)  # [3,]

        R = BaseGeometryTransformer.Rz(np.pi / 2.0) @ BaseGeometryTransformer.Ry(np.pi / 2.0 - lat) @ BaseGeometryTransformer.Rz(lon)  # [3, 3]
        R = R.T  # inverse rotation
        relative = R @ points_enu  # [3, N]

        points_ecef = ref_ecef[:, np.newaxis] + relative  # [3, N]
        return points_ecef


    def ecef_to_lla(self, points_ecef: np.ndarray) -> np.ndarray:
        """transform N x [x, y, z] coords measured in Earth-Centered-Earth-Fixed frame into
        N x [longitude(deg), latitude(deg), altitude(m)] coords.
        """
        # approximate inverse transformation of `lla_to_ecef`
        
        x = points_ecef[0]  # [N,]
        y = points_ecef[1]  # [N,]
        z = points_ecef[2]  # [N,]

        p = np.sqrt(x ** 2. + y ** 2.)  # [N,]
        theta = np.arctan(z * self._a / (p * self._b))  # [N,]

        lon = np.arctan(y / x)  # [N,]
        lat = np.arctan(
            (z + (self._e_prime ** 2.) * self._b * (np.sin(theta) ** 3.)) / \
            (p - (self._e ** 2.) * self._a * (np.cos(theta)) ** 3.)
        )  # [N,]
        N = self._a / np.sqrt(1. - (self._e * np.sin(lat)) ** 2.)  # [N,]
        alt = p / np.cos(lat) - N  # [N,]

        lon = np.degrees(lon)
        lat = np.degrees(lat)

        points_lla = np.stack([lon, lat, alt], axis=0)  # [3, N]
        return points_lla


    def enu_to_lla(self, points_enu: np.ndarray, ref_lla: np.ndarray) -> np.ndarray:
        """transform N x [x, y, z] coords measured in a local East-North-Up frame into
        N x [longitude(deg), latitude(deg), altitude(m)] coords.
        """
        points_ecef = self._enu_to_ecef(points_enu, ref_lla)
        points_lla = self._ecef_to_lla(points_ecef)
        return points_lla

    def _get_rigid_transformation(self, calib_path: str) -> np.ndarray:
        with open(calib_path, 'r') as f:
            calib = f.readlines()
        R = np.array([float(x) for x in calib[1].strip().split(' ')[1:]]).reshape((3, 3))
        t = np.array([float(x) for x in calib[2].strip().split(' ')[1:]])[:, None]
        T = np.vstack((np.hstack((R, t)), np.array([0, 0, 0, 1])))
        return T
    
    @staticmethod
    def Rx(theta):
        """rotation matrix around x-axis
        """
        c = np.cos(theta)
        s = np.sin(theta)
        return np.array([
            [1, 0, 0],
            [0, c, s],
            [0, -s, c]
        ])


    @staticmethod
    def Ry(theta):
        """rotation matrix around y-axis
        """
        c = np.cos(theta)
        s = np.sin(theta)
        return np.array([
            [c, 0, -s],
            [0, 1, 0],
            [s, 0, c]
        ])


    @staticmethod
    def Rz(theta):
        """rotation matrix around z-axis
        """
        c = np.cos(theta)
        s = np.sin(theta)
        return np.array([
            [c, s, 0],
            [-s, c, 0],
            [0, 0, 1]
        ])
    
    @staticmethod
    def yaw_misalignment_from_two_points(gps_1, gps_2, gt_1, gt_2):
        # Compute displacement vectors in XY plane
        v_gps = np.array(gps_2[:2]) - np.array(gps_1[:2])
        v_gt = np.array(gt_2[:2]) - np.array(gt_1[:2])

        # Normalize vectors
        v_gps_norm = v_gps / np.linalg.norm(v_gps)
        v_gt_norm = v_gt / np.linalg.norm(v_gt)

        # Compute angle (radians)
        dot = np.clip(np.dot(v_gt_norm, v_gps_norm), -1.0, 1.0)
        cross = np.cross(v_gps_norm, v_gt_norm)  # scalar in 2D
        theta_rad = np.arctan2(cross, dot)

        return theta_rad

    @abc.abstractmethod
    def transform_data(self, fields: TransformationField) -> np.ndarray:
        """By default, given data is transformed into the inertial frame coordinate.

        Args:
            sensor_type (SensorType): _description_
            value (np.ndarray): _description_
            state (State): _description_
            to (SensorType, optional): _description_. Defaults to None.

        Returns:
            np.ndarray: _description_
        """
        pass
        
    @abc.abstractmethod
    def _transform_virtual_imu_data(self, data: SensorDataField) -> np.ndarray:
        """Transform IMU data into virtual IMU coordinate frame."""
        pass

