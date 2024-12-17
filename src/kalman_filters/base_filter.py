# references: 
# [1] https://docs.ufpr.br/~danielsantos/ProbabilisticRobotics.pdf

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.dirname(__file__), '../interfaces'))

from custom_types import FilterType, SensorType, CoordinateSystem, DatasetType
from config import FilterConfig, DatasetConfig
from interfaces import Pose, State, MotionModel
from constants import (
    DECLINATION_OFFSET_RADIAN_IN_ESTONIA
)


class BaseFilter:
    mu_x = None
    mu_y = None
    mu_z = None

    errors = []
    
    innovations = []
    
    def __init__(
            self, 
            config: FilterConfig,
            dataset_config: DatasetConfig,
            x: State,
            P: np.ndarray,
            coordinate_system: CoordinateSystem = CoordinateSystem.ENU
        ):
        """ 
        Args:
            x (numpy.array): state to estimate: 
                setup1 or setup2: [px, py, pz, vx, vy, vz, qw, qx, qy, qz] 
                setup3          : [px, py, pz, qw, qx, qy, qz]
            P (numpy.array): state error covariance matrix
            q (numpy.array): process noise vector
        """
        self.x = x
        self.P = P
        
        self.coordinate_system = coordinate_system
        self.g = self._get_gravitational_vector().reshape(-1, 1)
        
        self.config = config
        self.dataset_config = dataset_config
        self.dataset_type = DatasetType.get_type_from_str(self.dataset_config.type)
        self.dimension = self.config.dimension
        self.innovation_masking = self.config.innovation_masking
        self.motion_model = MotionModel.get_motion_model(self.config.motion_model)
        self.filter_type = FilterType.get_filter_type_from_str(self.config.type)
        assert self.filter_type is not None, "Please specify proper Kalman filter type."
        
        
    def _get_gravitational_vector(self) -> np.ndarray:
        match (self.coordinate_system):
            case CoordinateSystem.ENU:
                return np.array([0., 0., 9.81])
            case CoordinateSystem.NED:
                return np.array([0., 0., -9.81])
            case _:
                return np.array([0., 0., 9.81])

    def _get_params(self, params: dict|None, key: str, default_value):
        if params is None:
            return default_value
        return params[key] if params[key] is not None else default_value
        
    def _get_diagonal_matrix(self, vector):
        return np.eye(len(vector)) * np.array(vector) ** 2

    def compute_norm_w(self, w):
        return np.sqrt(np.sum(w**2))
    
    def get_euler_angle_from_quaternion(self, q):
        qw, qx, qy, qz = q[:, 0]
        _theta = 2*(qw*qy - qx*qz)
        if _theta > 1:
            _theta = 1
        elif _theta < -1:
            _theta = -1
        # _theta = -np.pi/2 + 2*np.arctan2(np.sqrt(1+2*(qw*qy - qx*qz)), np.sqrt(1 - 2*(qw*qy - qx*qz)))
        
        phi = np.arctan2(2*(qw*qx + qy*qz), qw**2 - qx**2 - qy**2 + qz**2)
        theta = np.arcsin(_theta)
        psi = np.arctan2(2*(qw*qz + qx*qy), qw**2 + qx**2 - qy**2 - qz**2)
        
        if theta == np.pi/2:
            phi = 0
            psi = -2*np.arctan2(qx, qw)
        elif theta == -np.pi/2:
            phi = 0
            psi = 2*np.arctan2(qx, qw)
        
        return np.array([phi, theta, psi])
    
    def get_rotation_matrix(self, q):
        q0, q1, q2, q3 = q[:, 0]
        # https://ahrs.readthedocs.io/en/latest/filters/ekf.html
        # https://www.iri.upc.edu/people/jsola/JoanSola/objectes/notes/kinematics.pdf
        return np.array([
            [q0**2 + q1**2 - q2**2 - q3**2, 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)],
            [2*(q1*q2 + q0*q3), q0**2 - q1**2 + q2**2 - q3**2, 2*(q2*q3 - q0*q1)],
            [2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), q0**2 - q1**2 - q2**2 + q3**2]
        ])

    def get_quaternion_update_matrix(self, w):
        wx, wy, wz = w[:, 0]
        # https://ahrs.readthedocs.io/en/latest/filters/ekf.html
        # https://www.iri.upc.edu/people/jsola/JoanSola/objectes/notes/kinematics.pdf
        return np.array([ # w, x, y, z
            [0, -wx, -wy, -wz],
            [wx, 0, wz, -wy],
            [wy, -wz, 0, wx],
            [wz, wy, -wx, 0]
        ])

    def get_estimated_trajectory(self):
        return np.concatenate([
            np.array(self.mu_x).reshape(-1, 1), 
            np.array(self.mu_y).reshape(-1, 1), 
            np.array(self.mu_z).reshape(-1, 1)], axis=1)
    
    def visualize_trajectory(self, data, dimension=2, title=None, xlim=None, ylim=None, interval=None):
        if dimension == 2:
            fig, ax1 = plt.subplots(1, 1, figsize=(6, 6))
            xs, ys, _ = data.GPS_measurements_in_meter.T
            ax1.plot(xs, ys, lw=2, label='ground-truth trajectory', color='black')
            xs, ys, _ = data.VO_measurements.T
            ax1.plot(xs, ys, lw=2, label='VO trajectory', color='b')
            ax1.plot(
                self.mu_x, self.mu_y, lw=2, 
                label='estimated trajectory', color='r')
            if title is not None:
                ax1.title.set_text(title)
            if xlim is not None:
                ax1.set_xlim(xlim)
            if ylim is not None:
                ax1.set_ylim(ylim)
            ax1.set_xlabel('X [m]')
            ax1.set_ylabel('Y [m]')
            ax1.legend()
            ax1.grid()
        else:
            fig = plt.figure()
            ax1 = fig.add_subplot(111, projection='3d')
            ax1.set_title("ground-truth trajectory (GPS)")
            
            xs, ys, zs = data.GPS_measurements_in_meter.T
            ax1.plot(xs, ys, zs, label='ground-truth trajectory (GPS)', color='black')
            
            xs, ys, zs = data.VO_measurements.T
            ax1.plot(xs, ys, zs, label='Visual odometry trajectory', color='blue')
            
            ax1.plot(self.mu_x, self.mu_y, self.mu_z, label='Estimated trajectory', color='red')

            ax1.set_xlabel('X [m]', fontsize=14)
            ax1.set_ylabel('Y [m]', fontsize=14)
            ax1.set_zlabel('Z [m]', fontsize=14)

            fig.tight_layout()
            ax1.legend(loc='best', bbox_to_anchor=(1.1, 0., 0.2, 0.9))
        
        if interval is not None:
            plt.pause(interval=interval)

    def plot_error(self):
        plt.plot([i for i in range(len(self.errors))], self.errors, label='Error', color='r')

    def _get_magnetometer_update_yaw_H(self):
        q = self.x.q
        qw, qx, qy, qz = q.reshape(-1)
        
        # compute pitch from the quaternion to check Gimbal lock
        pitch = np.arcsin( 2*(qw*qy - qx*qz) )

        H = np.zeros((1, self.P.shape[0]))
        
        if pitch == np.pi / 2:
            H[0, -4] = 2*qx / (qw**2 + qx**2)
            H[0, -3] = -2*qw / (qw**2 + qx**2)
            return H
        elif pitch == -np.pi / 2:
            H[0, -4] = -2*qx / (qw**2 + qx**2)
            H[0, -3] = 2*qw / (qw**2 + qx**2)
            return H
        else:
            dqw = ( 2*qw*(-2*qw*qz - 2*qx*qy)/((2*qw*qz + 2*qx*qy)**2 + (qw**2 + qx**2 - qy**2 - qz**2)**2) ) +\
                ( (2*qz*(qw**2 + qx**2 - qy**2 - qz**2)) / ( (2*qw*qz + 2*qx*qy)**2 + (qw**2 + qx**2 - qy**2 -qz**2)**2 ))
            dqx = ( 2*qx*(-2*qw*qz - 2*qx*qy) / ( (2*qw*qz + 2*qx*qy)**2 + (qw**2 + qx**2 - qy**2 - qz**2)**2 ) ) +\
                ( (2*qy*(qw**2 + qx**2 - qy**2 - qz**2)) / ( (2*qw*qz + 2*qx*qy)**2 + (qw**2 + qx**2 - qy**2 - qz**2)**2 ))
            dqy = ( 2*qx*(qw**2 + qx**2 - qy**2 - qz**2) / ( (2*qw*qz + 2*qx*qy)**2 + (qw**2 + qx**2 - qy**2 - qz**2)**2 ) ) -\
                ( 2*qy*(-2*qw*qz - 2*qx*qy) / ( (2*qw*qz + 2*qx*qy)**2 + (qw**2 + qx**2 - qy**2 - qz**2)**2 ) )
            dqz = ( 2*qw*(qw**2 + qx**2 - qy**2 - qz**2) / ( (2*qw*qz + 2*qx*qy)**2 + (qw**2 + qx**2 - qy**2 - qz**2)**2 ) ) -\
                ( 2*qz*(-2*qw*qz - 2*qx*qy) / ( (2*qw*qz + 2*qx*qy)**2 + (qw**2 + qx**2 - qy**2 - qz**2)**2 ) )
        
            H[0, -4] = dqw
            H[0, -3] = dqx
            H[0, -2] = dqy
            H[0, -1] = dqz
            return H
    
    def _get_position_update_H(self):
        return np.eye(self.P.shape[0])[:3, :] # x, y, z coordinate
    
    def _get_velocity_update_H(self):
        """Given a condition that, VO estimate failed, then VO can no longer estimate the position of a vehicle but can estimate the velocity from two consecutive frames.

        Returns:
            H (np.ndarray): Measurement space transition matrix H. 
        """
        return np.eye(self.P.shape[0])[3:6, :] # vx, vy, vz
    
    def _get_position_velocity_update_H(self):
        return np.eye(self.P.shape[0])[:6, :] # x, y, z, vx, vy, vz
    
    def _get_upward_leftward_update_H(self):
        return np.eye(self.P.shape[0])[4:6, :] # vy, vz
    
    def get_transition_matrix(
            self, 
            sensor_type: SensorType,
            z_dim: int=3
        ) -> np.ndarray:
        """
            returns transition matrix H based on the equation below:
                h(x) = H*x.T
        """
        match(sensor_type.name):
            case SensorType.PX4_MAG.name:
                return self._get_magnetometer_update_yaw_H()
            case SensorType.KITTI_CUSTOM_VO.name:
                update_H = self._get_velocity_update_H if z_dim == 3 else\
                            self._get_position_velocity_update_H
                return update_H()
            case SensorType.PX4_VO.name:
                update_H = self._get_velocity_update_H if z_dim == 3 else\
                            self._get_position_velocity_update_H
                return update_H()
            case SensorType.KITTI_UPWARD_LEFTWARD_VELOCITY.name | SensorType.VIZTRACK_UPWARD_LEFTWARD_VELOCITY.name:
                return self._get_upward_leftward_update_H()
            case _:
                # NOTE: all transition matrix for GPS, UWB, any position update is handled by this.
                return self._get_position_update_H()

    def get_innovation_mask(self, sensor_type: SensorType, z_dim: int):
        if not self.innovation_masking:
            return np.ones(10)
        
        match(sensor_type.name):
            case SensorType.PX4_MAG.name:
                return np.array([0., 0., 0., 0., 0., 0., 1., 1., 1., 1.])
            case SensorType.KITTI_CUSTOM_VO.name:
                if z_dim == 3:
                    return np.array([0., 0., 0., 1., 1., 1., 0., 0., 0., 0.])
                return np.array([1., 1., 1., 1., 1., 1., 0., 0., 0., 0.])
            case SensorType.PX4_VO.name:
                if z_dim == 3:
                    return np.array([0., 0., 0., 1., 1., 1., 0., 0., 0., 0.])
                return np.array([1., 1., 1., 1., 1., 1., 0., 0., 0., 0.])
            case SensorType.KITTI_UPWARD_LEFTWARD_VELOCITY.name | SensorType.VIZTRACK_UPWARD_LEFTWARD_VELOCITY.name:
                return np.array([0., 0., 0., 0., 1., 1., 0., 0., 0., 0.])
            case _:
                # NOTE: all transition matrix for GPS, UWB, any position update is handled by this.
                return np.array([1., 1., 1., 0., 0., 0., 0., 0., 0., 0.])
        
    def get_current_estimate(self) -> Pose:
        return Pose.from_state(state=self.x)
    
    def get_forward_velocity(self, v) -> np.ndarray:
        if self.filter_type is FilterType.EKF:
            return np.linalg.norm(v) + 1.e-100 #v[0] + 1.e-100
        
        vf = np.linalg.norm(v, axis=1)
        vf += 1.e-100
        return vf
    
    def _correct_acceleration_for_kitti(self, acc_val: np.ndarray, q_list: np.ndarray) -> np.ndarray:
        
        get_R = lambda yaw: np.array([
            [np.cos(yaw), np.sin(yaw)],
            [-np.sin(yaw), np.cos(yaw)] 
        ])
            
        if q_list.shape[0] == 4:
            yaw = self.get_euler_angle_from_quaternion(q_list)[-1]
            R_z = get_R(yaw)
            acc_x, acc_y, acc_z = acc_val.flatten()
            a_xy = np.array([acc_x, acc_y]).reshape(-1, 1)
            a_xy = R_z @ a_xy
            a_xy = a_xy.flatten()
            return np.array([a_xy[0], a_xy[1], acc_z]).reshape(-1, 1)
            
        
        angles = np.array([self.get_euler_angle_from_quaternion(q.reshape(-1, 1)) for q in q_list]) # Nx1
        yaw = angles[:, 2]
        R_z = np.array([get_R(y) for y in yaw]) # Nx2x2
        acc_val = np.squeeze(acc_val)
        acc_x, acc_y, acc_z = acc_val.T
        a_xy = np.vstack([acc_x, acc_y]).T # Nx2
        a_xy = np.array([R_z[i] @ a_xy[i] for i in range(a_xy.shape[0])]) # Nx2
        return np.hstack([a_xy, acc_z.reshape(-1, 1)])
        
    def correct_acceleration(self, acc_val: np.ndarray, q: np.ndarray) -> np.ndarray:
        
        match (self.dataset_type):
            case DatasetType.KITTI | DatasetType.EXPERIMENT:
                return self._correct_acceleration_for_kitti(acc_val=acc_val, q_list=q)
            case _:
                return acc_val
                

if __name__ == "__main__":
    x = State(p=np.zeros(3), v=np.zeros(3), q=np.array([1., 0., 0., 0.]))
    f = BaseFilter(
        x=x, 
        P=np.eye(10), 
        q=np.zeros(10)
    )