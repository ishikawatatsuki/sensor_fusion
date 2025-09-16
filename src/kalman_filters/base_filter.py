import os
import sys
import abc
import logging
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt

from ..common import (
    FusionData,
    FilterConfig, 
    HardwareConfig, 
    SensorType,
    FilterType,
    CoordinateSystem,
    IMUSensorErrors,
    GimbalCondition,
    Pose, State, MotionModel, TimeUpdateField, MeasurementUpdateField,
    MeasurementUpdateField,
    DECLINATION_OFFSET_RADIAN_IN_ESTONIA
)

class BaseFilter(abc.ABC):
    mu_x = None
    mu_y = None
    mu_z = None

    errors = []
    
    innovations = []
    
    def __init__(
            self, 
            config: FilterConfig,
            hardware_config: HardwareConfig,
            x: State,
            P: np.ndarray,
            coordinate_system: CoordinateSystem = CoordinateSystem.ENU,
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
        self.g = self._get_gravitational_vector(type=hardware_config.type).reshape(-1, 1)
        
        self.config = config
        self.hardware_config = hardware_config

        self.residual = None
        self.innovation = None
        self.K = None
        self.H = None

        self.dimension = self.config.dimension
        self.innovation_masking = self.config.innovation_masking
        self.motion_model = MotionModel.get_motion_model(self.config.motion_model)
        self.filter_type = FilterType.get_filter_type_from_str(self.config.type)
        assert self.filter_type is not None, "Please specify proper Kalman filter type."
        
        self.predict = self._get_motion_model()

    def _get_motion_model(self):
        
        match (self.motion_model):
            case MotionModel.KINEMATICS:
                return self.kinematics_motion_model
            case MotionModel.VELOCITY:
                return self.velocity_motion_model
            case _:
                raise ValueError(f"No motion model found for {self.motion_model}")

    @abc.abstractmethod
    def kinematics_motion_model(self, dt: float, u: np.ndarray, Q: np.ndarray):
        """Motion model that follows basics of Kinematics equations.

        Args:
            u (np.ndarray): control input that contains IMU measurements (linear acceleration and angular velocity)
            dt (float): delta time
            Q (np.ndarray): Process noise covariance matrix
        """
        pass
    
    @abc.abstractmethod
    def velocity_motion_model(self, dt: float, u: np.ndarray, Q: np.ndarray):
        """Motion model that follows Velocity motion model.

        Args:
            u (np.ndarray): control input that contains IMU measurements (linear acceleration and angular velocity)
            dt (float): delta time
            Q (np.ndarray): Process noise covariance matrix
        """
        pass
    
    # @abc.abstractmethod
    # def drone_kinematics_motion_model(self, dt: float, u: np.ndarray, Q: np.ndarray):
    #     """Drone kinematics equation
    #     This motion model is based on the paper: https://iopscience.iop.org/article/10.1088/1757-899X/270/1/012007/pdf
    #     Args:
    #         u (np.ndarray): Rotation speed(rad/s) of each rotor of a drone
    #         dt (float): Delta time (s)
    #         Q (np.ndarray): Process noise covariance matrix
    #     """
    #     pass
    
    def time_update(self, data: TimeUpdateField):
        """Time update step of Kalman filter
        Args:
            - TimeUpdateField containing:
                u  (np.ndarray): control input
                dt (float): time difference
                Q: (np.ndarray): process noise covariance matrix
        """
        
        self.predict(u=data.u, dt=data.dt, Q=data.Q)
        
    @abc.abstractmethod
    def measurement_update(self, data: MeasurementUpdateField):
        """Measurement update step of Kalman filter
        Args:
            data (MeasurementUpdateField)
                z (np.ndarray): measurement input
                R (np.ndarray): measurement noise covariance matrix
                sensor_type (SensorType): state transformation matrix, which transform state vector x to measurement space
        """
        pass
    
    def _get_gravitational_vector(self, type: str) -> np.ndarray:
        match (type):
            case "kitti":
                logging.info("Setting gravitational vector for KITTI")
                return np.array([0., 0., -9.81])
            case "uav":
                logging.info("Setting gravitational vector for UAV")
                return np.array([0., 0., -9.81])
            case "euroc":
                logging.info("Setting gravitational vector for EuRoC")
                return np.array([0., 0., -9.81])
                # return np.array([-9.81, 0., 0.])
            case _:
                logging.info("Setting a default gravitational vector")
                return np.array([0., 0., 9.81])

    def _get_params(self, params: dict|None, key: str, default_value):
        if params is None:
            return default_value
        return params[key] if params[key] is not None else default_value
        
    def _get_diagonal_matrix(self, vector):
        return np.eye(len(vector)) * np.array(vector) ** 2

    def compute_norm_w(self, w):
        return np.sqrt(np.sum(w**2))
    
    def _gimbal_check(self, q):
        qw, qx, qy, qz = q[:, 0]
        _theta = 2*(qw*qy - qx*qz)
        if _theta > 1:
            _theta = 1
        elif _theta < -1:
            _theta = -1
        theta = np.arcsin(_theta)
        if theta == np.pi/2:
            return GimbalCondition.NOSE_UP
        elif theta == -np.pi/2:
            return GimbalCondition.NOSE_DOWN
        return GimbalCondition.LEVEL

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
        H = np.zeros((3, self.P.shape[0])) # 3 x 16
        H[:, 3:6] = self.x.get_rotation_matrix().T
        return H
    
    def _get_quaternion_update_H(self):
        return np.eye(self.P.shape[0])[6:10, :] # qw, qx, qy, qz coordinate
    
    def _get_position_velocity_update_H(self):
        H = np.zeros((6, self.P.shape[0])) # 6 x 16
        H[:3, :3] = np.eye(3)
        H[3:6, 3:6] = self.x.get_rotation_matrix().T
        return H
    
    def _get_upward_leftward_update_H(self):
        H = np.zeros((2, self.P.shape[0])) # 2 x 16
        H[:, 3:6] = self.x.get_rotation_matrix().T[1:3, :]
        return H

    def _get_angle_update_H(self):
        return np.eye(self.P.shape[0])[6:10, :] # qw, qx, qy, qz
    
    def get_transition_matrix(
            self, 
            sensor_type: SensorType,
            z_dim: int=3
        ) -> np.ndarray:
        """
            returns transition matrix H based on the equation below:
                h(x) = H*x.T
        """
        fusion_fields = self.config.sensors.get(sensor_type, [])
        match(sensor_type.name):
            case SensorType.KITTI_VO.name | SensorType.EuRoC_VO.name:
                H = np.empty((0, self.P.shape[0])) # 3 x 16
                if FusionData.POSITION in fusion_fields:
                    H = np.vstack((H, self._get_position_update_H())) # [I_3x3, 0_3x3, 0_3x4, 0_3x3, 0_3x3]
                if FusionData.LINEAR_VELOCITY in fusion_fields:
                    H = np.vstack((H, self._get_velocity_update_H())) # [0_2x3, 0_2x3, I_2x4, 0_2x3, 0_2x3]
                if FusionData.ORIENTATION in fusion_fields:
                    H = np.vstack((H, self._get_quaternion_update_H())) # [0_4x3, 0_4x3, 0_4x4, 0_4x3, 0_4x3]
                return H
            
            case SensorType.KITTI_UPWARD_LEFTWARD_VELOCITY.name:
                return self._get_upward_leftward_update_H()
            case _:
                # NOTE: all transition matrix for GPS, UWB, any position update is handled by this.
                return self._get_position_update_H()

    def get_innovation_mask(self, sensor_type: SensorType, z_dim: int):
        """Returns innovation mask for the given sensor type."""
        if not self.innovation_masking:
            return np.ones(self.P.shape[0])
        x_dim = self.P.shape[0]
        mask = None
        match(sensor_type.name):
            case SensorType.KITTI_VO.name | SensorType.EuRoC_VO.name:
                if z_dim == 3:
                    mask = np.array([0., 0., 0., 1., 1., 1., 0., 0., 0., 0.])
                mask = np.array([1., 1., 1., 1., 1., 1., 0., 0., 0., 0.])
            case SensorType.KITTI_UPWARD_LEFTWARD_VELOCITY.name:
                mask = np.array([0., 0., 0., 0., 1., 1., 0., 0., 0., 0.])
            case _:
                # NOTE: all transition matrix for GPS, UWB, any position update is handled by this.
                mask = np.array([1., 1., 1., 0., 0., 0., 0., 0., 0., 0.])

        if mask is None:
            raise ValueError(f"Unsupported sensor type: {sensor_type.name}")
        return np.pad(mask, (0, x_dim - mask.shape[0]), 'constant')
        
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
        
    def get_imu_sensor_error(self) -> IMUSensorErrors:
        """Get IMU noise according to the IMU specification.
        """
        size = (3, 1)
        acc_bias = np.random.normal(0, self.hardware_config.imu_config.sigma_accel_bias ** 2, size=size)
        gyro_bias = np.random.normal(0, self.hardware_config.imu_config.sigma_gyro_bias ** 2, size=size)
        acc_noise = np.random.normal(0, self.hardware_config.imu_config.sigma_accel ** 2, size=size)
        gyro_noise = np.random.normal(0, self.hardware_config.imu_config.sigma_gyro ** 2, size=size)

        return IMUSensorErrors(
            acc_bias=acc_bias, 
            gyro_bias=gyro_bias, 
            acc_noise=acc_noise, 
            gyro_noise=gyro_noise
        )

    def correct_velocity(self):
        """Correct velocity based on the current state.
        """
        if self.x.v is None:
            return
        v_norm = np.linalg.norm(self.x.v)
        self.x.v = np.array([0., 0., v_norm]).reshape(-1, 1)

if __name__ == "__main__":
    x = State(p=np.zeros(3), v=np.zeros(3), q=np.array([1., 0., 0., 0.]))
    f = BaseFilter(
        x=x, 
        P=np.eye(10), 
        q=np.zeros(10)
    )