import os
import sys
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from base_filter import BaseFilter
from custom_types import SensorType    
from config import FilterConfig, DatasetConfig
from interfaces import State, MotionModel
from filterpy.kalman import MerweScaledSigmaPoints

np.random.seed(777)

class UnscentedKalmanFilter(BaseFilter):

    sigma_points = None
    sigma_points = None

    def __init__(
        self, 
        config: FilterConfig,
        dataset_config=DatasetConfig, 
        *args,
        **kwargs,
        ):
        super().__init__(config=config, dataset_config=dataset_config, *args, **kwargs)
        
        alpha = self._get_params(params=self.config.params, key="alpha", default_value=1.0)
        beta = self._get_params(params=self.config.params, key="beta", default_value=2.0) 
        kappa = self._get_params(params=self.config.params, key="kappa", default_value=0.0)
        
        x = self.x.get_state_vector()
        self.N = x.shape[0]
        self.points = MerweScaledSigmaPoints(n=self.N, alpha=alpha, beta=beta, kappa=kappa)

    def _compute_sigma_points(self):
        x = self.x.get_state_vector()
        return self.points.sigma_points(x.reshape(-1,), self.P)

    def kinematics_motion_model(self, u: np.ndarray, dt: float, Q: np.ndarray):

        sigma_points = self._compute_sigma_points()
        p = sigma_points[:, :3]
        v = sigma_points[:, 3:6]
        q = sigma_points[:, 6:10]
        a = u[:3]
        w = u[3:]
        a = a.reshape(-1, 1)
        w = w.reshape(-1, 1)
        R = np.array([self.x.get_rotation_matrix(q_) for q_ in q]) #21x3x3
        Omega = self.get_quaternion_update_matrix(w)
        norm_w = self.compute_norm_w(w)

        A = np.cos(norm_w*dt/2) * np.eye(4)
        B = (1/norm_w)*np.sin(norm_w*dt/2) * Omega

        acc_val = (R @ a - self.g)
        acc_val = self.correct_acceleration(acc_val=acc_val, q=q)
        acc_val_reshaped = acc_val.reshape(acc_val.shape[0], acc_val.shape[1])
        p_k = p + v * dt + acc_val_reshaped*dt**2 / 2 # 21x3
        v_k = v + acc_val_reshaped * dt # 21x3
        q_k = (np.array(A + B) @ q.T).T # 21x4
        q_k = np.array([q_ / np.linalg.norm(q_) if np.linalg.norm(q_) > 0 else q_  for q_ in q_k])
        
        self.sigma_points = np.concatenate([
            p_k,
            v_k,
            q_k,
        ], axis=1) # 21x10
        
        x = (self.points.Wm @ self.sigma_points).reshape(-1, 1) # 10x1
        self.x = State.get_new_state_from_array(x)
        
        P = np.zeros((self.N, self.N)) # 10x10
        for i, sigma_point in enumerate(self.sigma_points):
            var = sigma_point.reshape(-1, 1) - x
            P += self.points.Wc[i] * (var @ var.T)
        self.P = P + Q # 10x10 additive process noise

    def velocity_motion_model(self, u: np.ndarray, dt: float, Q: np.ndarray):
        """estimate x and P based on previous stete of x and control input u
        Args:
            u  (numpy.array): control input u
            dt (numpy.array): difference of current time and previous time
            Q  (numpy.array): process noise 
        """
        sigma_points = self._compute_sigma_points()

        p = sigma_points[:, :3]
        v = sigma_points[:, 3:6]
        q = sigma_points[:, 6:10]
        a = u[:3]
        w = u[3:]
        wx, wy, wz = w
        a = a.reshape(-1, 1)
        w = w.reshape(-1, 1)
        
        R = np.array([self.x.get_rotation_matrix(q_) for q_ in q])
        omega = self.get_quaternion_update_matrix(w)
        norm_w = self.compute_norm_w(w)
        
        A = np.cos(norm_w*dt/2) * np.eye(4)
        B = (1/norm_w)*np.sin(norm_w*dt/2) * omega
        
        phi, _, psi = np.array([self.get_euler_angle_from_quaternion(q_row.reshape(-1, 1)) for q_row in q]).T
        
        vf = self.get_forward_velocity(v)
        
        acc_val = (R @ a - self.g)
        # acc_val = self.correct_acceleration(acc_val=acc_val, q=q)
        acc_val_reshaped = acc_val.reshape(acc_val.shape[0], acc_val.shape[1])
        
        
        rx = vf / wx  # turning radius for x axis
        rz = vf / wz  # turning radius for z axis
        dphi = wx * dt
        dpsi = wz * dt
        dpx = - rz * np.sin(psi) + rz * np.sin(psi + dpsi)
        dpy = + rz * np.cos(psi) - rz * np.cos(psi + dpsi)
        dpz = + rx * np.cos(phi) - rx * np.cos(phi + dphi)
        
        dp = np.vstack([dpx, dpy, dpz]).T
        
        p_k = p + dp
        v_k = v + acc_val_reshaped * dt
        q_k = (np.array(A + B) @ q.T).T # 21x4
        q_k = np.array([q_ / np.linalg.norm(q_) if np.linalg.norm(q_) > 0 else q_  for q_ in q_k])

        self.sigma_points = np.concatenate([
            p_k,
            v_k,
            q_k,
        ], axis=1)
        
        x = (self.points.Wm @ self.sigma_points).reshape(-1, 1)
        self.x = State.get_new_state_from_array(x)
        
        P = np.zeros((self.N, self.N)) # 3x3
        for i, sigma_point in enumerate(self.sigma_points):
            var = sigma_point.reshape(-1, 1) - x
            P += self.points.Wc[i] * (var @ var.T)
        self.P = P + Q # 10x10 additive process noise

    
    def time_update(self, u: np.ndarray, dt: float, Q: np.ndarray):
        """
            u: np.ndarray -> control input, assuming IMU input
            dt: int       -> delta time in second 
        """
        predict = self.kinematics_motion_model if self.motion_model is MotionModel.KINEMATICS else\
                    self.velocity_motion_model
        
        predict(u=u, dt=dt, Q=Q)
        
    def measurement_update(
            self, 
            z: np.ndarray, 
            R: np.ndarray,
            sensor_type: SensorType
        ):
        
        z_dim = z.shape[0]
        x = self.x.get_state_vector()
        H = self.get_transition_matrix(sensor_type, z_dim=z_dim)
        mask = self.get_innovation_mask(sensor_type=sensor_type, z_dim=z_dim).reshape(-1, 1)
        
        sigma_points = self._compute_sigma_points()
        y_sigma_points = sigma_points @ H.T
        y_hat = (self.points.Wm @ y_sigma_points).reshape(-1, 1)

        x_dim = sigma_points.shape[1]
        z_dim = y_sigma_points.shape[1]

        # compute covariance matrix for residuals
        P_y = np.zeros((z_dim, z_dim)) # 2x2
        for i, y_sigma_point in enumerate(y_sigma_points):
            var_y = y_sigma_point.reshape(-1, 1) - y_hat
            P_y += self.points.Wc[i] * (var_y @ var_y.T)
        P_y += R # additive measurement noise
        
        # compute cross-covariance matrix 
        P_xy = np.zeros((x_dim, z_dim)) # 10x2
        for idx in range(self.N):
            var_x = sigma_points[idx].reshape(-1, 1) - x
            var_y = y_sigma_points[idx].reshape(-1, 1) - y_hat
            P_xy += self.points.Wc[idx] * (var_x @ var_y.T)
            
        # compute kalman gain
        K = P_xy @ np.linalg.inv(P_y)
        
        # compute residual
        residual = z - y_hat
        innovation = K @ residual
        innovation *= mask
        # update state vector and error covariance matrix
        x = x + innovation
        
        self.x = State.get_new_state_from_array(x)
        self.P = self.P - K @ P_y @ K.T
        
        self.innovations.append(np.sum(residual))


if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), '../dataset'))
    from dataset import (
        UAVDataset,
        KITTIDataset
    )
    from custom_types import (
        UAV_SensorType,
        KITTI_SensorType
    )
    from config import DatasetConfig
    
    import time
    import logging
    
    logger = logging.getLogger(__name__)
    logging.basicConfig(format='%(asctime)s > %(message)s', 
                        datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
    
    def _runner(
        kf: UnscentedKalmanFilter,
        dataset: KITTIDataset|UAVDataset
    ):
        # NOTE: Start loading data
        dataset.start()
        time.sleep(0.5)
        
        try:
            while True:
                if dataset.is_queue_empty():
                    break
                
                sensor_data = dataset.get_sensor_data(kf.x)
                if SensorType.is_stereo_image_data(sensor_data.type):
                # NOTE: enqueue stereo data and process vo estimation
                    ...
                elif SensorType.is_time_update(sensor_data.type):
                # NOTE: process time update step
                    kf.time_update(*sensor_data.data)
                
                elif SensorType.is_measurement_update(sensor_data.type):
                # NOTE: process measurement update step
                    kf.measurement_update(*sensor_data.data)
                
                logger.info(f"[{dataset.output_queue.qsize():05}] time: {sensor_data.timestamp}, sensor: {sensor_data.type}\n")
        
        except Exception as e:
            logger.warning(e)
        finally:
            logger.info("Process finished!")
            dataset.stop()
            
    config = FilterConfig(type='ukf', dimension=2, motion_model='kinematics', noise_type=False, params={
        "alpha": 1e-3,
        "beta": 2.0,
        "kappa": 0.0
    })
    dataset_config = DatasetConfig(type='uav', mode='stream', root_path='../../data/UAV', variant="log0001", sensors=[UAV_SensorType.VOXL_IMU0, UAV_SensorType.PX4_MAG, UAV_SensorType.PX4_VO])
    dataset = UAVDataset(
        config=dataset_config,
        uav_sensor_path="../dataset/uav_sensor_path.yaml"
    )
    inital_state = dataset.get_initial_state(config.motion_model, filter_type=config.type)
    
    ukf = UnscentedKalmanFilter(
        config=config, 
        x=inital_state.x, 
        P=inital_state.P, 
        q=inital_state.q
    )
    
    _runner(
        kf=ukf,
        dataset=dataset
    )