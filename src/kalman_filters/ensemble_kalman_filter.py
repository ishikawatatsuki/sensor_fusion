import os
import sys
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from base_filter import BaseFilter
from custom_types import SensorType    
from config import FilterConfig, DatasetConfig
from interfaces import State, MotionModel, Pose


class EnsembleKalmanFilter(BaseFilter):

    def __init__(
            self, 
            config: FilterConfig,
            dataset_config=DatasetConfig, 
            *args,
            **kwargs
        ):
        super().__init__(config=config, dataset_config=dataset_config, *args, **kwargs)
        
        self.ensemble_size = self._get_params(params=self.config.params, key="ensemble_size", default_value=1024)
        
        x = self.x.get_state_vector()
        self.x_dim = x.shape[0]
        self.samples = self._generate_ensembles(mean=x)

    def _generate_ensembles(self, mean: np.ndarray):
        return np.random.multivariate_normal(
            mean=mean.reshape(-1), 
            cov=self.P,
            size=self.ensemble_size
            )

    def kinematics_motion_model(self, u: np.ndarray, dt: float, Q: np.ndarray):
        
        p = self.samples[:, :3]
        v = self.samples[:, 3:6]
        q = self.samples[:, 6:]
        a = u[:3]
        w = u[3:]
        a = a.reshape(-1, 1)
        w = w.reshape(-1, 1)
        R = np.array([self.x.get_rotation_matrix(q_) for q_ in q]) #Nx3x3
        omega = self.get_quaternion_update_matrix(w) 
        norm_w = self.compute_norm_w(w)

        A = np.cos(norm_w*dt/2) * np.eye(4)
        B = (1/norm_w)*np.sin(norm_w*dt/2) * omega

        acc_val = (R @ a - self.g)
        acc_val = self.correct_acceleration(acc_val=acc_val, q=q)
        acc_val_reshaped = acc_val.reshape(acc_val.shape[0], acc_val.shape[1])
        p_k = p + v * dt + acc_val_reshaped*dt**2 / 2 # Nx3
        v_k = v + acc_val_reshaped * dt # Nx3
        q_k = (np.array(A + B) @ q.T).T # Nx4
        q_k = np.array([q_ / np.linalg.norm(q_) if np.linalg.norm(q_) > 0 else q_  for q_ in q_k])

        process_noise_cov = np.random.multivariate_normal(
                                mean=np.zeros(self.x_dim), 
                                cov=Q, 
                                size=self.ensemble_size)
        
        self.samples = np.concatenate([
            p_k,
            v_k,
            q_k,
        ], axis=1) + process_noise_cov # Nx10
        
        x = np.mean(self.samples, axis=0)
        self.x = State.get_new_state_from_array(x)
        
        
    def velocity_motion_model(self, u: np.ndarray, dt: float, Q: np.ndarray):
        """ 
            move according to control input u (heading change, velocity) with noise std
            u: control input vector
            dt: delta time
            Q: process noise matrix
        """
        p = self.samples[:, :3]
        v = self.samples[:, 3:6]
        q = self.samples[:, 6:10]
        
        a = u[:3]
        w = u[3:]
        wx, _, wz = w
        a = a.reshape(-1, 1)
        w = w.reshape(-1, 1)
        
        omega = self.get_quaternion_update_matrix(w)
        norm_w = self.compute_norm_w(w)
        phi, _, psi = np.array([self.get_euler_angle_from_quaternion(q_row.reshape(-1, 1)) for q_row in q]).T
        R = np.array([self.x.get_rotation_matrix(q_) for q_ in q])

        A = np.cos(norm_w*dt/2) * np.eye(4)
        B = (1/norm_w)*np.sin(norm_w*dt/2) * omega
        
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
        q_k = (np.array(A + B) @ q.T).T # Nx4
        q_k = np.array([q_ / np.linalg.norm(q_) if np.linalg.norm(q_) > 0 else q_  for q_ in q_k])
        
        process_noise = np.random.multivariate_normal(
            mean=np.zeros(self.x_dim), 
            cov=Q, 
            size=self.ensemble_size
        )
        
        self.samples = np.concatenate([
            p_k,
            v_k,
            q_k
        ], axis=1) + process_noise
        
        x = np.mean(self.samples, axis=0)
        self.x = State.get_new_state_from_array(x)
        
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
        mask = self.get_innovation_mask(sensor_type=sensor_type, z_dim=z_dim)
        
        mean = np.mean(self.samples, axis=0)
        P = np.zeros((self.x_dim, self.x_dim))
        for sample in self.samples:
            x_var = (sample - mean).reshape(-1, 1)
            P += x_var @ x_var.T
        
        P /= (self.ensemble_size - 1)
        K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
        
        samples = self.samples @ H.T
        measurement_noise = np.random.multivariate_normal(
                                mean=np.zeros(z_dim), 
                                cov=R, size=self.ensemble_size)
        z = z.reshape(1, -1) + measurement_noise # N x m
        residuals = (z - samples) 
        innovation = residuals @ K.T
        innovation *= mask
        self.samples += innovation
        
        x = np.mean(self.samples, axis=0)
        self.x = State.get_new_state_from_array(x)
        
        self.innovations.append(np.sum(np.average(residuals, axis=1)))
    
    
    def get_current_estimate(self) -> Pose:
        # NOTE: Overwrite parent's method 
        x = np.mean(self.samples, axis=0)
        state = State.get_new_state_from_array(x)
        return Pose.from_state(state=state)
    
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
        kf: EnsembleKalmanFilter,
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
            
    type = "kitti"
    if type == "uav":
        dataset_config = DatasetConfig(type='uav', mode='stream', root_path='../../data/UAV', variant="log0001", sensors=[UAV_SensorType.VOXL_IMU0, UAV_SensorType.PX4_MAG, UAV_SensorType.PX4_VO])
        dataset = UAVDataset(
            config=dataset_config,
            uav_sensor_path="../dataset/uav_sensor_path.yaml"
        )
    else:
        dataset_config = DatasetConfig(type='kitti', mode='stream', root_path='../../data/KITTI', variant="0033", sensors=[KITTI_SensorType.OXTS_IMU, KITTI_SensorType.OXTS_GPS, KITTI_SensorType.KITTI_STEREO])
        dataset = KITTIDataset(
            config=dataset_config
        )
        
    config = FilterConfig(type='enkf', dimension=2, motion_model='kinematics', noise_type=False, params={
        'ensemble_size': 512
    })

    inital_state = dataset.get_initial_state(config.motion_model, filter_type=config.type)
    
    ukf = EnsembleKalmanFilter(
        config=config, 
        x=inital_state.x, 
        P=inital_state.P, 
        q=inital_state.q
    )
    
    _runner(
        kf=ukf,
        dataset=dataset
    )