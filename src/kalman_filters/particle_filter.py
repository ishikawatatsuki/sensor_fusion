import os
import sys
import logging
import numpy as np
from enum import Enum, auto
from scipy.stats import multivariate_normal

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from base_filter import BaseFilter
from custom_types import SensorType    
from config import FilterConfig, DatasetConfig
from interfaces import State, MotionModel, Pose

from filterpy.monte_carlo import (
    multinomial_resample, residual_resample, systematic_resample, stratified_resample
)


logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s > %(message)s', 
                        datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG)


class ResamplingAlgorithms(Enum):
    MULTINOMIAL = auto()
    RESIDUAL = auto()
    STRATIFIED = auto()
    SYSTEMATIC = auto()
    
    @staticmethod
    def get_enum_name_list():
        return [s.lower() for s in list(ResamplingAlgorithms.__members__.keys())]
    
    @classmethod
    def get_resampling_algorithm_from_str(cls, sensor_str: str):
        s = sensor_str.lower()
        try: 
            index = ResamplingAlgorithms.get_enum_name_list().index(s)
            return cls(index + 1)
        except:
            return None

class ParticleFilter(BaseFilter):
    def __init__(
            self, 
            config: FilterConfig,
            dataset_config=DatasetConfig, 
            *args,
            **kwargs
        ):
        super().__init__(config=config, dataset_config=dataset_config, *args, **kwargs)
        
        self.particle_size = self._get_params(params=self.config.params, key="particle_size", default_value=1024)
        resampling_algorithm = self._get_params(params=self.config.params, key="resampling_algorithm", default_value="multinomial")
        self.scale_for_ess_threshold = self._get_params(params=self.config.params, key="scale_for_ess_threshold", default_value=1.)
        
        
        x = self.x.get_state_vector()
        
        self.particles = self._create_gaussian_particles(mean=x, var=self.P)

        self.weights = np.ones(self.particle_size) / self.particle_size
        self.resampling_algorithm = ResamplingAlgorithms.get_resampling_algorithm_from_str(resampling_algorithm)
    
    def _create_gaussian_particles(self, mean, var):
        return mean.reshape(-1) + np.array([np.random.randn(self.particle_size) for _ in range(var.shape[0])]).T @ var

    def kinematics_motion_model(self, u: np.ndarray, dt: float, Q: np.ndarray):
        """ 
            move according to control input u (heading change, velocity) with noise std
            u: control input vector
            dt: delta time
            Q: process noise matrix
        """
        p = self.particles[:, :3]
        v = self.particles[:, 3:6]
        q = self.particles[:, 6:]
        
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

        process_noise = np.random.multivariate_normal(np.zeros(Q.shape[0]), Q, self.particle_size)
        self.particles = np.concatenate([
            p_k,
            v_k,
            q_k,
        ], axis=1) + process_noise #Nx10
        
        x, _ = self.estimate()
        self.x = State.get_new_state_from_array(x)
        
    def velocity_motion_model(self, u: np.ndarray, dt: float, Q: np.ndarray):
        """ 
            move according to control input u (heading change, velocity) with noise std
            u: control input vector
            dt: delta time
            Q: process noise matrix
        """
        p = self.particles[:, :3]
        v = self.particles[:, 3:6]
        q = self.particles[:, 6:10]
        
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
            np.zeros(Q.shape[0]), 
            Q, 
            self.particle_size
        )
        
        self.particles = np.concatenate([
            p_k,
            v_k,
            q_k
        ], axis=1) + process_noise
        
        x, _ = self.estimate()
        self.x = State.get_new_state_from_array(x)

    def time_update(self, u: np.ndarray, dt: float, Q: np.ndarray):
        predict = self.kinematics_motion_model if self.motion_model is MotionModel.KINEMATICS else\
                    self.velocity_motion_model
        
        predict(u=u, dt=dt, Q=Q)

    def measurement_update(
        self, 
        z: np.ndarray, 
        R: np.ndarray,
        sensor_type: SensorType
        ):
        """ 
            calculate the likelihood p(zk|xk)
            z: measurement
            R: measurement noise covariance
        """
        H = self.get_transition_matrix(sensor_type, z_dim=z.shape[0])
        
        target_distribution = multivariate_normal(mean=z.flatten(), cov=R) 
        measurement_noise = np.random.multivariate_normal(
            np.zeros(R.shape[0]), 
            R, 
            self.particle_size
        )
        
        # residual calculation
        x_, _ = self.estimate()
        z_ = H @ x_
        residual = z - z_
        self.innovations.append(np.sum(residual))
        
        for i, particle in enumerate(self.particles):
            y_hat = H @ particle + measurement_noise[i]
            self.weights[i] = target_distribution.pdf(y_hat)

        self.weights += 1.e-300 # avoiding dividing by zero
        self.weights /= sum(self.weights) # normalize
        
        # Resample when a sensor data is given and is allowed by importance resampling
        if self._allow_resampling():
            self._resample()
            
        x, _ = self.estimate()
        self.x = State.get_new_state_from_array(x)
        
        
    def _allow_resampling(self):
        '''
            Allow resampling when ESS < particle size:
                Effective sample size (ESS)
                When the ESS gets close to zero resulted from many particles having small weight, it indicates particle degeneracy meaning that many particles with small weight estimate the measurement poorly.
                To prevent particle degeneracy, resampling comes into play.
        '''
        def _calculate_ess():
            return 1. / np.sum(np.square(self.weights))
        
        N_eff = _calculate_ess()
        return N_eff < self.particle_size * self.scale_for_ess_threshold

    def estimate(self):
        """ 
            computer posterior of the system by calcurating the weighted average
        """
        pos = self.particles
        mu = np.average(pos, weights=self.weights, axis=0)
        var = np.average((pos - mu)**2, weights=self.weights, axis=0)

        return mu, var

    def _resample_from_index(self, indexes):
        assert len(indexes) == self.particle_size
        
        self.particles = self.particles[indexes]
        self.weights = self.weights[indexes]
        self.weights /= np.sum(self.weights)

    def _resample(self):
        if self.resampling_algorithm is ResamplingAlgorithms.RESIDUAL:
            indexes = residual_resample(self.weights)
        elif self.resampling_algorithm is ResamplingAlgorithms.STRATIFIED:
            indexes = stratified_resample(self.weights)
        elif self.resampling_algorithm is ResamplingAlgorithms.SYSTEMATIC:
            indexes = systematic_resample(self.weights)
        else:
            # ResamplingAlgorithms.MULTINOMIAL
            indexes = multinomial_resample(self.weights)
        
        self._resample_from_index(indexes)

    def get_current_estimate(self) -> Pose:
        # NOTE: Overwrite parent's method 
        x, _ = self.estimate()
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
        kf: ParticleFilter,
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
            
    dataset_config = DatasetConfig(type='uav', mode='stream', root_path='../../data/UAV', variant="log0001", sensors=[UAV_SensorType.VOXL_IMU0, UAV_SensorType.PX4_MAG, UAV_SensorType.PX4_VO])
    dataset = UAVDataset(
        config=dataset_config,
        uav_sensor_path="../dataset/uav_sensor_path.yaml"
    )
    
    config = FilterConfig(type='pf', dimension=2, motion_model='kinematics', noise_type=False, params=None)
    inital_state = dataset.get_initial_state(config.motion_model, filter_type=config.type)

    pf = ParticleFilter(
        config=config, 
        x=inital_state.x, 
        P=inital_state.P, 
        q=inital_state.q
    )
    
    _runner(
        kf=pf,
        dataset=dataset
    )