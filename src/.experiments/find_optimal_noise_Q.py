import os
import sys
import time
import logging
import numpy as np

from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from sklearn.metrics import mean_absolute_error

sys.path.append(os.path.join(os.path.dirname(__file__), 'kalman_filters'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'interfaces'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'dataset'))
from config import DatasetConfig, FilterConfig
from custom_types import SensorType, FilterType, CoordinateSystem
from dataset import KITTIDataset, TimeUpdateField, MeasurementUpdateField
from config import Config
from kalman_filters import (
  ExtendedKalmanFilter,
  UnscentedKalmanFilter,
  ParticleFilter,
  EnsembleKalmanFilter,
  CubatureKalmanFilter
)
from utils import (
  time_reporter,
  ErrorReporter,
  KITTI_GeometricTransformer,
)

iteration = 0
logger = logging.getLogger(__name__)
logging.basicConfig(format='[%(asctime)s] [%(levelname)5s] > %(message)s (%(filename)s:%(lineno)s)', 
                      datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

def get_process_noise(**params):
  return np.eye(10) * np.array([
    params['pos_x'], 
    params['pos_y'], 
    params['pos_z'], 
    params['vel_x'], 
    params['vel_y'], 
    params['vel_z'],
    params['q_w'], 
    params['q_x'], 
    params['q_y'], 
    params['q_z']
  ])

def _runner(
    kf,
    dataset: KITTIDataset,
    error_reporter,
    geo_transformer,
    **params
):
    # NOTE: Start loading data
    dataset.start()
    time.sleep(0.5)
    
    process_failed = False
    try:
        while True:
            if dataset.is_queue_empty():
                break
            
            sensor_data = dataset.get_sensor_data(current_state=kf.x)
            if SensorType.is_time_update(sensor_data.type):
            # NOTE: process time update step
                Q = get_process_noise(**params)
                param = TimeUpdateField(u=sensor_data.data.u, dt=sensor_data.data.dt, Q=Q)
                kf.time_update(*param)
            
            elif SensorType.is_measurement_update(sensor_data.type):
            # NOTE: process measurement update step
                sensor = {
                    'z': geo_transformer.transform_data(
                        sensor_type=sensor_data.type,
                        value=sensor_data.data.z,
                        state=kf.x
                    ),
                    'R': sensor_data.data.R,
                    'sensor_type': sensor_data.type
                  }
                kf.measurement_update(**sensor)
            
            if SensorType.is_reference_data(sensor_data.type):
              # NOTE: append reference trajectory and estimated trajectory in the corresponding list to compare
              ground_truth = geo_transformer.transform_data(
                    sensor_type=SensorType.GROUND_TRUTH,
                    value=sensor_data.data,
                    state=kf.x
                )
              diverge = error_reporter.set_trajectory(
                estimated=kf.x.p.flatten(),
                expected=ground_truth
              )
              if diverge:
                raise ValueError("Filter diverged.")
            logger.debug(f"[{iteration}][{dataset.output_queue.qsize():05}] time: {sensor_data.timestamp}, sensor: {sensor_data.type}\n")
    
    except Exception as e:
        logger.warning(e)
        process_failed = True
    finally:
        logger.debug("Process finished!")
        dataset.stop()
        
    if process_failed:
      return 1e6
    
    return mean_absolute_error(error_reporter.referenced_trajectory, error_reporter.estimated_trajectory)
  
def get_kalman_filter(filter_config: FilterConfig, dataset: KITTIDataset):
  filter_type = str(filter_config.type).lower()
  initial_state = dataset.get_initial_state()
  
  kwargs = {
    'config': filter_config,
    'x': initial_state.x,
    'P': initial_state.P,
    'coordinate_system': CoordinateSystem.ENU,
  }
  match (filter_type):
    case "ekf":
      logger.debug(f"Configuring Extended Kalman Filter.")
      return ExtendedKalmanFilter(**kwargs)
    case "ukf":
      logger.debug(f"Configuring Unscented Kalman Filter.")
      return UnscentedKalmanFilter(**kwargs)
    case "pf":
      logger.debug(f"Configuring Particle Filter.")
      return ParticleFilter(**kwargs)
    case "enkf":
      logger.debug(f"Configuring Ensemble Kalman Filter.")
      return EnsembleKalmanFilter(**kwargs)
    case "ckf":  
      logger.debug(f"Configuring Cubature Kalman Filter.")
      return CubatureKalmanFilter(**kwargs)
    case _:
      # NOTE: Set EKF as a default filter
      logger.warning(f"dataset: {filter_type} is not found. EKF is used instead.")
      return ExtendedKalmanFilter(**kwargs)
    

param_space  = [
  Real(10**-2, 10**3, "log-uniform", name='pos_x'),
  Real(10**-2, 10**3, "log-uniform", name='pos_y'),
  Real(10**-2, 10**3, "log-uniform", name='pos_z'),
  Real(10**-2, 10**3, "log-uniform", name='vel_x'),
  Real(10**-2, 10**3, "log-uniform", name='vel_y'),
  Real(10**-2, 10**3, "log-uniform", name='vel_z'),
  Real(10**-5, 10**0, "log-uniform", name='q_w'),
  Real(10**-5, 10**0, "log-uniform", name='q_x'),
  Real(10**-5, 10**0, "log-uniform", name='q_y'),
  Real(10**-5, 10**0, "log-uniform", name='q_z'),
]



# Define the objective function to minimize RMSE
# @use_named_args(param_space)
# def objective_function(**params):
#     config_filepath = "./config_kitti.yaml"
    
#     config = Config(
#       config_filepath=config_filepath
#     )
#     _new_filter_config = config.filter._asdict()
#     config.filter = config.filter._replace(**_new_filter_config)
#     error_reporter = ErrorReporter(
#       filter_config=config.filter,
#       dataset_config=config.dataset
#     )
#     geo_transformer = KITTI_GeometricTransformer(dataset_config=config.dataset)
#     dataset = KITTIDataset(config=config.dataset, filter_config=config.filter, geo_transformer=geo_transformer)
#     kf = get_kalman_filter(filter_config=config.filter, dataset=dataset)
#     error = _runner(
#       kf=kf,
#       dataset=dataset,
#       error_reporter=error_reporter,
#       geo_transformer=geo_transformer,
#       **params
#     )
#     logger.info(error)
#     return error



if __name__ == "__main__":
  # Run Bayesian optimization
  from tqdm import tqdm
  filter_names = ["ekf", "ukf", "pf", "enkf", "ckf"]
  motion_models = ["kinematics", "velocity"]
  
  # results = gp_minimize(objective_function, param_space, n_calls=100, random_state=0)
  # logger.info("-"*100)
  # logger.info(f"score: {results.fun}\n")
  # logger.info(f"params: {results.x}\n\n")
  
  for filter_name in tqdm(filter_names):
      with open("./optimized_process_noise_params.txt", "+a") as f:
        f.write(f"{filter_name}:\n")
        for motion_model in motion_models:
            
            @use_named_args(param_space)
            def objective_function(**params):
                config_filepath = "./config_kitti.yaml"
                logger.info(f"Filter: {filter_name.upper()}, Motion model: {motion_model}")
                
                config = Config(
                  config_filepath=config_filepath
                )
                _new_filter_config = config.filter._asdict()
                _new_filter_config['type'] = filter_name
                _new_filter_config['motion_model'] = motion_model
                config.filter = config.filter._replace(**_new_filter_config)
                error_reporter = ErrorReporter(
                  filter_config=config.filter,
                  dataset_config=config.dataset
                )
                geo_transformer = KITTI_GeometricTransformer(dataset_config=config.dataset)
                dataset = KITTIDataset(config=config.dataset, filter_config=config.filter, geo_transformer=geo_transformer)
                kf = get_kalman_filter(filter_config=config.filter, dataset=dataset)
                error = _runner(
                  kf=kf,
                  dataset=dataset,
                  error_reporter=error_reporter,
                  geo_transformer=geo_transformer,
                  **params
                )
                logger.info(error)
                return error

            results = gp_minimize(objective_function, param_space, n_calls=100, random_state=0)
            f.write(f"{motion_model}:\n")
            f.write(f"score: {results.fun}\n")
            f.write(f"params: {results.x}\n\n")
  

