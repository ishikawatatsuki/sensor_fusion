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
    
def get_measurement_noise(**params):
  return np.eye(6) * np.array([
      params['position_x'], 
      params['position_y'], 
      params['position_z'], 
      params['velocity_x'], 
      params['velocity_y'], 
      params['velocity_z']
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
                R = get_measurement_noise(**params)
                sensor = {
                    'z': geo_transformer.transform_data(
                        sensor_type=sensor_data.type,
                        value=sensor_data.data.z,
                        state=kf.x
                    ),
                    'R': R,
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
  # Real(10**-5, 10**0, "log-uniform", name='acc_x'),
  # Real(10**-5, 10**0, "log-uniform", name='acc_y'),
  # Real(10**-5, 10**0, "log-uniform", name='acc_z'),
  # Real(10**-5, 10**0, "log-uniform", name='ang_x'),
  # Real(10**-5, 10**0, "log-uniform", name='ang_y'),
  # Real(10**-5, 10**0, "log-uniform", name='ang_z'),
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
  
  Real(10**-2, 10**3, "log-uniform", name='position_x'),
  Real(10**-2, 10**3, "log-uniform", name='position_y'),
  Real(10**-2, 10**3, "log-uniform", name='position_z'),
  Real(10**-2, 10**3, "log-uniform", name='velocity_x'),
  Real(10**-2, 10**3, "log-uniform", name='velocity_y'),
  Real(10**-2, 10**3, "log-uniform", name='velocity_z'),
]



# Define the objective function to minimize RMSE



if __name__ == "__main__":
  # Run Bayesian optimization
  from tqdm import tqdm
  filter_names = ["ekf", "ukf", "pf", "enkf", "ckf"]
  motion_models = ["kinematics", "velocity"]
  
  for filter_name in tqdm(filter_names):
      with open("./optimized_params.txt", "+a") as f:
        f.write(f"{filter_name}:\n")
        for motion_model in motion_models:
            @use_named_args(param_space)
            def objective_function(**params):
                config_filepath = "./config_kitti.yaml"
                
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
  

  
# EKF (kinematics): [0.014670055193985593, 631.948971727927, 0.01, 0.01021575305098594, 0.010605333562104052, 999.4641280195813, 1e-05, 0.9630196247322559, 1.0025461351553265e-05, 0.9891984513705958, 
# 0.011747232332638073, 0.011817042276099907, 564.3277851925918, 0.01047083165717207, 0.013867349410388866, 0.04990715905574493] -> 4.477640485425052
# EKF (velocity): [0.21934366582675052, 0.01, 0.10974155002889978, 1000.0, 0.05152543370345932, 0.018159718279698138, 0.0022959676077731364, 0.014226776664917652, 1.0, 0.046776278788683255,
# 1.9805680136269788, 0.5251240948577188, 0.20541097042692433, 491.85007021002355, 228.08179551373112, 0.015388810318609232] -> 6.41065196582601

# UKF (kinematics):  [0.01, 0.35934711507177725, 4.295094490583036, 1000.0, 92.16309376575647, 0.0304509607549078, 0.27123049463264315, 0.0002292663598958985, 1.0, 0.05470658333660313, 
# 0.4215637615991426, 0.11948187272794387, 26.71110156481842, 780.8717093093309, 81.3332631033559, 1000.0]  -> 5.971483626390288
# UKF (velocity): [0.01, 640.672037290966, 0.01, 720.0539139781827, 0.01, 12.396406769138714, 0.16909698813668775, 1.0, 1.0, 1.0, 
# 0.01, 10.912617286290844, 0.01, 0.01, 0.043410270380834094, 1000.0] -> 6.2569574833360795

# PF (kinematics): [12.848290749664129, 59.43317050790529, 0.0480447564393259, 102.09517198456143, 11.734385300621323, 157.26829365731734, 0.020249249993994065, 0.0007047857628603913, 0.00028015474078105793, 0.0023508153574800772, 
# 0.06375906529331718, 2.875535138088875, 723.5991590541895, 0.9373645815656223, 1.153535800292531, 26.970474138198533] -> 6.383465431379847
# PF (velocity): [1000.0, 17.806786631095797, 3.6477191541486844, 1.1435532949849434, 9.406789363628478, 5.986441022075565, 0.17590700684262384, 0.0020504032355239416, 0.001402183733422435, 0.001654978260057851, 
# 1.5104159813733657, 4.400910915668542, 3.330527868090535, 290.8027794044911, 40.54470039635388, 3.855209440255587] -> 6.660868462098347

# EnKF (kinematics): [0.01, 0.01, 0.01, 1000.0, 0.18905576092966714, 9.814662471189816, 0.12794194771485246, 1.0, 1.0, 1.0, 
# 0.01, 2.5168794191306247, 1000.0, 0.01, 1000.0, 0.029119194870239146] -> 4.953830598159648
# EnKF (velocity): [0.18773059228382463, 121.17107002715927, 44.20141072281373, 199.4040274580971, 1000.0, 0.01, 1e-05, 0.005413748973176144, 0.9999303768614346, 0.0009071608145668149, 
# 7.128773659040808, 0.11148232016860647, 0.014872826120798096, 0.01, 1000.0, 181.89341498804532] -> 5.969078608754844

# CKF (kinematics): [1.749565184443932, 0.6997813072187518, 0.07896636430023458, 3.121210620258517, 184.8804925896857, 0.2877408191863357, 4.1233214862835026e-05, 0.00179617985659767, 0.001751876436881199, 0.1625915112401301, 
# 0.5300251324235535, 51.89818957252114, 1000.0, 284.5809507954432, 0.8051644656076546, 1000.0] ->5.010726424873242
# CKF (velocity):  [3.0731070838227534, 1000.0, 241.05596460270525, 1000.0, 596.7573190301572, 0.01, 1e-05, 1e-05, 1e-05, 0.41889080609899676, 
# 0.05596880168740783, 7.067552517805686, 0.9959565576581076, 153.158983761175, 1000.0, 0.37815068277533076] -> 5.960954913684584