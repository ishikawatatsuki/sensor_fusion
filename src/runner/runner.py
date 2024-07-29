import sys
if __name__ == "__main__":
    sys.path.append('../../src')
if __name__ == "__main__":
    from serial_manager.serial_message import get_imu_data
    from src.kalman_filters.experiments.federated_unscented_kalman_filter import FederatedUnscentedKalmanFilter, IMU
    from utils.repeated_timer import RepeatedTimer
else:
    from ..serial_manager.serial_message import get_imu_data
    from ..kalman_filters.experiments.federated_unscented_kalman_filter import FederatedUnscentedKalmanFilter, IMU
    from ..utils.repeated_timer import RepeatedTimer
import numpy as np
import matplotlib.pyplot as plt
import threading

PORT1 = "/dev/cu.usbmodem1201"
PORT2 = "/dev/cu.usbmodem1401"
DEFAULT_BAUD_RATE = 115200

IMU_LABEL1 = "imu1"
IMU_LABEL2 = "imu2"

config = {
  "port1": PORT1,
  "port2": PORT2,
  "baud_rate": DEFAULT_BAUD_RATE,
}

T_imu2_to_imu1_matrix = np.array([
  [1., 0., 0., 0.],
  [0., 1., 0., 0.],
  [0., 0., 1., 0.],
  [0., 0., 0., 1.]
])

def run_prediction(fukf):
  for data in get_imu_data(config=config):
    if data["value"].shape[0] == 6:
      d = data["value"]
      if data["label"] == IMU_LABEL2:
        
        a = T_imu2_to_imu1_matrix @ np.concatenate([d[:3], np.ones(1)]).reshape(-1, 1)
        w = T_imu2_to_imu1_matrix @ np.concatenate([d[3:], np.ones(1)]).reshape(-1, 1)
        d = np.concatenate([a[:3], w[:3]]).flatten()
      fukf.local_prediction_step(data["label"], z=d)

def runner():
  q = np.ones(15) * 0.01
  r = np.ones(6) * 0.01
  imu1 = IMU(IMU_LABEL1, q=q, r=r)
  imu2 = IMU(IMU_LABEL2, q=q, r=r)
  imus = [imu1, imu2]
  fukf = FederatedUnscentedKalmanFilter(imu_list=imus, r=r, q=q)
  
  prediction_thread = threading.Thread(target=run_prediction, args=[fukf])
  prediction_thread.start()
  scheduled_job = RepeatedTimer(0.1, fukf.correction_step) 
  
  ax = plt.figure().add_subplot(projection='3d')
  ax.set_xlabel('$X$', fontsize=14)
  ax.set_ylabel('$Y$', fontsize=14)
  ax.set_zlabel('$Z$', fontsize=14)
  
  ax.set_xlim([-5, 5])
  ax.set_ylim([-5, 5])
  x, y, z = fukf.imus[IMU_LABEL1].filter.get_position()
  ax.plot(x, y, z, label='IMU1', color='blue')
  
  x, y, z = fukf.imus[IMU_LABEL2].filter.get_position()
  ax.plot(x, y, z, label='IMU2', lw=2, color='red')
  ax.legend()  
  
  try:
    while True:
      # x, y, z = fukf.get_master_position()
      # ax.plot(x, y, z, label='Fused angular velocity', color='black')
      
      x, y, z = fukf.imus[IMU_LABEL1].filter.get_position()
      ax.plot(x, y, z, label='IMU1', color='blue')
      
      x, y, z = fukf.imus[IMU_LABEL2].filter.get_position()
      ax.plot(x, y, z, label='IMU2', lw=2, color='red')
      plt.pause(interval=0.5)
  except Exception as e:
    print("Runner exited with an error: ", e)
  finally:
    print("Closing the scheduled task.")
    scheduled_job.stop()
    prediction_thread.join()

if __name__ == "__main__":
  print("Start listening IMU data")
  runner()