import os
import pandas as pd

root = "../data/UAV/"
IMU_FILE_PATH = {
    "log0000": {
      "root": os.path.join(root, "log0000/px4/13_07_41"),
      "imu0_gyro": "13_07_41_sensor_gyro_0.csv",
      "imu1_gyro": "13_07_41_sensor_gyro_1.csv",
      "imu0_acc": "13_07_41_sensor_accel_0.csv",
      "imu1_acc": "13_07_41_sensor_accel_1.csv",
      "imu0_export_filename": "log0000_px4_imu0_combined.csv",
      "imu1_export_filename": "log0000_px4_imu1_combined.csv",
    },
    "log0001": {
      "root": os.path.join(root, "log0001/px4"),
      "imu0_gyro": "13_13_57_sensor_gyro_0.csv",
      "imu1_gyro": "13_13_57_sensor_gyro_1.csv",
      "imu0_acc": "13_13_57_sensor_accel_0.csv",
      "imu1_acc": "13_13_57_sensor_accel_1.csv",
      "imu0_export_filename": "log0001_px4_imu0_combined.csv",
      "imu1_export_filename": "log0001_px4_imu1_combined.csv",
    },
    "log0003": {
      "root": os.path.join(root, "log0003/px4"),
      "imu0_gyro": "13_16_27_sensor_gyro_0.csv",
      "imu1_gyro": "13_16_27_sensor_gyro_1.csv",
      "imu0_acc": "13_16_27_sensor_accel_0.csv",
      "imu1_acc": "13_16_27_sensor_accel_1.csv",
      "imu0_export_filename": "log0003_px4_imu0_combined.csv",
      "imu1_export_filename": "log0003_px4_imu1_combined.csv",
    },
    "log0004": {
      "root": os.path.join(root, "log0004/px4"),
      "imu0_gyro": "log_144_2024-7-12-13-04-36_sensor_gyro_0.csv",
      "imu1_gyro": "log_144_2024-7-12-13-04-36_sensor_gyro_1.csv",
      "imu0_acc": "log_144_2024-7-12-13-04-36_sensor_accel_0.csv",
      "imu1_acc": "log_144_2024-7-12-13-04-36_sensor_accel_1.csv",
      "imu0_export_filename": "log0004_px4_imu0_combined.csv",
      "imu1_export_filename": "log0004_px4_imu1_combined.csv",
    },
    "log0005": {
      "root": os.path.join(root, "log0005/px4"),
      "imu0_gyro": "log_145_UnknownDate_sensor_gyro_0.csv",
      "imu1_gyro": "log_145_UnknownDate_sensor_gyro_1.csv",
      "imu0_acc": "log_145_UnknownDate_sensor_accel_0.csv",
      "imu1_acc": "log_145_UnknownDate_sensor_accel_1.csv",
      "imu0_export_filename": "log0005_px4_imu0_combined.csv",
      "imu1_export_filename": "log0005_px4_imu1_combined.csv",
    },
    
}

columns = ['timestamp_x','device_id_x', 'x_x', 'y_x', 'z_x', 'x_y', 'y_y', 'z_y', 'temperature_x']
rename_dict = {
    'timestamp_x': 'timestamp',
    'device_id_x': 'device_id',
    'x_x': 'AX(m/s2)',
    'y_x': 'AY(m/s2)',
    'z_x': 'AZ(m/s2)',
    'x_y': 'GX(rad/s)', 
    'y_y': 'GY(rad/s)', 
    'z_y': 'GZ(rad/s)',
    'temperature_x': 'T(C)'
}

def main(path):
  imu0_gyro = pd.read_csv(os.path.join(path["root"], path["imu0_gyro"]))
  imu0_acc = pd.read_csv(os.path.join(path["root"], path["imu0_acc"]))
  imu1_gyro = pd.read_csv(os.path.join(path["root"], path["imu1_gyro"]))
  imu1_acc = pd.read_csv(os.path.join(path["root"], path["imu1_acc"]))
  
  
  imu0 = pd.merge(imu0_acc, imu0_gyro, on="timestamp_sample")
  imu0 = imu0[columns].rename(columns=rename_dict)
  
  imu1 = pd.merge(imu1_acc, imu1_gyro, on="timestamp_sample")
  imu1 = imu1[columns].rename(columns=rename_dict)
  
  print(imu0_gyro.values.shape)
  print(imu0_acc.values.shape)
  print(imu0.values.shape)
  
  print("-----")
  
  print(imu1_gyro.values.shape)
  print(imu1_acc.values.shape)
  print(imu1.values.shape)

  export_dir = os.path.join(path["root"], "imu_combined")
  if not os.path.exists(export_dir):
    os.mkdir(export_dir)
    
  imu0.to_csv(os.path.join(export_dir, path["imu0_export_filename"]))
  imu1.to_csv(os.path.join(export_dir, path["imu1_export_filename"]))
  
  
if __name__ == "__main__":
  path = IMU_FILE_PATH["log0005"]
  main(path)