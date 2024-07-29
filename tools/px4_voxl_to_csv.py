import os
import pandas as pd

ROOT = "../data/UAV"
EXPORT_PATH = os.path.join(ROOT, "v4")

FILE_PATH = {
    "log0000": {
        "voxl": {
            "imu0": os.path.join(ROOT, "log0000/run/mpa/imu0/data.csv"),
            "imu1": os.path.join(ROOT, "log0000/run/mpa/imu1/data.csv"),
            "qvio": os.path.join(ROOT, "log0000/run/mpa/qvio/data.csv"),
        },
        "px4": {
            "imu0": os.path.join(ROOT, "log0000/px4/13_07_41/imu_combined/log0000_px4_imu0_combined.csv"),
            "imu1": os.path.join(ROOT, "log0000/px4/13_07_41/imu_combined/log0000_px4_imu1_combined.csv"),
            "gps": os.path.join(ROOT, "log0000/px4/13_07_41/13_07_41_sensor_gps_0.csv"),
            "vo": os.path.join(ROOT, "log0000/px4/13_07_41/13_07_41_vehicle_visual_odometry_0.csv"),
            "vehicle_odometry": os.path.join(ROOT, "log0000/px4/13_07_41/13_07_41_vehicle_odometry_0.csv"),
            "imu0_bias": os.path.join(ROOT, "log0000/px4/13_07_41/13_07_41_estimator_sensor_bias_0.csv"),
            "imu1_bias": os.path.join(ROOT, "log0000/px4/13_07_41/13_07_41_estimator_sensor_bias_1.csv")
        },
        "export_filename": "log0000_timestamp_combined.csv",
    },
    "log0001": {
        "voxl": {
            "imu0": os.path.join(ROOT, "log0001/run/mpa/imu0/data.csv"),
            "imu1": os.path.join(ROOT, "log0001/run/mpa/imu1/data.csv"),
            "qvio": os.path.join(ROOT, "log0001/run/mpa/qvio/data.csv"),
        },
        "px4": {
            "imu0": os.path.join(ROOT, "log0001/px4/imu_combined/log0001_px4_imu0_combined.csv"),
            "imu1": os.path.join(ROOT, "log0001/px4/imu_combined/log0001_px4_imu1_combined.csv"),
            "gps": os.path.join(ROOT, "log0001/px4/13_13_57_sensor_gps_0.csv"),
            "vo": os.path.join(ROOT, "log0001/px4/13_13_57_vehicle_visual_odometry_0.csv"),
            "vehicle_odometry": os.path.join(ROOT, "log0001/px4/13_13_57_vehicle_odometry_0.csv"),
            "imu0_bias":os.path.join(ROOT, "log0001/px4/13_13_57_estimator_sensor_bias_1.csv"),
            "imu1_bias": os.path.join(ROOT, "log0001/px4/13_13_57_sensor_gps_0.csv"),
        },
        "export_filename": "log0001_timestamp_combined.csv",
    },
    "log0003": {
        "voxl": {
            "imu0": os.path.join(ROOT, "log0003/run/mpa/imu0/data.csv"),
            "imu1": os.path.join(ROOT, "log0003/run/mpa/imu1/data.csv"),
            "qvio": os.path.join(ROOT, "log0003/run/mpa/qvio/data.csv"),
        },
        "px4": {
            "imu0":  os.path.join(ROOT, "log0003/px4/imu_combined/log0003_px4_imu0_combined.csv"),
            "imu1":  os.path.join(ROOT, "log0003/px4/imu_combined/log0003_px4_imu1_combined.csv"),
            "gps": os.path.join(ROOT, "log0003/px4/13_16_27_sensor_gps_0.csv"),
            "vo":  os.path.join(ROOT, "log0003/px4/13_16_27_vehicle_visual_odometry_0.csv"),
            "vehicle_odometry": os.path.join(ROOT, "log0003/px4/13_16_27_vehicle_odometry_0.csv"),
            "imu0_bias": os.path.join(ROOT, "log0003/px4/13_16_27_estimator_sensor_bias_0.csv"),
            "imu1_bias": os.path.join(ROOT, "log0003/px4/13_16_27_estimator_sensor_bias_1.csv"),
        },
        "export_filename": "log0003_timestamp_combined.csv",
    },
    "log0005": {
        "voxl": {
            "imu0": os.path.join(ROOT, "log0005/run/mpa/imu0/data.csv"),
            "imu1": os.path.join(ROOT, "log0005/run/mpa/imu1/data.csv"),
            "qvio": os.path.join(ROOT, "log0005/run/mpa/qvio/data.csv"),
        },
        "px4": {
            "imu0":  os.path.join(ROOT, "log0005/px4/imu_combined/log0005_px4_imu0_combined.csv"),
            "imu1":  os.path.join(ROOT, "log0005/px4/imu_combined/log0005_px4_imu1_combined.csv"),
            "gps": os.path.join(ROOT, "log0005/px4/log_145_UnknownDate_sensor_gps_0.csv"),
            "vo":  os.path.join(ROOT, "log0005/px4/log_145_UnknownDate_vehicle_visual_odometry_0.csv"),
            "vehicle_odometry": os.path.join(ROOT, "log0005/px4/log_145_UnknownDate_vehicle_odometry_0.csv"),
            "imu0_bias": os.path.join(ROOT, "log0005/px4/log_145_UnknownDate_estimator_sensor_bias_0.csv"),
            "imu1_bias": os.path.join(ROOT, "log0005/px4/log_145_UnknownDate_estimator_sensor_bias_1.csv"),
        },
        "export_filename": "log0005_timestamp_combined.csv",
    },
}

# PX4_ROOT_PATH = "log_91_2024-5-29-15-52-22-csv/"
# VOXL_ROOT_PATH = "2024-05-29-log0005/run/mpa/"

# PX4_IMU0_ACCEL_FILE_PATH = PX4_ROOT_PATH + "log_91_2024-5-29-15-52-22_sensor_accel_0.csv"
# PX4_IMU0_GYRO_FILE_PATH = PX4_ROOT_PATH + "log_91_2024-5-29-15-52-22_sensor_gyro_0.csv"
# PX4_IMU1_ACCEL_FILE_PATH = PX4_ROOT_PATH + "log_91_2024-5-29-15-52-22_sensor_accel_1.csv"
# PX4_IMU1_GYRO_FILE_PATH = PX4_ROOT_PATH + "log_91_2024-5-29-15-52-22_sensor_gyro_1.csv"
# PX4_GPS_FILE_PATH = PX4_ROOT_PATH + "log_91_2024-5-29-15-52-22_sensor_gps_0.csv"
# VOXL_IMU0_FILE_PATH = VOXL_ROOT_PATH + "imu0/data.csv"
# VOXL_IMU1_FILE_PATH = VOXL_ROOT_PATH + "imu1/data.csv"
# VOXL_QVIO_FILE_PATH = VOXL_ROOT_PATH + "qvio/data.csv"

def read_px4_csv(file_path, device):
    data = pd.read_csv(file_path)
    return pd.DataFrame({
        "timestamp": data["timestamp"].values,
        "device": device,
    })

def read_voxl_csv(file_path, device):
    data = pd.read_csv(file_path)
    timestamps = (data["timestamp(ns)"].values / 1000).astype(int)
    return pd.DataFrame({
        "timestamp": timestamps,
        "device": device,
    })
    

if __name__ == "__main__":
    
    sequence = "log0005"
    
    file_path = FILE_PATH[sequence]
    
    px4 = file_path["px4"]
    px4_imu0 = read_px4_csv(file_path=px4["imu0"], device="px4_imu0")
    px4_imu1 = read_px4_csv(file_path=px4["imu1"], device="px4_imu1")
    px4_gps = read_px4_csv(file_path=px4["gps"], device="px4_gps")
    px4_vo = read_px4_csv(file_path=px4["vo"], device="px4_vo")
    px4_vehicle_odom = read_px4_csv(file_path=px4["vehicle_odometry"], device="px4_vehicle_odom")
    px4_imu0_bias = read_px4_csv(file_path=px4["imu0_bias"], device="px4_imu0_bias")
    px4_imu1_bias = read_px4_csv(file_path=px4["imu1_bias"], device="px4_imu1_bias")
    
    voxl = file_path["voxl"]
    voxl_imu0 = read_voxl_csv(file_path=voxl["imu0"], device="voxl_imu0")
    voxl_imu1 = read_voxl_csv(file_path=voxl["imu1"], device="voxl_imu1")
    voxl_qvio = read_voxl_csv(file_path=voxl["qvio"], device="voxl_vo")

    # data_px4_imu0_accel = read_px4_csv(PX4_IMU0_ACCEL_FILE_PATH, "acc", "px4_imu_0")
    # data_px4_imu1_accel = read_px4_csv(PX4_IMU1_ACCEL_FILE_PATH, "acc", "px4_imu_1")
    # data_px4_imu0_gyro = read_px4_csv(PX4_IMU0_GYRO_FILE_PATH, "gyro", "px4_imu_0")
    # data_px4_imu1_gyro = read_px4_csv(PX4_IMU1_GYRO_FILE_PATH, "gyro", "px4_imu_1")
    # data_px4_gps = read_px4_csv(PX4_GPS_FILE_PATH, "gps", "px4_gps")

    # data_voxl_imu0_accel = read_voxl_csv(VOXL_IMU0_FILE_PATH, "acc", "voxl_imu_0")
    # data_voxl_imu1_accel = read_voxl_csv(VOXL_IMU1_FILE_PATH, "acc", "voxl_imu_1")
    # data_voxl_imu0_gyr = read_voxl_csv(VOXL_IMU0_FILE_PATH, "gyr", "voxl_imu_0")
    # data_voxl_imu1_gyr = read_voxl_csv(VOXL_IMU1_FILE_PATH, "gyr", "voxl_imu_1")
    # data_voxl_vio = read_voxl_csv(VOXL_QVIO_FILE_PATH, "vo", "voxl_vo")

    # all_data = pd.concat([
    #     data_px4_imu0_accel, data_px4_imu1_accel, data_px4_imu0_gyro, data_px4_imu1_gyro, data_px4_gps,
    #     data_voxl_imu0_accel, data_voxl_imu1_accel, data_voxl_imu0_gyr, data_voxl_imu1_gyr, data_voxl_vio
    # ])
    
    all_data = pd.concat([
        px4_imu0, px4_imu1, px4_gps, px4_vo, px4_vehicle_odom, px4_imu0_bias,
        px4_imu1_bias, voxl_imu0, voxl_imu1, voxl_qvio
    ])

    all_data = all_data.sort_values(by="timestamp")

    output_file = os.path.join(EXPORT_PATH, f"combined_{sequence}.csv")
    if not os.path.exists(EXPORT_PATH):
        os.mkdir(EXPORT_PATH)
        
    all_data.to_csv(output_file, index=False)

    print(f"Combined CSV file created: {output_file}")
