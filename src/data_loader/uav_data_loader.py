import os
import sys
if __name__ == "__main__":
    sys.path.append('../../src')
import yaml
import random
import logging
from enum import Enum
from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from decimal import Decimal, getcontext
from utils import lla_to_ned, get_gyroscope_noise, get_acceleration_noise
from configs.configs import Configs, IMU_Type, SetupEnum, MeasurementDataEnum, SensorType

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s > %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
    
class UAVSensorColumns:    
    
    imu_acc = ['AX(m/s2)', 'AY(m/s2)', 'AZ(m/s2)']
    imu_gyr = ['GX(rad/s)', 'GY(rad/s)', 'GZ(rad/s)']
    imu = ['AX(m/s2)', 'AY(m/s2)', 'AZ(m/s2)', 'GX(rad/s)', 'GY(rad/s)', 'GZ(rad/s)']
    
    px4_imu_bias = ['gyro_bias[0]', 'gyro_bias[1]', 'gyro_bias[2]', 'accel_bias[0]', 'accel_bias[1]', 'accel_bias[2]']
    
    mag = ['x', 'y', 'z']
    
    actuator_outputs = ['output[0]', 'output[1]', 'output[2]', 'output[3]']
    actuator_motors = ['control[0]', 'control[1]', 'control[2]', 'control[3]']
    
    gps_lla_pose = ['lon', 'lat', 'alt']
    gps_ned_pose = ['north', 'east', 'down']
    gps_pose_var = ['eph', 'eph', 'epv'] # https://docs.px4.io/main/en/msg_docs/SensorGps.html
    
    vo_pose = ['position[0]', 'position[1]', 'position[2]']
    vo_pose_var = ['position_variance[0]', 'position_variance[1]', 'position_variance[2]']
    
    voxl_vo_pose = ['T_imu_wrt_vio_x(m)', 'T_imu_wrt_vio_y(m)', 'T_imu_wrt_vio_z(m)']
    
    px4_synced_gps_ned_pose = ['px4_gps_north', 'px4_gps_east', 'px4_gps_down']
    px4_synced_imu0_bias = ['px4_imu0_bias_gyro_bias[0]', 'px4_imu0_bias_gyro_bias[1]', 'px4_imu0_bias_gyro_bias[2]', 'px4_imu0_bias_accel_bias[0]', 'px4_imu0_bias_accel_bias[1]', 'px4_imu0_bias_accel_bias[2]']
    px4_synced_imu1_bias = ['px4_imu1_bias_gyro_bias[0]', 'px4_imu1_bias_gyro_bias[1]', 'px4_imu1_bias_gyro_bias[2]', 'px4_imu1_bias_accel_bias[0]', 'px4_imu1_bias_accel_bias[1]', 'px4_imu1_bias_accel_bias[2]']
    
    voxl_synced_vo_pose = ['voxl_vo_T_imu_wrt_vio_x(m)', 'voxl_vo_T_imu_wrt_vio_y(m)', 'voxl_vo_T_imu_wrt_vio_z(m)']
    voxl_synced_imu0 = ["voxl_imu0_" + col for col in imu]
    voxl_synced_imu1 = ["voxl_imu1_" + col for col in imu]
    px4_synced_imu0 = ["px4_imu0_" + col for col in imu]
    px4_synced_imu1 = ["px4_imu1_" + col for col in imu]
    px4_synced_mag = ["px4_mag_" + col for col in mag]
    
class Sensor:
    def __init__(self, df: pd.DataFrame, column: List[str], synced_column: List[str]):
        self.df = df
        self.column = column
        self.synced_column = synced_column
        self.index = 0
        
class UAV_DataLoader:
    
    time_update = ['voxl_imu0', 'voxl_imu1', 'px4_imu0', 'px4_imu1', 'px4_imu0_bias', 'px4_imu1_bias', 'actuator_motors', 'actuator_outputs']
    measurement_update = ['voxl_vo', 'px4_vo', 'px4_vehicle_odom', 'px4_gps', 'uwb_position']
    
    
    SENSOR_SLIP_THRESHOLD = 2 # 2 seconds
    
    vo_indices = []
    gps_indices = []

    vo_indices = None
    gps_indices = None


    def __init__(
        self,
        root_path,
        sequence_nr="log0001",
        version="v1",
        dataset_filename="config.yaml",
        imu_config_filepath="./imu_config.yaml",
        is_debugging=False,
        regenerate_custom_data=False,
        ):
        
        self.is_debugging = is_debugging
        self.regenerate_custom_data = regenerate_custom_data
        
        self.columns = UAVSensorColumns()
        
        self.version = version
        self.root_path = root_path
        uav_root_path = os.path.join(root_path, "UAV")
        uav_sequence_root_path = os.path.join(root_path, "UAV", sequence_nr)
        
        dataset_filepath = os.path.join(root_path, "UAV", dataset_filename)
        
        sensor_data = None
        with open(dataset_filepath, "r") as f:
            sensor_data = yaml.safe_load(f)
            f.close()
        
        assert sensor_data != None, "Please specify proper config file name"
        
        imu_config = None
        with open(imu_config_filepath, "r") as f:
            imu_config = yaml.safe_load(f)
            f.close()
        assert imu_config != None, "Failed to load IMU configuration"
        
        self.imu_config = imu_config
        
        self.flight_log = sensor_data["sensor_data"][sequence_nr]
        
        self.data_type = self.flight_log["type"]
        
        px4_root_path = os.path.join(uav_sequence_root_path, "px4")
        
        # PX4 data
        px4 = self.flight_log["px4"]
        px4_imu0_path = os.path.join(px4_root_path, "imu_combined", px4["imu0"])
        px4_imu1_path = os.path.join(px4_root_path, "imu_combined", px4["imu1"])
        px4_gps_path = os.path.join(px4_root_path,  px4["gps"])
        px4_vo_path = os.path.join(px4_root_path,  px4["vo"])
        px4_vehicle_odom_path = os.path.join(px4_root_path,  px4["vehicle_odometry"])
        imu0_bias_path = os.path.join(px4_root_path,  px4["imu0_bias"])
        imu1_bias_path = os.path.join(px4_root_path,  px4["imu1_bias"])
        actuator_motors_path = os.path.join(px4_root_path, px4["actuator_motors"])
        actuator_outputs_path = os.path.join(px4_root_path, px4["actuator_outputs"])
        magnetometer_path = os.path.join(px4_root_path, px4["mag"])

        # VOXL data
        voxl_root_path = os.path.join(uav_sequence_root_path, "run/mpa")
        voxl_imu0_path = os.path.join(voxl_root_path, "imu0/data.csv")
        voxl_imu1_path = os.path.join(voxl_root_path, "imu1/data.csv")
        voxl_qvio_path = os.path.join(voxl_root_path, "qvio/data.csv")
        voxl_stereo_path = os.path.join(voxl_root_path, "stereo/data.csv")
        
        self.px4_imu0_df = self._get_combined_imu_data(px4_root_path, px4_imu0_path, imu_name="imu0")
        self.px4_imu1_df = self._get_combined_imu_data(px4_root_path, px4_imu1_path, imu_name="imu1")
        self.px4_gps_df = self._init_gps_data(px4_gps_path=px4_gps_path)
        self.px4_vo_df = self.load_px4_data(px4_vo_path, "PX4 VO")
        self.px4_vehicle_odom_df = self.load_px4_data(px4_vehicle_odom_path, "PX4 Vehicle Odom")
        self.px4_imu0_bias_df = self.load_px4_data(imu0_bias_path, "PX4 IMU0 bias")
        self.px4_imu1_bias_df = self.load_px4_data(imu1_bias_path, "PX4 IMU1 bias")
        self.px4_mag_df = self.load_px4_data(magnetometer_path, "PX4 Magnetometer")
        
        self.px4_actuator_motor_df = self.load_px4_data(actuator_motors_path)
        self.px4_actuator_output_df = self.load_px4_data(actuator_outputs_path)
        if self.data_type == "outdoor":
            self.px4_actuator_motor_df = self.px4_actuator_motor_df[self.px4_actuator_motor_df.columns[:6]]
            self.px4_actuator_output_df = self.px4_actuator_output_df[self.px4_actuator_output_df.columns[:6]]
        

        self.voxl_imu0_df = self.load_voxl_data(voxl_imu0_path, "VOXL IMU0")
        self.voxl_imu1_df = self.load_voxl_data(voxl_imu1_path, "VOXL IMU1")
        self.voxl_qvio_df = self.load_voxl_data(voxl_qvio_path, "VOXL QVIO")
        self.voxl_stereo_df = self.load_voxl_data(voxl_stereo_path, "VOXL stereo")

        
        # generated data
        
        self.vo_df = self._get_vo_data(uav_root_path)
        self.uwb_df = self._get_uwb_data(uav_root_path)
        
        aggregated_file_path = os.path.join(uav_root_path, version)

        self.ref_df = self._get_reference_data(aggregated_file_path)
        self.synced_df = self._get_synced_data(aggregated_file_path)
        
        self.N = self.synced_df.shape[0]
        self.ts = np.cumsum(np.concatenate([np.array([0.]), np.diff(self.synced_df["timestamp"].values / 1e6)]))
        self.dimension = 3
        
        self.voxl_imu_sampling_rate = 1000
        self.px4_imu_sampling_rate = 200
        
        self._calibrate_sensors()
        
    def load_voxl_data(self, path, label=""):
        df = pd.read_csv(path)
        df = df[df.columns[1:]]
        if self.is_debugging:
            freq = df.shape[0] / ((df["timestamp(ns)"].iloc[-1] - df["timestamp(ns)"].iloc[0]) / 1e9)
            print(f"[{label}] Sampling rate: {round(freq, 2)}Hz")
        return df

    def load_px4_data(self, csv_path, label=""):
        if self.data_type == "outdoor":
            df = pd.read_csv(csv_path)
            if df.columns[0] == "1":
                df = df[df.columns[1:]]
            if self.is_debugging:
                freq = df.shape[0] / ((df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]) / 1e9)
                print(f"[{label}] Sampling rate: {round(freq, 2)}Hz")
            return df
        
        logger.info(f"Csv file does not exists: {csv_path}")
        return None
    
    def _get_uwb_data(self, uav_root_path):
        generated_data = self.flight_log["generated"]
        
        if generated_data["uwb"] is not None and self.data_type == "indoor":
            return pd.read_csv(os.path.join(uav_root_path, generated_data["uwb"]))
        
        logger.warning("UWB position data not found.")
        return None
    
    def _get_vo_data(self, uav_root_path):
        generated_data = self.flight_log["generated"]
        
        if generated_data["vo"] is not None:
            return pd.read_csv(os.path.join(uav_root_path, generated_data["vo"]))

        logger.warning("VO estimate data not found.")
        return None
            
    def _init_gps_data(self, px4_gps_path):
        if self.data_type == "indoor":
            logger.info(f"Csv file does not exists: {px4_gps_path}")
            return None
            
        # Load csv data
        df = pd.read_csv(px4_gps_path)
        # Convert integer lla into float lla
        df = UAV_DataLoader.int_lla_to_float_lla(df)
        # Compute North-East-Down coordinate
        if self.is_debugging:
            freq = df.shape[0] / ((df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]) / 1e9)
            print(f"[PX4 GPS] Sampling rate: {round(freq, 2)}Hz")
            
        return UAV_DataLoader.get_ned_coord(df)
    
    def _calibrate_sensors(self):
        """
            Subtract scale and add zero mean white Gaussian noise to each IMU measurement.
        """
        if self.is_debugging:
            print("Start calibrating IMUs")
        
        for i, col in enumerate(self.columns.imu_acc):
            if self.voxl_imu0_df is not None:
                offset = self.imu_config[IMU_Type.IMU0.value]['accelerometer']['offset'][i]
                scale = self.imu_config[IMU_Type.IMU0.value]['accelerometer']['scale'][i]
                noise = get_acceleration_noise(self.imu_config[IMU_Type.IMU0.value]['accelerometer']['noise'], self.voxl_imu_sampling_rate)
                self.voxl_imu0_df[col] = (self.voxl_imu0_df[col] - offset) * scale + noise
                
            if self.voxl_imu1_df is not None:
                offset = self.imu_config[IMU_Type.IMU1.value]['accelerometer']['offset'][i]
                scale = self.imu_config[IMU_Type.IMU1.value]['accelerometer']['scale'][i]
                noise = get_acceleration_noise(self.imu_config[IMU_Type.IMU1.value]['accelerometer']['noise'], self.voxl_imu_sampling_rate)
                self.voxl_imu1_df[col] = (self.voxl_imu1_df[col] + offset) * scale + noise
            
            if self.px4_imu0_df is not None:
                offset = self.imu_config[IMU_Type.IMU2.value]['accelerometer']['offset'][i]
                scale = self.imu_config[IMU_Type.IMU2.value]['accelerometer']['scale'][i]
                noise = get_acceleration_noise(self.imu_config[IMU_Type.IMU2.value]['accelerometer']['noise'], self.px4_imu_sampling_rate)
                self.px4_imu0_df[col] = (self.px4_imu0_df[col] + offset) * scale + noise
            
            if self.px4_imu1_df is not None:
                offset = self.imu_config[IMU_Type.IMU3.value]['accelerometer']['offset'][i]
                scale = self.imu_config[IMU_Type.IMU3.value]['accelerometer']['scale'][i]
                noise = get_acceleration_noise(self.imu_config[IMU_Type.IMU3.value]['accelerometer']['noise'], self.px4_imu_sampling_rate)
                self.px4_imu1_df[col] = (self.px4_imu1_df[col] + offset) * scale + noise
                
                
        for i, col in enumerate(self.columns.imu_gyr):
            if self.voxl_imu0_df is not None:
                offset = self.imu_config[IMU_Type.IMU0.value]['gyroscope']['offset'][i]
                noise = get_gyroscope_noise(self.imu_config[IMU_Type.IMU0.value]['gyroscope']['noise'], self.voxl_imu_sampling_rate)
                self.voxl_imu0_df[col] = self.voxl_imu0_df[col] + offset + noise
                
            if self.voxl_imu1_df is not None:
                offset = self.imu_config[IMU_Type.IMU1.value]['gyroscope']['offset'][i]
                noise = get_gyroscope_noise(self.imu_config[IMU_Type.IMU1.value]['gyroscope']['noise'], self.voxl_imu_sampling_rate)
                self.voxl_imu1_df[col] = self.voxl_imu1_df[col] + offset + noise
            
            if self.px4_imu0_df is not None:
                offset = self.imu_config[IMU_Type.IMU2.value]['gyroscope']['offset'][i]
                noise = get_gyroscope_noise(self.imu_config[IMU_Type.IMU2.value]['gyroscope']['noise'], self.px4_imu_sampling_rate)
                self.px4_imu0_df[col] = self.px4_imu0_df[col] + offset + noise
            
            if self.px4_imu1_df is not None:
                offset = self.imu_config[IMU_Type.IMU3.value]['gyroscope']['offset'][i]
                noise = get_gyroscope_noise(self.imu_config[IMU_Type.IMU3.value]['gyroscope']['noise'], self.px4_imu_sampling_rate)
                self.px4_imu1_df[col] = self.px4_imu1_df[col] + offset + noise
                
        if self.is_debugging:
            print("Finish calibrating IMUs")
    
    def _get_synced_data(self, aggregated_file_path) -> pd.DataFrame:
        """
            If not exist, generate UAV dataset synced at stereo camera sampling rate.
        """
        generated_data = self.flight_log["generated"]
        filename = os.path.join(aggregated_file_path, generated_data["synced"])
        if not self.regenerate_custom_data and generated_data["synced"] is not None and os.path.exists(filename):
            return pd.read_csv(filename)

        logger.warning('synced data not found.')
        
        def _get_index_by_timestamp(df_timestamp, timestamp, divider=1):
            return np.argmin(np.abs((df_timestamp/divider) - timestamp))
            
        stereo = self.voxl_stereo_df
        stereo_timestamp = stereo['timestamp(ns)'] / 1000
        # Sync drone data by 10Hz as KITTI dataset
        stereo_camera_sample_rate = len(stereo) / ((stereo['timestamp(ns)'].iloc[-1] - stereo['timestamp(ns)'].iloc[0]) / 1e9)
        
        logger.info(f'Creating synch data at the sampling rate of stereo camera, {round(stereo_camera_sample_rate, 2)}Hz.')
        
        # Get index based on the timestamp that is collected at 10Hz
        voxl_imu0_index = stereo_timestamp\
            .apply(lambda x: _get_index_by_timestamp(self.voxl_imu0_df['timestamp(ns)'], x, divider=1000))
        voxl_imu1_index = stereo_timestamp\
            .apply(lambda x: _get_index_by_timestamp(self.voxl_imu1_df['timestamp(ns)'], x, divider=1000))
        voxl_qvio_index = stereo_timestamp\
            .apply(lambda x: _get_index_by_timestamp(self.voxl_qvio_df['timestamp(ns)'], x, divider=1000))
            
        
        df_index = [voxl_imu0_index, voxl_imu1_index, voxl_qvio_index]
        sensor_types = [SensorType.VOXL_IMU0, SensorType.VOXL_IMU1, SensorType.VOXL_QVIO]

        if self.data_type == "outdoor":
            px4_imu0_index = stereo_timestamp\
                .apply(lambda x: _get_index_by_timestamp(self.px4_imu0_df['timestamp'], x))
            px4_imu1_index = stereo_timestamp\
                .apply(lambda x: _get_index_by_timestamp(self.px4_imu1_df['timestamp'], x))
            px4_gps_index = stereo_timestamp\
                .apply(lambda x: _get_index_by_timestamp(self.px4_gps_df['timestamp'], x))
            px4_vo_index = stereo_timestamp\
                .apply(lambda x: _get_index_by_timestamp(self.px4_vo_df['timestamp'], x))
            px4_vehicle_odom_index = stereo_timestamp\
                .apply(lambda x: _get_index_by_timestamp(self.px4_vehicle_odom_df['timestamp'], x))
            px4_imu0_bias_index = stereo_timestamp\
                .apply(lambda x: _get_index_by_timestamp(self.px4_imu0_bias_df['timestamp'], x))
            px4_imu1_bias_index = stereo_timestamp\
                .apply(lambda x: _get_index_by_timestamp(self.px4_imu1_bias_df['timestamp'], x))

            px4_actuator_motor_index = stereo_timestamp\
                .apply(lambda x: _get_index_by_timestamp(self.px4_actuator_motor_df['timestamp'], x))
            px4_actuator_output_index = stereo_timestamp\
                .apply(lambda x: _get_index_by_timestamp(self.px4_actuator_output_df['timestamp'], x))
                
            px4_magnetometer_index = stereo_timestamp\
                .apply(lambda x: _get_index_by_timestamp(self.px4_mag_df['timestamp'], x))

            px4_sensor_indices = [
                px4_imu0_index, px4_imu1_index, px4_gps_index, px4_vo_index, 
                px4_vehicle_odom_index, px4_imu0_bias_index, px4_imu1_bias_index, px4_actuator_motor_index, px4_actuator_output_index, px4_magnetometer_index]
            px4_sensor_names = [
                SensorType.PX4_IMU0, SensorType.PX4_IMU1, SensorType.PX4_GPS, 
                SensorType.PX4_VO, SensorType.PX4_VEHICLE_ODOM, SensorType.PX4_IMU0_BIAS, SensorType.PX4_IMU1_BIAS, 
                SensorType.PX4_ACTUATOR_MOTORS, SensorType.PX4_ACTUATOR_OUTPUTS, SensorType.PX4_MAG]
            
            # concatenate PX4 sensor data indices
            df_index.extend(px4_sensor_indices)
            
            # concatenate PX4 sensor names
            sensor_types.extend(px4_sensor_names)
            
        # Concatenate all sensors and make a single table 
        # Get available sensors
        sensors = self.get_sensors(timestamp=stereo_timestamp.iloc[0])

        # Concatenate all sensors and create a single table 
        result = stereo_timestamp.reset_index(drop=True)
        for index, sensor_type in zip(df_index, sensor_types):
            sensor = sensors[sensor_type]
            sensor_name = sensor_type.value
            result = pd.concat([
                result,
                sensor.df.iloc[index][sensor.column]\
                    .rename({col: sensor_name + '_' + col for col in sensor.column}, axis=1)\
                    .reset_index(drop=True)
            ], axis=1)

        # The timestamp is no longer nano second, so renmae it
        result.rename(columns={'timestamp(ns)': 'timestamp'}, inplace=True)
        
        export_filename = os.path.join(self.root_path, "UAV", self.version, sequence_nr + "_sync.csv")
        result.to_csv(export_filename, index=False)
        
        logger.info("Creating synced data completed.")
        return result
    
    def _get_reference_data(self, aggregated_file_path) -> pd.DataFrame:
        """
            Combine all data and sorted by timestamp if csv file does not exist.
            If exist, return DataFrame
        """
        generated_data = self.flight_log["generated"]
        filename = os.path.join(aggregated_file_path, generated_data["timestamp"])
        if not self.regenerate_custom_data and generated_data["timestamp"] is not None and os.path.exists(filename):
            return pd.read_csv(filename)

        def _get_formatted_df(df: pd.DataFrame, device: str, timestamp_label="timestamp", divider=1) -> pd.DataFrame:
            return pd.DataFrame({
                "timestamp": (df[timestamp_label].values / divider).astype(int),
                "device": device,
            })
            
        logger.warning('Timestamp combined data not found.')
        logger.info('Creating timestamp combined data.')
        
        df_list = []
        if self.data_type == "outdoor":
            px4_imu0 = _get_formatted_df(df=self.px4_imu0_df, device=SensorType.PX4_IMU0.value)
            px4_imu1 = _get_formatted_df(df=self.px4_imu1_df, device=SensorType.PX4_IMU1.value)
            px4_gps = _get_formatted_df(df=self.px4_gps_df, device=SensorType.PX4_GPS.value)
            px4_vo = _get_formatted_df(df=self.px4_vo_df, device=SensorType.PX4_VO.value)
            px4_vehicle_odom = _get_formatted_df(df=self.px4_vehicle_odom_df, device=SensorType.PX4_VEHICLE_ODOM.value)
            px4_imu0_bias = _get_formatted_df(df=self.px4_imu0_bias_df, device=SensorType.PX4_IMU0_BIAS.value)
            px4_imu1_bias = _get_formatted_df(df=self.px4_imu1_bias_df, device=SensorType.PX4_IMU1_BIAS.value)
            px4_mag = _get_formatted_df(df=self.px4_mag_df, device=SensorType.PX4_MAG.value)
            df_list = [
                px4_imu0, px4_imu1, px4_gps, px4_vo, px4_vehicle_odom, px4_imu0_bias, px4_imu1_bias, px4_mag
            ]
            
        voxl_imu0 = _get_formatted_df(df=self.voxl_imu0_df, device=SensorType.VOXL_IMU0.value, timestamp_label="timestamp(ns)", divider=1000)
        voxl_imu1 = _get_formatted_df(df=self.voxl_imu1_df, device=SensorType.VOXL_IMU1.value, timestamp_label="timestamp(ns)", divider=1000)
        voxl_qvio = _get_formatted_df(df=self.voxl_qvio_df, device=SensorType.VOXL_QVIO.value, timestamp_label="timestamp(ns)", divider=1000)
        
        df_list.append(voxl_imu0)
        df_list.append(voxl_imu1)
        df_list.append(voxl_qvio)

        all_data = pd.concat(df_list)

        all_data = all_data.sort_values(by="timestamp")
        all_data.to_csv(filename, index=False)
        
        logger.info('Creating timestamp combined data completed.')
        return all_data
            
    def _get_combined_imu_data(self, px4_root_path: str, combined_imu_path: str, imu_name: str) -> pd.DataFrame:
        """
            PX4 imu data is separated into acceleration and gyroscope data.
            Hence this method combine PX4 imu gyro and accel data into one single csv file.
        """
        if self.data_type == "indoor":
            return None
        
        if os.path.exists(combined_imu_path):
            return self.load_px4_data(combined_imu_path)
        
        logger.info("Creating combined IMU data.")
        px4 = self.flight_log["px4"]
        
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
        
        
        imu_gyro = pd.read_csv(os.path.join(px4_root_path, px4[f"{imu_name}_gyro"]))
        imu_acc = pd.read_csv(os.path.join(px4_root_path, px4[f"{imu_name}_acc"]))
        
        imu = pd.merge(imu_acc, imu_gyro, on="timestamp_sample")
        imu = imu[columns].rename(columns=rename_dict)
        
        export_dir = os.path.join(px4_root_path, "imu_combined")
        if not os.path.exists(export_dir):
            os.mkdir(export_dir)
            
        imu.to_csv(combined_imu_path)
        return imu

    @staticmethod
    def init_gps_data(px4_gps_path):
        df = pd.read_csv(px4_gps_path)
        df = UAV_DataLoader.int_lla_to_float_lla(df)
        return UAV_DataLoader.get_ned_coord(df)

    @staticmethod
    def int2float_lla(x):
        return float(Decimal(x / Decimal(10**(len(str(x)) - 2))))

    @staticmethod
    def int_lla_to_float_lla(df):
        getcontext().prec = 10
        df['lat'] = df['lat'].apply(lambda x: UAV_DataLoader.int2float_lla(x))
        df['lon'] = df['lon'].apply(lambda x: UAV_DataLoader.int2float_lla(x))
        df['alt'] = df['alt'].apply(lambda x: UAV_DataLoader.int2float_lla(x))
        return df
    
    @staticmethod
    def get_ned_coord(df):    
        origin = df[['lon', 'lat', 'alt']].iloc[0].values
        ned_pose = lla_to_ned(df[['lon', 'lat', 'alt']].values.T, origin).T
        df = pd.concat([
            df,
            pd.DataFrame(ned_pose, columns=['north', 'east', 'down'])
        ], axis=1)
        
        return df
    
    @staticmethod
    def visualize_result(gt, history):
        fig = plt.figure()
        
        ax1 = fig.add_subplot(111, projection='3d')
        ax1.set_title("EKF estimated trajectory vs GPS trajectory")
        
        xs, ys, zs = gt
        ax1.plot(xs, ys, zs, label='GPS trajectory', color='black')
        
        mu_x, mu_y, mu_z = history.reshape(history.shape[0], history.shape[1])[:, :3].T
        ax1.plot(mu_x, mu_y, mu_z, label='Estimated trajectory', color='red')
        
        ax1.set_xlabel('$X$', fontsize=14)
        ax1.set_ylabel('$Y$', fontsize=14)
        ax1.set_zlabel('$Z$', fontsize=14)
        
        # ax1.set_zlim((-1, 1))
        ax1.legend(loc='best', bbox_to_anchor=(1.1, 0., 0.2, 0.9))
        fig.tight_layout()
        plt.show()
        
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        xs, ys, zs = gt
        ax.plot(xs, ys, lw=2, label='QVIO estimated trajectory', color='black')
        
        xs, ys = mu_x, mu_y
        ax.plot(xs, ys, lw=2, label='Estimated trajectory', color="red")
        
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.legend()
        ax.grid()
        
    def visualize_imu_data(self, title="IMU data"):
        fig, (ax1, ax2) = plt.subplots(2, 3, figsize=(16, 8))
        acc_titles = ["AX(m/s2)", "AY(m/s2)", "AZ(m/s2)"]
        gyr_titles = ["GX(rad/s)", "GY(rad/s)", "GZ(rad/s)"]
        fig.suptitle(title, fontsize=24)
        
        for i in range(0, 3):
            ax1[i].title.set_text(acc_titles[i])
            ax2[i].title.set_text(gyr_titles[i])
            acc_columns = self.voxl_imu0_df.columns[1:4]
            gyr_columns = self.voxl_imu0_df.columns[4:7]
            
            voxl_imu0_val = (
                    self.voxl_imu0_df[acc_columns].values[:, i] - self.imu_config[IMU_Type.IMU0.value]['accelerometer']['offset'][i]
                ) * self.imu_config[IMU_Type.IMU0.value]['accelerometer']['scale'][i]
            ax1[i].plot(voxl_imu0_val, color="red", lw=0.5, label=f"VOXL IMU0 ({IMU_Type.IMU0.value})")
            
            voxl_imu1_val = (
                    self.voxl_imu1_df[acc_columns].values[:, i] - self.imu_config[IMU_Type.IMU1.value]['accelerometer']['offset'][i]
                ) * self.imu_config[IMU_Type.IMU1.value]['accelerometer']['scale'][i]
            ax1[i].plot(voxl_imu1_val, color="blue", lw=0.5, label=f"VOXL IMU1 ({IMU_Type.IMU1.value})")
            
            if self.px4_imu0_df is not None and self.px4_imu1_df is not None:
                px4_imu0_val = (
                        self.px4_imu0_df[acc_columns].values[:, i] - self.imu_config[IMU_Type.IMU2.value]['accelerometer']['offset'][i]
                    ) * self.imu_config[IMU_Type.IMU0.value]['accelerometer']['scale'][i]
                ax1[i].plot(px4_imu0_val, color="red", lw=0.5, label=f"PX4 IMU0 ({IMU_Type.IMU2.value})")
                
                px4_imu1_val = (
                        self.px4_imu1_df[acc_columns].values[:, i] - self.imu_config[IMU_Type.IMU3.value]['accelerometer']['offset'][i]
                    ) * self.imu_config[IMU_Type.IMU1.value]['accelerometer']['scale'][i]
                ax1[i].plot(px4_imu1_val, color="blue", lw=0.5, label=f"PX4 IMU2 ({IMU_Type.IMU3.value})")
        
            
            voxl_imu0_val = (
                    self.voxl_imu0_df[gyr_columns].values[:, i] - self.imu_config[IMU_Type.IMU0.value]['gyroscope']['offset'][i]
                )
            ax2[i].plot(voxl_imu0_val, color="red", lw=0.5, label=f"VOXL IMU0 ({IMU_Type.IMU0.value})")

            voxl_imu1_val = (
                    self.voxl_imu1_df[gyr_columns].values[:, i] - self.imu_config[IMU_Type.IMU1.value]['gyroscope']['offset'][i]
                )
            ax2[i].plot(voxl_imu1_val, color="blue", lw=0.5, label=f"VOXL IMU1 ({IMU_Type.IMU1.value})")
        
            if self.px4_imu0_df is not None and self.px4_imu1_df is not None:
                px4_imu0_val = (
                        self.px4_imu0_df[gyr_columns].values[:, i] - self.imu_config[IMU_Type.IMU2.value]['gyroscope']['offset'][i]
                    )
                ax2[i].plot(px4_imu0_val, color="red", lw=0.5, label=f"PX4 IMU0 ({IMU_Type.IMU2.value})")

                px4_imu1_val = (
                        self.px4_imu1_df[gyr_columns].values[:, i] - self.imu_config[IMU_Type.IMU3.value]['gyroscope']['offset'][i]
                    )
                ax2[i].plot(px4_imu1_val, color="blue", lw=0.5, label=f"PX4 IMU1 ({IMU_Type.IMU3.value})")
                
            ax1[i].grid()
            ax1[i].legend()
            ax2[i].grid()
            ax2[i].legend()

    def visualize_qvio_trajectory(
        self, 
        dimension=2, 
        detail=False, 
        x_lim=None, 
        y_lim=None, 
        z_lim=None
        ):
        try:
            if dimension == 2:
                _, ax1 = plt.subplots(1, 1, figsize=(8, 6))
                xs, ys = self.voxl_qvio_df[['T_imu_wrt_vio_x(m)', 'T_imu_wrt_vio_y(m)']].values.T
                ax1.plot(xs, ys, label='QVIO estimated trajectory')
                ax1.set_title("QVIO estimated trajectory in 2D space")
                ax1.set_xlabel('X [m]')
                ax1.set_ylabel('Y [m]')
                ax1.legend()
                ax1.grid()
            else:
                fig = plt.figure()
                ax1 = fig.add_subplot(111, projection='3d')
                ax1.set_title("QVIO estimated trajectory in 3D space")

                xs, ys, zs = self.voxl_qvio_df[['T_imu_wrt_vio_x(m)', 'T_imu_wrt_vio_y(m)', 'T_imu_wrt_vio_z(m)']].values.T
                ax1.plot(xs, ys, zs, label='QVIO estimated trajectory', color='black')

                ax1.set_xlabel('X', fontsize=14)
                ax1.set_ylabel('Y', fontsize=14)
                ax1.set_zlabel('Z', fontsize=14)
                
                fig.tight_layout()
                ax1.legend(loc='best', bbox_to_anchor=(1.1, 0., 0.2, 0.9))
                
            if x_lim:
                ax1.set_xlim(x_lim)
            
            if y_lim:
                ax1.set_ylim(y_lim)
            
            if z_lim:
                ax1.set_zlim(z_lim)
            
            if detail:
                fig, axs = plt.subplots(1, 3, figsize=(12, 4))

                xs, ys = self.voxl_qvio_df[['T_imu_wrt_vio_x(m)', 'T_imu_wrt_vio_y(m)']].values.T
                axs[0].plot(xs, -ys, label='QVIO estimated trajectory')
                axs[0].set_title("QVIO estimated trajectory X-Y")
                axs[0].set_xlabel('X [m]')
                axs[0].set_ylabel('Y [m]')
                axs[0].legend()
                axs[0].grid()

                xs, ys = self.voxl_qvio_df[['T_imu_wrt_vio_y(m)', 'T_imu_wrt_vio_z(m)']].values.T
                axs[1].plot(-xs, ys, label='QVIO estimated trajectory')
                axs[1].set_title("QVIO estimated trajectory Y-Z")
                axs[1].set_xlabel('Y [m]')
                axs[1].set_ylabel('Z [m]')
                axs[1].legend()
                axs[1].grid()

                xs, ys = self.voxl_qvio_df[['T_imu_wrt_vio_z(m)', 'T_imu_wrt_vio_x(m)']].values.T
                axs[2].plot(xs, ys, label='QVIO estimated trajectory')
                axs[2].set_title("QVIO estimated trajectory Z-X")

                axs[2].set_xlabel('Z [m]')
                axs[2].set_ylabel('X [m]')
                axs[2].legend()
                axs[2].grid()

                fig.tight_layout()
        except Exception as e:
            logger.info(f"An exception occurs: {e}")
    
    def get_sensors(self, timestamp):
        return {
            SensorType.VOXL_IMU0: Sensor(df=self.voxl_imu0_df, column=self.columns.imu, synced_column=self.columns.voxl_synced_imu0),
            SensorType.VOXL_IMU1: Sensor(df=self.voxl_imu1_df, column=self.columns.imu, synced_column=self.columns.voxl_synced_imu1),
            SensorType.VOXL_QVIO: Sensor(df=self.voxl_qvio_df, column=self.columns.voxl_vo_pose, synced_column=self.columns.voxl_synced_vo_pose),
            SensorType.VOXL_STEREO: Sensor(df=self.voxl_stereo_df, column=self.voxl_stereo_df.columns.to_list(), synced_column=['voxl_stereo_' + col for col in self.voxl_stereo_df.columns.tolist()]),
            SensorType.PX4_IMU0: Sensor(df=self.px4_imu0_df, column=self.columns.imu, synced_column=self.columns.px4_synced_imu0),
            SensorType.PX4_IMU1: Sensor(df=self.px4_imu1_df, column=self.columns.imu, synced_column=self.columns.px4_synced_imu1), 
            SensorType.PX4_GPS: Sensor(df=self.px4_gps_df, column=self.columns.gps_ned_pose, synced_column=self.columns.px4_synced_gps_ned_pose),
            SensorType.PX4_VO: Sensor(df=self.px4_vo_df, column=self.columns.vo_pose, synced_column=["px4_vo_"+col for col in self.columns.vo_pose]),
            SensorType.PX4_VEHICLE_ODOM: Sensor(df=self.px4_vehicle_odom_df, column=self.columns.vo_pose, synced_column=["px4_vehicle_odom_"+col for col in self.columns.vo_pose]), 
            SensorType.PX4_IMU0_BIAS: Sensor(df=self.px4_imu0_bias_df, column=self.columns.px4_imu_bias, synced_column=self.columns.px4_synced_imu0_bias),
            SensorType.PX4_IMU1_BIAS: Sensor(df=self.px4_imu1_bias_df, column=self.columns.px4_imu_bias, synced_column=self.columns.px4_synced_imu1_bias),
            SensorType.PX4_ACTUATOR_MOTORS: Sensor(df=self.px4_actuator_motor_df, column=self.columns.actuator_motors, synced_column=self.columns.actuator_motors),
            SensorType.PX4_ACTUATOR_OUTPUTS: Sensor(df=self.px4_actuator_output_df, column=self.columns.actuator_outputs, synced_column=self.columns.actuator_outputs),
            SensorType.PX4_MAG: Sensor(df=self.px4_mag_df, column=self.columns.mag, synced_column=self.columns.px4_synced_mag),
        }
        
    def set_dropout_rate(self, vo_dropout_ratio, gps_dropout_ratio):
        self.vo_dropout_ratio = vo_dropout_ratio
        self.gps_dropout_ratio = gps_dropout_ratio
        indices = [i for i in range(self.N)]
        
        self.vo_indices = np.sort(
            random.sample(indices, int(len(indices)*(1 - self.vo_dropout_ratio)))).tolist()
        self.gps_indices = np.sort(
            random.sample(indices, int(len(indices)*(1 - self.gps_dropout_ratio)))).tolist()

    def get_control_input_by_index(self, index, setup):
        ax, ay, az = self.synced_df[['voxl_imu0_AX(m/s2)', 'voxl_imu0_AY(m/s2)', 'voxl_imu0_AZ(m/s2)']].iloc[index].values
        wx, wy, wz = self.synced_df[['voxl_imu0_GX(rad/s)', 'voxl_imu0_GY(rad/s)', 'voxl_imu0_GZ(rad/s)']].iloc[index].values
        if setup is SetupEnum.SETUP_1 or setup is SetupEnum.SETUP_2:
            u = np.array([
                ax,
                ay,
                az,
                wx,
                wy,
                wz
            ])
        else: #SetupEnum.SETUP_3
            if self.dimension == 2:
                u = np.array([
                    0, # not provided
                    wz
                ])
            else:
                u = np.array([
                    0, # not provided
                    wx, #wx
                    wz #wz
                ])
        return u
    
    def get_gps_measurement_by_index(
        self, 
        index, 
        setup=SetupEnum.SETUP_1,
        measurement_type=MeasurementDataEnum.ALL_DATA):

        if self.data_type == "indoor":
            return (None, None)
            
        if setup is SetupEnum.SETUP_1:
            return (None, None)
        
        gps_data = self.synced_df[self.columns.px4_synced_gps_ned_pose].iloc[index].values.reshape(-1, 1)
                        
        if measurement_type is MeasurementDataEnum.ALL_DATA:
            return (gps_data, None)
        elif measurement_type is MeasurementDataEnum.DROPOUT:
            if index not in self.gps_indices:
                return (None, None)
            return (gps_data, None)
        elif measurement_type is MeasurementDataEnum.COVARIANCE:
            error = 10. if index in self.vo_indices else 100.
            q = np.repeat(error ** 2, self.dimension)
            return (gps_data, np.eye(self.dimension) * q)
        
        return (None, None)
        
    def get_vo_measurement_by_index(
        self, 
        index, 
        measurement_type=MeasurementDataEnum.ALL_DATA):
        
        vo_data = self.synced_df[self.columns.voxl_vo_pose].iloc[index].values.reshape(-1, 1)
        
        if measurement_type is MeasurementDataEnum.ALL_DATA:
            return (vo_data, None)
        elif measurement_type is MeasurementDataEnum.DROPOUT:
            if index not in self.vo_indices:
                return (None, None)
            return (vo_data, None)
        elif measurement_type is MeasurementDataEnum.COVARIANCE:
            error = 10.0 if index in self.vo_indices else 100.0
            q = np.repeat(error ** 2, self.dimension)
            return (vo_data, np.eye(self.dimension) * q)
        
        return (None, None)
    
    def get_trajectory_to_compare(self) -> np.ndarray:
        if self.data_type == "indoor":
            return self.synced_df[self.columns.voxl_vo_pose].values.T
        return self.synced_df[['px4_gps_north', 'px4_gps_east', 'px4_gps_down']].values.T
    
    def get_observation_matrix_and_noise(
        self,
        x: np.ndarray, 
        sensor: SensorType, 
        yaw_update=True
    ) -> (np.ndarray, np.ndarray):
        
        H_pos = np.eye(x.shape[0])[:3, :]
        R = np.eye(3)
        
        match (sensor):
            case SensorType.PX4_MAG:
                if yaw_update:
                    R = np.array([0.5]) ** 2
                    
                    q = x[6:10]
                    qw, qx, qy, qz = q.reshape(-1)
                    
                    # compute pitch from the quaternion to check Gimbal lock
                    pitch = np.arcsin( 2*(qw*qy - qx*qz) )

                    if pitch == np.pi / 2:
                        return np.array([
                            [0., 0., 0., 0., 0., 0., 2*qx / (qw**2 + qx**2), -2*qw / (qw**2 + qx**2), 0., 0.]
                        ]), R
                        
                    elif pitch == -np.pi / 2:
                        return np.array([
                            [0., 0., 0., 0., 0., 0., -2*qx / (qw**2 + qx**2), 2*qw / (qw**2 + qx**2), 0., 0.]
                        ]), R
                        
                    else:
                        dqw = 2*qz*(-2*qy**2 - 2*qz**2 + 1)/((2*qw*qz + 2*qx*qy)**2 + (-2*qy**2 - 2*qz**2 + 1)**2)
                        dqx = 2*qy*(-2*qy**2 - 2*qz**2 + 1)/((2*qw*qz + 2*qx*qy)**2 + (-2*qy**2 - 2*qz**2 + 1)**2)
                        dqy = (2*qx*(-2*qy**2 - 2*qz**2 + 1)/((2*qw*qz + 2*qx*qy)**2 + (-2*qy**2 - 2*qz**2 + 1)**2)) - (4*qy*(-2*qw*qz - 2*qx*qy) / ((2*qw*qz + 2*qx*qy)**2 + (-2*qy**2 - 2*qz**2 + 1)**2))
                        dqz = (2*qw*(-2*qy**2 - 2*qz**2 + 1)/((2*qw*qz + 2*qx*qy)**2 + (-2*qy**2 - 2*qy**2 + 1)**2)) - (4*qz*(-2*qw*qz - 2*qx*qy) / ((2*qw*qz + 2*qx*qy)**2 + (-2*qy**2 - 2*qz**2 + 1)**2))
                        return np.array([
                            [0., 0., 0., 0., 0., 0., dqw, dqx, dqy, dqz]
                        ]), R
                else:
                    R = np.eye(3) * np.array([2., 2., 2.]) ** 2
                    
                    q = x[6:10]
                    qw, qx, qy, qz = q.reshape(-1)
                
                    theta = Configs.declination_angle * np.pi / 180 # in radian
                    r_norm = np.sqrt(np.cos(theta)**2 + np.sin(theta)**2) ** -1
                    
                    # ENU
                    rx = 0.
                    ry = np.cos(theta) * r_norm
                    rz = -np.sin(theta) * r_norm
                    
                    # NED
                    # rx = np.cos(theta) * r_norm
                    # ry = 0.
                    # rz = np.sin(theta) * r_norm
                        
                    return np.array([
                        [0., 0., 0., 0., 0., 0., -2*qy*rz +2*qz*ry, 2*qy*ry +2*qz*rz, -2*qw*rz +2*qx*ry -4*qy*rx, 2*qw*ry +2*qx*rz -4*qz*rx],
                        [0., 0., 0., 0., 0., 0., 2*qx*rz -2*qz*rx, 2*qw*rz -4*qx*ry +2*qy*rx, 2*qx*rx +2*qz*rz, -2*qw*rx +2*qy*rz -4*qz*ry],
                        [0., 0., 0., 0., 0., 0., -2*qx*ry +2*qy*rx, -2*qw*ry -4*qx*rz +2*qz*rx, 2*qw*rx -4*qy*rz +2*qz*ry, 2*qx*rx +2*qy*ry],
                    ]), R
                    
            case SensorType.PX4_GPS:
                R *= np.array([1., 1., 2.0]) ** 2
                return H_pos, R
            
            case SensorType.VOXL_QVIO:
                R *= np.array([2., 2., 3.0]) ** 2
                return H_pos, R
            
            case SensorType.PX4_VEHICLE_ODOM:
                R *= np.array([2., 2., 3.]) ** 2
                return H_pos, R
            
            case SensorType.PX4_VO:
                R *= np.array([3., 3., 3.]) ** 2
                return H_pos, R
            
            case _:
                return H_pos, R * np.array([5., 5., 5.]) ** 2

        if sensor is SensorType.PX4_MAG:
            if yaw_update:
                R = np.array([0.1])
                
                q = x[6:10]
                qw, qx, qy, qz = q.reshape(-1)
                
                # compute pitch from the quaternion to check Gimbal lock
                pitch = np.arcsin( 2*(qw*qy - qx*qz) )

                if pitch == np.pi / 2:
                    return np.array([
                        [0., 0., 0., 0., 0., 0., 2*qx / (qw**2 + qx**2), -2*qw / (qw**2 + qx**2), 0., 0.]
                    ]), R
                    
                elif pitch == -np.pi / 2:
                    return np.array([
                        [0., 0., 0., 0., 0., 0., -2*qx / (qw**2 + qx**2), 2*qw / (qw**2 + qx**2), 0., 0.]
                    ]), R
                    
                else:
                    dqw = 2*qz*(-2*qy**2 - 2*qz**2 + 1)/((2*qw*qz + 2*qx*qy)**2 + (-2*qy**2 - 2*qz**2 + 1)**2)
                    dqx = 2*qy*(-2*qy**2 - 2*qz**2 + 1)/((2*qw*qz + 2*qx*qy)**2 + (-2*qy**2 - 2*qz**2 + 1)**2)
                    dqy = (2*qx*(-2*qy**2 - 2*qz**2 + 1)/((2*qw*qz + 2*qx*qy)**2 + (-2*qy**2 - 2*qz**2 + 1)**2)) - (4*qy*(-2*qw*qz - 2*qx*qy) / ((2*qw*qz + 2*qx*qy)**2 + (-2*qy**2 - 2*qz**2 + 1)**2))
                    dqz = (2*qw*(-2*qy**2 - 2*qz**2 + 1)/((2*qw*qz + 2*qx*qy)**2 + (-2*qy**2 - 2*qy**2 + 1)**2)) - (4*qz*(-2*qw*qz - 2*qx*qy) / ((2*qw*qz + 2*qx*qy)**2 + (-2*qy**2 - 2*qz**2 + 1)**2))
                    return np.array([
                        [0., 0., 0., 0., 0., 0., dqw, dqx, dqy, dqz]
                    ]), R
                    
            else:
                R = np.eye(3) * np.array([2., 2., 3.0]) ** 2
                
                q = x[6:10]
                qw, qx, qy, qz = q.reshape(-1)
            
                theta = self.declination_angle
                r_norm = np.sqrt(np.cos(theta)**2 + np.sin(theta)**2) ** -1
                rx = 0.
                ry = np.cos(theta) * r_norm
                rz = -np.sin(theta) * r_norm
                    
                return np.array([
                    [0., 0., 0., 0., 0., 0., -2*qy*rz +2*qz*ry, 2*qy*ry +2*qz*rz, -2*qw*rz +2*qx*ry -4*qy*rx, 2*qw*ry +2*qx*rz -4*qz*rx],
                    [0., 0., 0., 0., 0., 0., 2*qx*rz -2*qz*rx, 2*qw*rz -4*qx*ry +2*qy*rx, 2*qx*rx +2*qz*rz, -2*qw*rx +2*qy*rz -4*qz*ry],
                    [0., 0., 0., 0., 0., 0., -2*qx*ry +2*qy*rx, -2*qw*ry -4*qx*rz +2*qz*rx, 2*qw*rx -4*qy*rz +2*qz*ry, 2*qx*rx +2*qy*ry],
                ]), R
                
        
        if sensor is SensorType.VOXL_QVIO:
            R *= np.array([2., 2., 3.0]) ** 2
            return H_pos, R
            
        if sensor is SensorType.PX4_GPS:
            R *= np.array([1., 1., 2.0]) ** 2
            return H_pos, R

        if sensor is SensorType.PX4_VEHICLE_ODOM:
            R *= np.array([2., 2., 3.]) ** 2
            return H_pos, R
            
        if sensor is SensorType.PX4_VO:
            R *= np.array([3., 3., 3.]) ** 2
            return H_pos, R

        return H_pos, R * np.array([5., 5., 5.]) ** 2

if __name__ == "__main__":
    
    
    # root_data_path = "../../example_data"
    # sequence_nr="log0001"
    
    # uncomment this line to load full data
    root_data_path = "../../data"
    sequence_nr="log0001"
    
    loader = UAV_DataLoader(root_path=root_data_path, sequence_nr=sequence_nr, regenerate_custom_data=False)
    sensors = loader.get_sensors(timestamp=0)
    loader.visualize_imu_data(title="Test")