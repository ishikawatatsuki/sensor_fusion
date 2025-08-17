
import logging
import os
import time
import csv
import numpy as np
from typing import Union
from collections import namedtuple
from dataclasses import dataclass

from ..internal.extended_common import SensorType, ControlInput, MeasurementUpdateField, SensorData, VisualOdometryData
from ..internal.extended_common.extended_config import GeneralConfig, DatasetConfig


IMU_COLUMNS = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
GPS_COLUMNS = ['gps_x', 'gps_y', 'gps_z']
GPS_RAW_COLUMNS = ['gps_lon', 'gps_lat', 'gps_alt']
VO_COLUMNS = ['vo_dt', 'vo_0', 'vo_1', 'vo_2', 'vo_3', 'vo_4', 'vo_5', 'vo_6', 'vo_7', 'vo_8', 'vo_9', 'vo_10', 'vo_11'] # vo 12x1 relative pose estimates
LATERAL_UPWARD_VEL_COLUMNS = ['lat_vel', 'upward_vel']

@dataclass
class LoggingMessage:
    sensor_type: SensorType
    timestamp: int
    data: Union[
        ControlInput, 
        SensorData, 
        VisualOdometryData]

@dataclass
class LoggingData:
    timestamp: float
    type: SensorType

    acc_x = np.nan
    acc_y = np.nan
    acc_z = np.nan
    gyro_x = np.nan
    gyro_y = np.nan
    gyro_z = np.nan

    gps_x = np.nan
    gps_y = np.nan
    gps_z = np.nan

    vo_dt = np.nan
    vo_0 = np.nan
    vo_1 = np.nan
    vo_2 = np.nan
    vo_3 = np.nan
    vo_4 = np.nan
    vo_5 = np.nan
    vo_6 = np.nan
    vo_7 = np.nan
    vo_8 = np.nan
    vo_9 = np.nan
    vo_10 = np.nan
    vo_11 = np.nan

    lat_vel = np.nan
    upward_vel = np.nan


class DataLogger:
    def __init__(self, config: GeneralConfig, data_config: DatasetConfig):
        self.config = config
        self.base_log_dir = os.path.join(config.sensor_data_save_path, str(int(time.time())))

        self.log_filename = os.path.join(self.base_log_dir, "sensor_data.csv")
        self.raw_log_filename = os.path.join(self.base_log_dir, "raw_sensor_data.csv")

    def prepare(self):
        if not self.config.save_sensor_data:
            return
        
        os.makedirs(self.base_log_dir, exist_ok=True)

        with open(self.log_filename, "w") as f:
            f.write("timestamp,type,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z,gps_x,gps_y,gps_z,vo_dt,vo_0,vo_1,vo_2,vo_3,vo_4,vo_5,vo_6,vo_7,vo_8,vo_9,vo_10,vo_11,lat_vel,upward_vel\n")
        with open(self.raw_log_filename, "w") as f:
            f.write("timestamp,type,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z,gps_x,gps_y,gps_z,vo_dt,vo_0,vo_1,vo_2,vo_3,vo_4,vo_5,vo_6,vo_7,vo_8,vo_9,vo_10,vo_11,lat_vel,upward_vel\n")

        logging.info("Data logging initialized.")

    def _process(self, message: LoggingMessage) -> str:
        _data = LoggingData(
            timestamp=message.timestamp,
            type=message.sensor_type
        )
        if message.sensor_type == SensorType.OXTS_IMU:
            data: ControlInput = message.data
            _data.acc_x = data.u[0]
            _data.acc_y = data.u[1]
            _data.acc_z = data.u[2]
            _data.gyro_x = data.u[3]
            _data.gyro_y = data.u[4]
            _data.gyro_z = data.u[5]

        elif message.sensor_type == SensorType.OXTS_GPS:
            data: SensorData = message.data

            _data.gps_x = data.z[0]
            _data.gps_y = data.z[1]
            _data.gps_z = data.z[2]
        elif message.sensor_type == SensorType.KITTI_VO:
            data: VisualOdometryData = message.data
            _relative_pose = data.relative_pose.flatten()
            _data.vo_dt = data.dt
            _data.vo_0 = _relative_pose[0]
            _data.vo_1 = _relative_pose[1]
            _data.vo_2 = _relative_pose[2]
            _data.vo_3 = _relative_pose[3]
            _data.vo_4 = _relative_pose[4]
            _data.vo_5 = _relative_pose[5]
            _data.vo_6 = _relative_pose[6]
            _data.vo_7 = _relative_pose[7]
            _data.vo_8 = _relative_pose[8]
            _data.vo_9 = _relative_pose[9]
            _data.vo_10 = _relative_pose[10]
            _data.vo_11 = _relative_pose[11]
        elif message.sensor_type == SensorType.KITTI_UPWARD_LEFTWARD_VELOCITY:
            data: SensorData = message.data
            _data.lat_vel = data.z[0]
            _data.upward_vel = data.z[1]

        return _data

    def write(self, data: LoggingData, filename: str):
        data = [
                data.timestamp,
                data.type.name,
                data.acc_x, data.acc_y, data.acc_z,
                data.gyro_x, data.gyro_y, data.gyro_z,
                data.gps_x, data.gps_y, data.gps_z,
                data.vo_dt, data.vo_0, data.vo_1, data.vo_2,
                data.vo_3, data.vo_4, data.vo_5, data.vo_6,
                data.vo_7, data.vo_8, data.vo_9, data.vo_10,
                data.vo_11,
                data.lat_vel, data.upward_vel
            ]

        with open(filename, "a") as f:
            writer = csv.writer(f)
            writer.writerow(data)

    def log(self, message: LoggingMessage, is_raw: bool):
        if not self.config.save_sensor_data:
            return
        _data = self._process(message)

        if is_raw:
            self.write(_data, self.raw_log_filename)
        else:
            self.write(_data, self.log_filename)
        
if __name__ == "__main__":
    from ..internal.extended_common import SensorType


    rootpath = "/Volumes/Data_EXT/data/workspaces/sensor_fusion/data/KITTI"

    dataset_config = DatasetConfig(
        type='kitti',
        mode='stream',
        root_path=rootpath,
        variant='09',
        sensors=[SensorType.OXTS_IMU, SensorType.OXTS_GPS, SensorType.KITTI_VO],
    )
    config = GeneralConfig(
        log_level='debug',
        log_sensor_data=False,
        save_sensor_data=True,
        save_estimation=False,
        save_output_debug_frames=False,
        sensor_data_output_filepath='./.debugging/sensor_out.txt',
        sensor_data_save_path='./outputs/sensor_data'
    )

    data_logger = DataLogger(config=config, data_config=dataset_config)
    data_logger.prepare()

    for i in range(10):
        message = LoggingMessage(
            sensor_type=SensorType.OXTS_IMU,
            timestamp=i * 1000,
            data=ControlInput(
                u=np.random.rand(6),
                dt=0.1,
            )
        )
        raw_message = LoggingMessage(
            sensor_type=SensorType.OXTS_IMU,
            timestamp=i * 1000,
            data=ControlInput(
                u=np.random.rand(6),
                dt=0.1,
            )
        )

        gps_raw = LoggingMessage(
            sensor_type=SensorType.OXTS_GPS,
            timestamp=i * 1000,
            data=SensorData(
                z=np.random.rand(3),
            )
        )
        gps = LoggingMessage(
            sensor_type=SensorType.OXTS_GPS,
            timestamp=i * 1000,
            data=SensorData(
                z=np.random.rand(3),
            )
        )

        vo = LoggingMessage(
            sensor_type=SensorType.KITTI_VO,
            timestamp=i * 1000,
            data=VisualOdometryData(
                dt=0.1,
                relative_pose=np.random.rand(12),
                timestamp=i * 1000,
            )
        )
        vo_raw = LoggingMessage(
            sensor_type=SensorType.KITTI_VO,
            timestamp=i * 1000,
            data=VisualOdometryData(
                dt=0.1,
                relative_pose=np.random.rand(12),
                timestamp=i * 1000,
            )
        )

        data_logger.log(message=message, is_raw=False)
        data_logger.log(message=raw_message, is_raw=True)

        data_logger.log(message=gps, is_raw=False)
        data_logger.log(message=gps_raw, is_raw=True)
        data_logger.log(message=vo, is_raw=False)
        data_logger.log(message=vo_raw, is_raw=True)
        
