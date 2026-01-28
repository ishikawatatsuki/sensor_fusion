import os
import numpy as np
import pandas as pd
from datetime import datetime
from collections import namedtuple

if __name__ == "__main__":
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
else:
    pass

class UrbanNav_IMUDataReader:
    def __init__(
            self, 
            imu_path: str, 
            starttime=-float('inf'),
            window_size=None,
        ):
        self.multiplier = 1000
        self.imu_path = imu_path
        self.starttime = starttime
        self.field = namedtuple('data', 
            ['timestamp', 'a', 'w'])
        
        self.window_size = window_size
        self.buffer = []
    
    def parse(self, imu_line):
        """
        IMU data csv format:
        GPS TOW (s), GPS Week, Acceleration X (m/s^2), Acceleration Y (m/s^2), Acceleration Z (m/s^2), Angular rate X (rad/s), Angular rate Y (rad/s), Angular rate Z (rad/s), Wheel velocity (m/s)
            
        
        """
        lines = [float(_) for _ in imu_line.strip().split(',')]

        timestamp = lines[0] * self.multiplier
        a = np.array(lines[2:5])
        w = np.array(lines[5:8])
        return self.field(timestamp, a, w)
    
    def rolling_average(self, data):
        
        d = np.hstack([data.a, data.w])
        self.buffer.append(d)
        if len(self.buffer) > self.window_size:
            mean = np.mean(self.buffer, axis=0)
            self.buffer = self.buffer[-self.window_size:]
            return self.field(timestamp=data.timestamp, a=mean[:3], w=mean[3:6])
            
        return self.field(timestamp=data.timestamp, a=data.a, w=data.w)
    
    def __iter__(self):
        with open(self.imu_path, 'r') as imu_f:
            next(imu_f)
            for imu_line in imu_f:
                data = self.parse(imu_line)
                if data.timestamp < self.starttime:
                    continue
                if self.window_size is not None:
                    data = self.rolling_average(data)
                yield data

    def start_time(self):
        with open(self.imu_path, 'r') as f:
            next(f)
            for line in f:
                return self.parse(line).timestamp

    def set_starttime(self, starttime):
        self.starttime = starttime
    

class UrbanNav_WheelOdometryDataReader:
    def __init__(self, path, starttime=-float('inf')):
        self.multiplier = 1000
        self.path = path
        self.starttime = starttime
        self.field = namedtuple('data', 
            ['timestamp', 'wheel_speed'])

    def parse(self, line):
        """
        line: 
            GPS TOW (s), GPS Week, Acceleration X (m/s^2), Acceleration Y (m/s^2), Acceleration Z (m/s^2), Angular rate X (rad/s), Angular rate Y (rad/s), Angular rate Z (rad/s), Wheel velocity (m/s)
        """
        line = [float(_) for _ in line.strip().split(',')]

        timestamp = line[0] * self.multiplier
        wheel_speed = np.array(line[8])
        return self.field(timestamp, wheel_speed)

    def __iter__(self):
        with open(self.path, 'r') as f:
            next(f)
            for line in f:
                data = self.parse(line)
                if data.timestamp < self.starttime:
                    continue
                yield data

    def start_time(self):
        with open(self.path, 'r') as f:
            next(f)
            for line in f:
                return self.parse(line).timestamp

    def set_starttime(self, starttime):
        self.starttime = starttime


class UrbanNav_ReferenceDataReader:
    def __init__(self, reference_path: str, starttime=-float('inf')):
        """
        Ground truth data reader from Applanix POS LV620 (10 Hz)
        
        Args:
            reference_path: Path to reference.csv file
            starttime: Starting timestamp for filtering data
        """
        self.multiplier = 1000
        self.reference_path = reference_path
        self.starttime = starttime
        self.field = namedtuple('data', 
            ['timestamp', 'latitude', 'longitude', 'ellipsoid_height',
             'ecef_x', 'ecef_y', 'ecef_z',
             'roll', 'pitch', 'heading', 
             'vel_x', 'vel_y', 'vel_z',
             'acc_x', 'acc_y', 'acc_z',
             'ang_rate_x', 'ang_rate_y', 'ang_rate_z'])

    def parse(self, line):
        """
        Reference CSV format (Tokyo UrbanNav dataset):
        GPS TOW (s), GPS Week, Latitude (deg), Longitude (deg), Ellipsoid Height (m), 
        ECEF X (m), ECEF Y (m), ECEF Z (m), 
        Roll (deg), Pitch (deg), Heading (deg), 
        Velocity X (m/s), Velocity Y (m/s), Velocity Z (m/s), 
        Acceleration X (m/s^2), Acceleration Y (m/s^2), Acceleration Z (m/s^2), 
        Angular rate X (rad/s), Angular rate Y (rad/s), Angular rate Z (rad/s)
        """
        line = [float(_) for _ in line.strip().split(',')]
        
        timestamp = line[0] * self.multiplier
        latitude = line[2]
        longitude = line[3]
        ellipsoid_height = line[4]
        ecef_x = line[5]
        ecef_y = line[6]
        ecef_z = line[7]
        roll = np.deg2rad(line[8])
        pitch = np.deg2rad(line[9])
        heading = np.deg2rad(line[10])
        vel_x = line[11]
        vel_y = line[12]
        vel_z = line[13]
        acc_x = line[14]
        acc_y = line[15]
        acc_z = line[16]
        ang_rate_x = line[17]
        ang_rate_y = line[18]
        ang_rate_z = line[19]
        
        return self.field(timestamp, latitude, longitude, ellipsoid_height,
                         ecef_x, ecef_y, ecef_z,
                         roll, pitch, heading, 
                         vel_x, vel_y, vel_z,
                         acc_x, acc_y, acc_z,
                         ang_rate_x, ang_rate_y, ang_rate_z)

    def __iter__(self):
        with open(self.reference_path, 'r') as f:
            next(f)  # Skip header
            for line in f:
                data = self.parse(line)
                if data.timestamp < self.starttime:
                    continue
                yield data

    def start_time(self):
        with open(self.reference_path, 'r') as f:
            next(f)
            for line in f:
                return self.parse(line).timestamp

    def set_starttime(self, starttime):
        self.starttime = starttime


class UrbanNav_GPSDataReader:
    def __init__(self, gps_path: str, starttime=-float('inf')):
        """
        GPS measurement data reader
        
        Args:
            gps_path: Path to GPS measurement CSV file (rover data)
            starttime: Starting timestamp for filtering data
            
        Note: For RINEX files (.obs, .nav), consider using specialized RINEX parsers
        like georinex or similar libraries. This reader assumes processed CSV format.
        """
        self.multiplier = 1000
        self.gps_path = gps_path
        self.starttime = starttime
        self.field = namedtuple('data', 
            ['timestamp', 'latitude', 'longitude', 'altitude', 
             'num_satellites', 'hdop'])

    def parse(self, line):
        """
        GPS CSV format (assuming processed format):
        GPS TOW (s), GPS Week, Latitude (deg), Longitude (deg), Altitude (m),
        Number of Satellites, HDOP
        
        Note: Adjust based on actual GPS data format
        For RINEX files, use specialized parsers like georinex
        """
        line = [float(_) if _ else 0.0 for _ in line.strip().split(',')]
        
        timestamp = line[0] * self.multiplier
        latitude = line[2]
        longitude = line[3]
        altitude = line[4]
        num_satellites = int(line[5]) if len(line) > 5 else 0
        hdop = line[6] if len(line) > 6 else 1.0
        
        return self.field(timestamp, latitude, longitude, altitude,
                         num_satellites, hdop)

    def __iter__(self):
        with open(self.gps_path, 'r') as f:
            next(f)  # Skip header
            for line in f:
                data = self.parse(line)
                if data.timestamp < self.starttime:
                    continue
                yield data

    def start_time(self):
        with open(self.gps_path, 'r') as f:
            next(f)
            for line in f:
                return self.parse(line).timestamp

    def set_starttime(self, starttime):
        self.starttime = starttime

