import os
import sys
import numpy as np
from collections import namedtuple

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from custom_types import UAV_SensorType
from config import (
    GyroSpecification,
    AccelSpecification
)
from utils import get_gyroscope_noise, get_acceleration_noise
from interfaces import State, Pose

class VOXL_IMUDataReader:
    """
        Read IMU data on VOXL side
    """
    def __init__(
            self, 
            path: str, 
            gyro_spec: GyroSpecification, 
            acc_spec: AccelSpecification, 
            starttime=-float('inf'),
            window_size=None
        ):
        self.divider = 1000
        self.path = path
        self.starttime = starttime
        self.field = namedtuple('data', 
            ['timestamp', 'a', 'w'])
        
        self.gyro_spec = gyro_spec
        self.acc_spec = acc_spec
        
        sampling_rate = 1000
        self.gyro_noise = get_gyroscope_noise(
                self.gyro_spec.noise, 
                sampling_rate, 
            )
        self.acc_noise = get_acceleration_noise(
                self.acc_spec.noise,
                sampling_rate
            )
        self.window_size = window_size
        self.buffer = []

    def parse(self, line):
        """
        line: 
            i,timestamp(ns),AX(m/s2),AY(m/s2),AZ(m/s2),GX(rad/s),GY(rad/s),GZ(rad/s),T(C)
            
            Apply calibration to IMU data:
                corrected_acc = (a_measured - offset) * scale
                corrected_gyro = w_measured - offset
        """
        line = [float(_) for _ in line.strip().split(',')]

        timestamp = line[1] / self.divider
        a = np.array(line[2:5])
        w = np.array(line[5:8])
        
        a = (a - np.array(self.acc_spec.offset)) * np.array(self.acc_spec.scale) - np.random.normal(0, self.acc_noise, 3)
        w = w - np.array(self.gyro_spec.offset) - np.random.normal(0, self.gyro_noise, 3)
        return self.field(timestamp, a, w)
    
    def rolling_average(self, data):
        
        d = np.hstack([data.w, data.a])
        self.buffer.append(d)
        if len(self.buffer) > self.window_size:
            mean = np.mean(self.buffer, axis=0)
            self.buffer = self.buffer[-self.window_size:]
            return self.field(timestamp=data.timestamp, w=mean[:3], a=mean[3:])
            
        return self.field(timestamp=data.timestamp, w=data.w, a=data.a)
    
    def __iter__(self):
        with open(self.path, 'r') as f:
            next(f)
            for line in f:
                data = self.parse(line)
                if data.timestamp < self.starttime:
                    continue
                if self.window_size is not None:
                    data = self.rolling_average(data)
                yield data

    def start_time(self):
        with open(self.path, 'r') as f:
            next(f)
            for line in f:
                return self.parse(line).timestamp

    def set_starttime(self, starttime):
        self.starttime = starttime
    
class VOXL_StereoFrameReader:
    def __init__(self, path, image_root_path, starttime=-float('inf')):
        self.divider = 1000
        self.starttime = starttime
        self.path = path
        self.image_root_path = image_root_path
        self.field = namedtuple('data', ['timestamp', 'left_frame_id', 'right_frame_id'])
        
    def parse(self, line):
        """
        line: 
            i,timestamp(ns),gain,exposure(ns),format,height,width,frame_id,reserved
        """
        line = [float(_) for _ in line.strip().split(',')]

        index = int(line[0])
        timestamp = line[1] / self.divider
        left_frame_id = f"{index:05}l.png"
        right_frame_id = f"{index:05}r.png"
        if not os.path.exists(os.path.join(self.image_root_path, left_frame_id)) or\
            not os.path.exists(os.path.join(self.image_root_path, right_frame_id)):
            return None

        return self.field(timestamp, left_frame_id, right_frame_id)

    def __iter__(self):
        with open(self.path, 'r') as f:
            next(f)
            for line in f:
                data = self.parse(line)
                if data is None or data.timestamp < self.starttime:
                    continue
                yield data

    def start_time(self):
        with open(self.gyro_path, 'r') as f:
            next(f)
            for line in f:
                return self.parse(line).timestamp

    def set_starttime(self, starttime):
        self.starttime = starttime
    

class VOXL_QVIOOverlayDataReader:
    
    def __init__(self, path: str, starttime=-float('inf')):
        self.path = path
        self.root_path = "/".join(path.split("/")[:-1])
        self.divider = 1000
        self.starttime = starttime
        self.field = namedtuple('data', 
            ['timestamp', 'image_path'])

    def parse(self, line):
        """
        line: 
            i,timestamp(ns),gain,exposure(ns),format,height,width,frame_id,reserved
        """
        line = [float(_) for _ in line.strip().split(',')]
        
        index = int(line[0])
        frame_id = f"{index:05}.png"
        image_path = os.path.join(self.root_path, frame_id)
        timestamp = line[1] / self.divider
        
        return self.field(timestamp, image_path)

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


class PX4_IMUDataReader:
    def __init__(
            self, 
            gyro_path: str, 
            acc_path: str,
            gyro_spec: GyroSpecification,
            acc_spec: AccelSpecification,
            starttime=-float('inf'),
            window_size=None,
        ):
        self.divider = 1
        self.gyro_path = gyro_path
        self.acc_path = acc_path
        self.starttime = starttime
        self.field = namedtuple('data', 
            ['timestamp', 'a', 'w'])
        
        self.gyro_spec = gyro_spec
        self.acc_spec = acc_spec
        sampling_rate = 100
        self.gyro_noise = get_gyroscope_noise(
                self.gyro_spec.noise, 
                sampling_rate, 
            )
        self.acc_noise = get_acceleration_noise(
                self.acc_spec.noise,
                sampling_rate
            )
        
        self.window_size = window_size
        self.buffer = []
    
    def parse(self, gyro_line, acc_line):
        """
        gyro_line: 
            timestamp,timestamp_sample,device_id,x,y,z,temperature,error_count,clip_counter[0],clip_counter[1],clip_counter[2],samples
        acc_line:
            timestamp,timestamp_sample,device_id,x,y,z,temperature,error_count,clip_counter[0],clip_counter[1],clip_counter[2],samples
            
        Apply calibration to IMU data:
            corrected_acc = (a_measured - offset) * scale
            corrected_gyro = w_measured - offset
        """
        gyro_line = [float(_) for _ in gyro_line.strip().split(',')]
        acc_line = [float(_) for _ in acc_line.strip().split(',')]

        timestamp = gyro_line[0] / self.divider
        w = np.array(gyro_line[3:6])
        a = np.array(acc_line[3:6])
        
        a = (a - np.array(self.acc_spec.offset)) * np.array(self.acc_spec.scale) - np.random.normal(0, self.acc_noise, 3)
        w = w - np.array(self.gyro_spec.offset) - np.random.normal(0, self.gyro_noise, 3)
        return self.field(timestamp, w, a)
    
    def rolling_average(self, data):
        
        d = np.hstack([data.w, data.a])
        self.buffer.append(d)
        if len(self.buffer) > self.window_size:
            mean = np.mean(self.buffer, axis=0)
            self.buffer = self.buffer[-self.window_size:]
            return self.field(timestamp=data.timestamp, w=mean[:3], a=mean[3:])
            
        return self.field(timestamp=data.timestamp, w=data.w, a=data.a)
    
    def __iter__(self):
        with open(self.gyro_path, 'r') as gyro_f, open(self.acc_path, 'r') as acc_f:
            next(gyro_f)
            next(acc_f)
            for gyro_line, acc_line in zip(gyro_f, acc_f):
                data = self.parse(gyro_line, acc_line)
                if data.timestamp < self.starttime:
                    continue
                if self.window_size is not None:
                    data = self.rolling_average(data)
                yield data

    def start_time(self):
        with open(self.gyro_path, 'r') as f:
            next(f)
            for line in f:
                return self.parse(line).timestamp

    def set_starttime(self, starttime):
        self.starttime = starttime
    
class PX4_MagnetometerDataReader:
    def __init__(self, path, divider, starttime=-float('inf')):
        self.divider = divider
        self.path = path
        self.starttime = starttime
        self.field = namedtuple('data', 
            ['timestamp', 'x', 'y', 'z'])

    def parse(self, line):
        """
        line: 
            timestamp,timestamp_sample,device_id,x,y,z,temperature,error_count
        """
        line = [float(_) for _ in line.strip().split(',')]

        timestamp = line[0] / self.divider
        x = np.array(line[3])
        y = np.array(line[4])
        z = np.array(line[5])
        return self.field(timestamp, x, y, z)

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

class PX4_GPSDataReader:
    def __init__(self, path, divider, starttime=-float('inf')):
        self.divider = divider
        self.path = path
        self.starttime = starttime
        self.initial_pose = None
        self.field = namedtuple('data', 
            ['timestamp', 'lon', 'lat', 'alt'])

    def parse(self, line):
        """
        line: 
            timestamp,timestamp_sample,time_utc_usec,device_id,lat,lon,alt,alt_ellipsoid,s_variance_m_s,c_variance_rad,eph,epv,hdop,vdop,noise_per_ms,jamming_indicator,vel_m_s,vel_n_m_s,vel_e_m_s,vel_d_m_s,cog_rad,timestamp_time_relative,heading,heading_offset,heading_accuracy,rtcm_injection_rate,automatic_gain_control,fix_type,jamming_state,spoofing_state,vel_ned_valid,satellites_used,selected_rtcm_instance
        """
        line = [float(_) for _ in line.strip().split(',')]
        
        timestamp = line[0] / self.divider
        return self.field(timestamp, lon=int(line[5]), lat=int(line[4]), alt=int(line[6]))

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
        
    def get_initial_state(self) -> State:
        with open(self.path, 'r') as f:
            next(f)
            line = f.readline()
            f.close()
            line = [float(_) for _ in line.strip().split(',')]
            lat=int(line[4])
            lon = int(line[5])
            alt=int(line[6])
            lla = np.array([lon, lat, alt]).reshape(-1, 1)
            heading = float(line[22] if not np.isnan(line[22]) else 0.)
            heading_offset = float(line[23] if not np.isnan(line[22]) else 0.)
            yaw = heading + heading_offset
            yaw = (yaw + np.pi) % (2 * np.pi) - np.pi
            yaw = np.pi / 2 - yaw
            euler = np.array([0., 0., yaw])
            quat = State.get_quaternion_from_euler_angle(euler)
            vel_n_m_s = float(line[17])
            vel_e_m_s = float(line[18])
            vel_d_m_s = float(line[19])
            vel = np.array([vel_n_m_s, vel_e_m_s, vel_d_m_s])
        p = np.zeros((3, 1))
        v = vel.reshape(-1, 1)
        q = quat.reshape(-1, 1)
        R = State.get_rotation_matrix_from_quaternion_vector(q.flatten())
        self.initial_pose = Pose(
            R=R,
            t=p
        )

        return State(p=p, v=v, q=q)

class PX4_VisualOdometryDataReader:
    
    def __init__(
            self, 
            path, 
            divider,
            window_size=None, 
            starttime=-float('inf')
        ):
        self.divider = divider
        self.path = path
        self.starttime = starttime
        self.field = namedtuple('data', 
            ['timestamp', 'x', 'y', 'z', 'vx', 'vy', 'vz'])

        self.last_position = None
        self.last_timestamp = None
        self.window_size = window_size
        self.buffer = []
        
    def parse(self, line):
        """
        line: 
            timestamp,timestamp_sample,position[0],position[1],position[2],q[0],q[1],q[2],q[3],velocity[0],velocity[1],velocity[2],angular_velocity[0],angular_velocity[1],angular_velocity[2],position_variance[0],position_variance[1],position_variance[2],orientation_variance[0],orientation_variance[1],orientation_variance[2],velocity_variance[0],velocity_variance[1],velocity_variance[2],pose_frame,velocity_frame,reset_counter,quality
        """
        line = [float(_) for _ in line.strip().split(',')]

        timestamp = line[0] / self.divider
        p = np.array(line[2:5])
        vel = np.array(line[9:12])
        x, y, z = p
        vx, vy, vz = vel
        
        return self.field(timestamp, x, y, z, vx, vy, vz)
    
    def rolling_average(self, velocity):
        self.buffer.append(velocity)
        if len(self.buffer) > self.window_size:
            mean = np.mean(self.buffer, axis=0)
            self.buffer = self.buffer[-self.window_size:]
            return mean
            
        return velocity

    def __iter__(self):
        with open(self.path, 'r') as f:
            next(f)
            for line in f:
                data = self.parse(line)
                position = np.array([data.x, data.y, data.z])
                if data.timestamp < self.starttime or self.last_timestamp is None:
                    self.last_timestamp = data.timestamp
                    self.last_position = position
                    continue
                
                dt = (data.timestamp - self.last_timestamp) / 1e6
                delta_p = position - self.last_position
                # velocity = delta_p / dt
                velocity = np.array([data.vx, data.vy, data.vz])
                
                self.last_timestamp = data.timestamp
                self.last_position = position
                
                if self.window_size is not None:
                    velocity = self.rolling_average(velocity=velocity)
                    data = self.field(data.timestamp, data.x, data.y, data.z, velocity[0], velocity[1], velocity[2])
                
                yield data

    def start_time(self):
        with open(self.path, 'r') as f:
            next(f)
            for line in f:
                return self.parse(line).timestamp

    def set_starttime(self, starttime):
        self.starttime = starttime


class UAVCustomVisualOdometryDataReader:
    # NOTE: This is experimental vo data
    
    def __init__(
            self, 
            path, 
            divider,
            window_size=None, 
            starttime=-float('inf')
        ):
        self.divider = divider
        self.path = path
        self.starttime = starttime
        self.data_field = namedtuple('data_field', ['timestamp', 'pose'])
        self.field = namedtuple('data', 
            ['timestamp', 'dt', 'delta_pose'])

        self.prev_pose = None
        self.last_timestamp = None
        self.window_size = window_size
        self.buffer = []
        
    def parse(self, line):
        """
        line: 
            timestamp,timestamp_sample,position[0],position[1],position[2],q[0],q[1],q[2],q[3],velocity[0],velocity[1],velocity[2],angular_velocity[0],angular_velocity[1],angular_velocity[2],position_variance[0],position_variance[1],position_variance[2],orientation_variance[0],orientation_variance[1],orientation_variance[2],velocity_variance[0],velocity_variance[1],velocity_variance[2],pose_frame,velocity_frame,reset_counter,quality
        """
        line = [float(_) for _ in line.strip().split(',')]

        timestamp = line[0] / self.divider
        t = np.array(line[2:5])
        q = np.array(line[5:9])
        
        return self.data_field(timestamp, pose=Pose(R=State.get_rotation_matrix_from_quaternion_vector(q), t=t))
    
    def rolling_average(self, velocity):
        self.buffer.append(velocity)
        if len(self.buffer) > self.window_size:
            mean = np.mean(self.buffer, axis=0)
            self.buffer = self.buffer[-self.window_size:]
            return mean
            
        return velocity

    def __iter__(self):
        with open(self.path, 'r') as f:
            next(f)
            for line in f:
                data = self.parse(line)
                if data.timestamp < self.starttime or self.last_timestamp is None:
                    self.last_timestamp = data.timestamp
                    self.prev_pose = data.pose
                    continue
                
                dt = (data.timestamp - self.last_timestamp) / 1e6
                delta_pose = self.prev_pose.inverse() * data.pose

                self.last_timestamp = data.timestamp
                self.prev_pose = data.pose
                
                yield self.field(timestamp=data.timestamp, dt=dt, delta_pose=delta_pose)

    def start_time(self):
        with open(self.path, 'r') as f:
            next(f)
            for line in f:
                return self.parse(line).timestamp

    def set_starttime(self, starttime):
        self.starttime = starttime
if __name__ == "__main__":
    import yaml
    
    imu_configs = None
    with open("./uav_imu_config.yaml", "r") as f:
        imu_configs = yaml.safe_load(f)
        f.close()
        
    imu_config = namedtuple('IMU_Configs', ["icm_42688_p", "icm_20948", "icm_20602", "icm_42688"])(**imu_configs)
    
    imu = PX4_IMUDataReader(
        gyro_path="../../data/UAV/log0001/px4/09_00_22_sensor_gps_0.csv",
        acc_path="../../data/UAV/log0001/px4/09_00_22_sensor_accel_0.csv",
        gyro_spec=GyroSpecification(**imu_config.icm_42688["gyroscope"]),
        acc_spec=AccelSpecification(**imu_config.icm_42688["accelerometer"])
        
    )
    
    overlay = VOXL_QVIOOverlayDataReader(
        path="../../data/UAV/log0001/run/mpa/qvio_overlay/data.csv"
    )
    
    # imu = VOXL_IMUDataReader(
    #     path="../../data/UAV/log0003/run/mpa/imu0/data.csv",
    #     gyro_spec=GyroSpecification(**imu_config.icm_42688_p["gyroscope"]),
    #     acc_spec=AccelSpecification(**imu_config.icm_42688_p["accelerometer"])
    # )
    
    # stereo = VOXL_StereoFrameReader(
    #     path="../../data/UAV/log0001/run/mpa/stereo/data.csv",
    #     image_root_path="../../data/UAV/log0001/run/mpa/stereo"
    # )
    # imu = OXTS_IMUDataReader(
    #   root_path="../../../data",
    #   date="2011_09_30",
    #   sequence_nr="0033"
    # )
    
    i = 0
    dataset = iter(overlay)
    while True:
        try:
            data = next(dataset)
            print(data.timestamp)
            print(data.image_path)
            # print(data.left_frame_id, data.right_frame_id)
            # print(data.a)
            # print(data.w)
        except StopIteration:
            break
        i += 1
