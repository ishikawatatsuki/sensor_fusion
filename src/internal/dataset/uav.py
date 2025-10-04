import os
import sys
import logging
import numpy as np
from collections import namedtuple

if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ...utils import get_gyroscope_noise, get_acceleration_noise
from ..extended_common import (
    State, Pose,
    FilterConfig,
    MotionModel,
    GyroSpecification,
    AccelSpecification
)

class VOXL_IMUDataReader:
    """
        Read IMU data on VOXL side
    """
    def __init__(
            self, 
            path: str, 
            starttime=-float('inf'),
            window_size=None
        ):
        self.path = path
        self.starttime = starttime
        self.field = namedtuple('data', 
            ['timestamp', 'a', 'w'])
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

        timestamp = line[1]
        a = np.array(line[2:5])
        w = np.array(line[5:8])
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
        timestamp = line[1]
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
        with open(self.path, 'r') as f:
            next(f)
            for line in f:
                return self.parse(line).timestamp

    def set_starttime(self, starttime):
        self.starttime = starttime
    

class VOXL_QVIOOverlayDataReader:
    
    def __init__(self, path: str, starttime=-float('inf')):
        self.path = path
        self.root_path = "/".join(path.split("/")[:-1])
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
        timestamp = line[1]
        
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


class VOXL_TrackingCameraDataReader:
    
    def __init__(self, path: str, starttime=-float('inf')):
        self.path = path
        self.root_path = "/".join(path.split("/")[:-1])
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
        timestamp = line[1]
        
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
            starttime=-float('inf'),
            window_size=None,
        ):
        self.multiplier = 1000
        self.gyro_path = gyro_path
        self.acc_path = acc_path
        self.starttime = starttime
        self.field = namedtuple('data', 
            ['timestamp', 'a', 'w'])
        
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

        timestamp = gyro_line[0] * self.multiplier
        w = np.array(gyro_line[3:6])
        a = np.array(acc_line[3:6])
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
    def __init__(self, path, starttime=-float('inf')):
        self.multiplier = 1000
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

        timestamp = line[0] * self.multiplier
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
    def __init__(self, path, starttime=-float('inf')):
        self.multiplier = 1000
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
        timestamp = int(line[0]) * self.multiplier
        # timestamp = line[1] 

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
        
    def get_initial_state(self, filter_config: FilterConfig) -> State:
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
        b_w = np.zeros((3, 1))
        b_a = np.zeros((3, 1))
        m = MotionModel.get_motion_model(filter_config.motion_model)
        if m is MotionModel.DRONE_KINEMATICS:
            return State(p=p, v=v, q=q, b_w=b_w, b_a=b_a, w=np.zeros((3, 1)))

        return State(p=p, v=v, q=q, b_w=b_w, b_a=b_a)

class PX4_VisualOdometryDataReader:
    
    def __init__(
            self, 
            path, 
            simulate_delay=False,
            min_confidence=0.8,
            max_confidence=1.0,
            avg_delay=0.5, # in seconds
            window_size=None, 
            starttime=-float('inf')
        ):
        self.multiplier = 1000
        self.path = path
        self.starttime = starttime
        self.parsed_field = namedtuple('data', ['timestamp', 'pose', 'quality'])
        self.field = namedtuple('data', ['timestamp', 'relative_pose', 'dt'])

        self.simulate_delay = simulate_delay
        self.min_confidence = min(1.0, max(0, min_confidence))
        self.max_confidence = max(0, min(1.0, max_confidence))
        self.avg_delay = avg_delay

        self.last_pose = None
        self.last_timestamp = None
        self.window_size = window_size
        self.buffer = []
        
    def parse(self, line):
        """
        line: 
            timestamp,timestamp_sample,position[0],position[1],position[2],q[0],q[1],q[2],q[3],velocity[0],velocity[1],velocity[2],angular_velocity[0],angular_velocity[1],angular_velocity[2],position_variance[0],position_variance[1],position_variance[2],orientation_variance[0],orientation_variance[1],orientation_variance[2],velocity_variance[0],velocity_variance[1],velocity_variance[2],pose_frame,velocity_frame,reset_counter,quality
        """
        line = [float(_) for _ in line.strip().split(',')]

        timestamp = line[0] * self.multiplier
        t = np.array(line[2:5]).reshape(-1, 1)
        q = np.array(line[5:9])
        R = State.get_rotation_matrix_from_quaternion_vector(q)
        pose = Pose(R=R, t=t)
        return self.parsed_field(timestamp=timestamp, pose=pose, quality=line[27])
    
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
                    self.last_pose = data.pose
                    continue
                
                processed_timestamp = data.timestamp
                if self.simulate_delay:
                    delay = max(0, np.random.normal(self.avg_delay, 0.3)) * 1e9 # in nano seconds
                    if self.last_processed_timestamp > processed_timestamp:
                        time_delta = self.last_processed_timestamp - processed_timestamp
                        delay += time_delta
                    processed_timestamp += delay

                relative_pose = self.last_pose.inverse() * data.pose
                dt = (data.timestamp - self.last_timestamp) / 1e6
                self.last_processed_timestamp = processed_timestamp
                self.last_pose = data.pose

                yield self.field(
                    timestamp=processed_timestamp, 
                    relative_pose=relative_pose.matrix(pose_only=True),
                    dt=dt
                )

    def start_time(self):
        with open(self.path, 'r') as f:
            next(f)
            for line in f:
                return self.parse(line).timestamp

    def set_starttime(self, starttime):
        self.starttime = starttime

class UAVCustomVisualOdometryDataReader:
    def __init__(self):
        pass
        
class PX4_ActuatorMotorDataReader:
    def __init__(
            self, 
            path: str,
            model_path: str,
            starttime=-float('inf'),
            window_size=None,
        ):
        assert os.path.exists(model_path), "Poly model does not exists."
        coef = np.load(model_path)
        
        self.multiplier = 1000
        self.path = path
        self.starttime = starttime
        self.field = namedtuple('data', 
            ['timestamp', 'c0', 'c1', 'c2', 'c3'])
        
        self.poly_model = np.poly1d(coef)
        self.get_rotor_speed = lambda x: self.poly_model(x*100)
        
        self.window_size = window_size
        self.buffer = []
    
    def parse(self, line):
        """
        line:
            timestamp,timestamp_sample,control[0],control[1],control[2],control[3],control[4],control[5],control[6],control[7],control[8],control[9],control[10],control[11],reversible_flags
        """
        line = [float(_) for _ in line.strip().split(',')]
        
        timestamp = int(line[0]) * self.multiplier
        controls = np.array([self.get_rotor_speed(x) for x in line[2:6]])
        c0, c1, c2, c3 = controls
        return self.field(timestamp, c0, c1, c2, c3)
    
    def rolling_average(self, data):
        
        d = np.array([data.c0, data.c1, data.c2, data.c3])
        self.buffer.append(d)
        if len(self.buffer) > self.window_size:
            mean = np.mean(self.buffer, axis=0)
            self.buffer = self.buffer[-self.window_size:]
            return self.field(timestamp=data.timestamp, c0=mean[0], c1=mean[1], c2=mean[2], c3=mean[3])
            
        return self.field(timestamp=data.timestamp, c0=data.c0, c1=data.c1, c2=data.c2, c3=data.c3)
    
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
    
class PX4_ActuatorOutputDataReader:
    def __init__(
            self, 
            path: str,
            starttime=-float('inf'),
            window_size=None,
        ):
        self.path = path
        self.multiplier = 1000
        self.starttime = starttime
        self.field = namedtuple('data', 
            ['timestamp', 'c0', 'c1', 'c2', 'c3'])
        
        self.window_size = window_size
        self.buffer = []
    
    def parse(self, line):
        """
        line:
            timestamp,noutputs,output[0],output[1],output[2],output[3],output[4],output[5],output[6],output[7],output[8],output[9],output[10],output[11],output[12],output[13],output[14],output[15]
        """
        line = [float(_) for _ in line.strip().split(',')]
        
        timestamp = int(line[0]) * self.multiplier
        controls = np.array(line[2:6])
        c0, c1, c2, c3 = controls
        return self.field(timestamp, c0, c1, c2, c3)
    
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
    
    
class PX4_CustomIMUDataReader:
    """
        Read IMU data on PX4 side + Magnetometer data
    """
    def __init__(
            self, 
            path: str, 
            starttime=-float('inf'),
            window_size=None
        ):
        self.multiplier = 1000
        self.path = os.path.join(path, "imu_combined/9_axis_imu_data.csv")
        self.starttime = starttime
        self.field = namedtuple('data', 
            ['timestamp', 'a', 'w', 'm'])
        
        self.window_size = window_size
        self.buffer = []

    def parse(self, line):
        """
        line: 
            i,timestamp,ax,ay,az,wx,wy,wz,mx,my,mz
        """
        line = [float(_) for _ in line.strip().split(',')]

        timestamp = int(line[1]) * self.multiplier
        a = np.array(line[2:5])
        w = np.array(line[5:8])
        m = np.array(line[8:11])
        
        return self.field(timestamp, a, w, m)
    
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
    

if __name__ == "__main__":
    import yaml
    import os

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(os.path.dirname(base_dir), "data", "UAV")
    def _main_general():
        
        imu = PX4_IMUDataReader(
            gyro_path=os.path.join(data_path, "log0001/px4/09_00_22_sensor_gyro_0.csv"),
            acc_path=os.path.join(data_path, "log0001/px4/09_00_22_sensor_accel_0.csv"),
        )
        
        overlay = VOXL_QVIOOverlayDataReader(
            path=os.path.join(data_path, "log0001/run/mpa/qvio_overlay/data.csv")
        )
        
        motor = PX4_ActuatorMotorDataReader(
            path=os.path.join(data_path, "log0001/px4/09_00_22_actuator_motors_0.csv"),
            model_path=os.path.join(data_path, "models/poly_model_rpm.npy"),
        )
        
        gps = PX4_GPSDataReader(
            path=os.path.join(data_path, "log0001/px4/09_00_22_sensor_gps_0.csv"),
        )

        tracking_camera = VOXL_TrackingCameraDataReader(
            path=os.path.join(data_path, "log0001/run/mpa/tracking/data.csv")
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
        
        i = 0
        dataset = iter(tracking_camera)
        while True:
            try:
                data = next(dataset)
                print(data)
            except StopIteration:
                break
            i += 1
    
    def _main_visualize():
            
        imu = PX4_IMUDataReader(
            gyro_path="../../data/UAV/log0001/px4/09_00_22_sensor_gps_0.csv",
            acc_path="../../data/UAV/log0001/px4/09_00_22_sensor_accel_0.csv",
            
        )
        overlay = VOXL_QVIOOverlayDataReader(
            path="../../data/UAV/log0001/run/mpa/qvio_overlay/data.csv"
        )
        gps = PX4_GPSDataReader(
            path="../../data/UAV/log0001/px4/09_00_22_sensor_gps_0.csv"
        )
        motor = PX4_ActuatorMotorDataReader(
            path="../../data/UAV/log0001/px4/09_00_22_actuator_motors_0.csv",
        )
        
    
    _main_general()
    # _main_visualize()
        