from enum import (
    Enum, 
    IntEnum,
    auto
)
from typing import Callable, Union
from extendableenum import inheritable_enum
import numpy as np
from typing import Union
from dataclasses import dataclass
from scipy.spatial.transform import Rotation
import multiprocessing as mp
from scipy.spatial.transform import Rotation

class FilterType(Enum):
    """
        FilterType specifies a filter
        - EKF:
            Extended Kalman Filter
        - UKF:
            Unscented Kalman Filter
        - PF:
            Particle Filter
        - ENKF:
            Ensemble Kalman Filter
        - CKF:
            Cubature Kalman Filter
    """
    EKF = "EKF"
    UKF = "UKF"
    PF = "PF"
    EnKF = "ENKF"
    CKF = "CKF"
    SRCKF = "SRCKF"
    
    @classmethod
    def get_filter_type_from_str(cls, filter_type: str):
        d = filter_type.upper()
        try: 
            return cls(d)
        except:
            return None
    
    @staticmethod
    def is_gaussian_based_filter(filter_type: str) -> bool:
        f = FilterType.get_filter_type_from_str(filter_type=filter_type)
        return f is FilterType.EKF or f is FilterType.UKF or f is FilterType.CKF
    
    @staticmethod
    def is_probabilistic_filter(filter_type: str) -> bool:
        f = FilterType.get_filter_type_from_str(filter_type=filter_type)
        return f is FilterType.PF or f is FilterType.EnKF
    
class CoordinateSystem(Enum):
    ENU = "ENU"
    NED = "NED"

class DatasetType(Enum):
    
    KITTI = "KITTI"

    # UAV = "UAV"
    # EUROC = "EUROC"
    # KAGARU = "KAGARU"
    # OAK = "OAK"
    # ROSARIO = "ROSARIO"
    # TUMVI = "TUMVI"
    # BLACKBIRD = "BLACKBIRD"
    # KUMAR = "KUMAR"
    
    EXPERIMENT = "EXPERIMENT"

    @staticmethod
    def get_names():
        return [e.name for e in DatasetType]

    @classmethod
    def get_type_from_str(cls, dataset: str):
        d = dataset.upper()
        try: 
            return cls(d)
        except:
            return None
    
    @staticmethod
    def get_coordinate_system(dataset: str) -> CoordinateSystem:
        d = DatasetType.get_type_from_str(dataset)
        match(d):
            case DatasetType.KITTI:
                return CoordinateSystem.ENU
            # case DatasetType.UAV:
            #     return CoordinateSystem.NED
            # case DatasetType.EUROC:
            #     return CoordinateSystem.NED
            case _:
                return CoordinateSystem.ENU


@inheritable_enum
class KITTI_SensorType(IntEnum):
    OXTS_IMU = auto()
    OXTS_GPS = auto()
    KITTI_STEREO = auto()
    
    OXTS_IMU_UNSYNCED = auto()
    OXTS_GPS_UNSYNCED = auto()
    KITTI_VO = auto()
    
    KITTI_COLOR_IMAGE = auto()
    KITTI_UPWARD_LEFTWARD_VELOCITY = auto()
    
    @staticmethod
    def get_enum_name_list():
        return [s.lower() for s in list(KITTI_SensorType.__members__.keys())]
    
    @classmethod
    def get_kitti_sensor_from_str(cls, sensor_str: str):
        s = sensor_str.lower()
        try: 
            index = KITTI_SensorType.get_enum_name_list().index(s)
            return cls(index + 1)
        except:
            return None


# @copy_enum_members(KITTI_SensorType, UAV_SensorType)
class SensorType(KITTI_SensorType):
    
    GROUND_TRUTH = 100

    @staticmethod
    def get_all_sensors():
        all_members = []
        all_members += KITTI_SensorType.__members__.values()
        all_members += [SensorType.GROUND_TRUTH]
        return all_members
    
    @staticmethod
    def is_imu_data(t):
        return t.name in [
            SensorType.OXTS_IMU.name
        ]
    
    @staticmethod
    def is_time_update(t):
        return SensorType.is_imu_data(t)
    
    @staticmethod
    def is_positioning_data(t):
        return SensorType.is_gps_data(t)

    @staticmethod
    def is_gps_data(t):
        return t.name in [
            SensorType.OXTS_GPS.name,
        ]
    
    @staticmethod
    def is_stereo_image_data(t):
        return t.name in [
            SensorType.KITTI_STEREO.name,
        ]
    
    @staticmethod
    def is_visualization_data(t):
        return t.name in [
            SensorType.KITTI_COLOR_IMAGE.name
        ]
    
    @staticmethod
    def is_vo_data(t):
        return t.name in [
            SensorType.KITTI_VO.name
        ]
    
    @staticmethod
    def is_reference_data(t):
        return t.name is SensorType.GROUND_TRUTH.name
        
    @staticmethod
    def is_measurement_update(t):
        return not SensorType.is_time_update(t) and\
            not SensorType.is_stereo_image_data(t) and\
                not SensorType.is_reference_data(t) and\
                    not SensorType.is_visualization_data(t)

    @staticmethod
    def get_sensor_from_str_func(d: str) -> Callable[[str], Union[KITTI_SensorType, None]]:
        """return function pointer based on passed dataset type

        Args:
            d (str): dataset type, either kitti or uav

        Returns:
            Callable[[str], Union[KITTI_SensorType, UAV_SensorType, None]]: pointer to a function
        """
        dataset = DatasetType.get_type_from_str(d)
        match(dataset):
            case DatasetType.KITTI:
                return KITTI_SensorType.get_kitti_sensor_from_str
            case _:
                return KITTI_SensorType.get_kitti_sensor_from_str
            
    @staticmethod
    def get_sensor_name(d: str, sensor_id: int):
        dataset = DatasetType.get_type_from_str(d)
        if sensor_id == SensorType.GROUND_TRUTH.value:
            return "GROUND_TRUTH"
        
        match(dataset):
            case DatasetType.KITTI:
                return KITTI_SensorType(sensor_id).name
            case DatasetType.EXPERIMENT:
                return KITTI_SensorType(sensor_id).name
            case _:
                return ""
            
        
class CoordinateFrame(Enum):
    IMU = "IMU"
    GPS = "GPS"
    LEICA = "LEICA"
    BEACON = "BEACON"
    STEREO_LEFT = "STEREO_LEFT"
    STEREO_RIGHT = "STEREO_RIGHT"
    MAGNETOMETER = "MAGNETOMETER"

    INERTIAL = "INERTIAL"  # Inertial world frame

class MotionModel(Enum):
    KINEMATICS = "KINEMATICS"
    VELOCITY = "VELOCITY"
    DRONE_KINEMATICS = "DRONE_KINEMATICS"
    
    @classmethod
    def get_motion_model(cls, s: str):
        try:
            return cls(s.upper())
        except:
            return MotionModel.KINEMATICS

class NoiseType(Enum):
    DEFAULT = "DEFAULT"
    OPTIMAL = "OPTIMAL"
    DYNAMIC = "DYNAMIC"
    
    @classmethod
    def get_noise_type_from_str(cls, s: str):
        try:
            return cls(s.upper())
        except:
            return NoiseType.DEFAULT
            
class SignalType(Enum):
    ACC = "acc"
    GYRO = "gyro"

class State:
    
    def __init__(
        self,
        p: np.ndarray, # position of body in world frame
        v: np.ndarray, # velocity of body in world frame
        q: np.ndarray, # rotation from the world frame to the body frame
        b_w: np.ndarray, # bias of angular velocity
        b_a: np.ndarray, # bias of acceleration
        w: np.ndarray = None, # angular velocity of body in body frame
    ):
        self.p = p if p.ndim == 2 else p.reshape(-1, 1)
        self.v = v if v.ndim == 2 else v.reshape(-1, 1)
        self.q = q if q.ndim == 2 else q.reshape(-1, 1)
        self.b_w = b_w if b_w.ndim == 2 else b_w.reshape(-1, 1) 
        self.b_a = b_a if b_a.ndim == 2 else b_a.reshape(-1, 1) 

        self.w = w
        if self.w is not None:
            self.w = self.w if self.w.ndim == 2 else self.w.reshape(-1, 1)
        
    def __str__(self):
        s = f'State:\n\
            \tp: {self.p.flatten()}\n\
            \tv: {self.v.flatten()}\n\
            \tq: {self.q.flatten()}\n\
            \tb_w: {self.b_w.flatten()}\n\
            \tb_a: {self.b_a.flatten()}\n'

        if self.w is not None:
            s += f'\tw: {self.w.flatten()}\n'
        
        return s

    @classmethod
    def get_initial_state_from_config(cls, filter_config):
        """create initial state object

        Args:
            config (dict): configuration dictionary

        Returns:
            State: new state object
        """
        motion_model = MotionModel.get_motion_model(filter_config.motion_model)
        
        p = np.zeros((3, 1))
        v = np.zeros((3, 1))
        q = np.array([1.0, 0.0, 0.0, 0.0]).reshape(-1, 1)
        b_w = np.zeros((3, 1))
        b_a = np.zeros((3, 1))

        w = None
        if motion_model == MotionModel.DRONE_KINEMATICS:
            w = np.zeros((3, 1))

        return cls(p=p, v=v, q=q, w=w, b_w=b_w, b_a=b_a)

    @classmethod
    def get_new_state_from_array(cls, x: np.ndarray):
        """create new state object

        Args:
            x (np.ndarray): current state vector as a numpy array
            motion_model (MotionModel): motion model enum

        Returns:
            State: new state object
        """
        x = x.flatten()
        p = x[:3].reshape(-1, 1)
        v = x[3:6].reshape(-1, 1)
        q = x[6:10].reshape(-1, 1)
        b_w = x[10:13].reshape(-1, 1)
        b_a = x[13:16].reshape(-1, 1)

        w = x[16:19].reshape(-1, 1)
        w = w if w.shape[0] == 3 else None

        return cls(p=p, v=v, q=q, w=w, b_w=b_w, b_a=b_a)
        
    def get_state_vector(self) -> np.ndarray:
        vec = np.vstack([self.p, self.v, self.q, self.b_w, self.b_a])
        if self.w is not None:
            vec = np.vstack([vec, self.w])
        
        return vec
    
    def get_vector_size(self) -> int:
        return self.get_state_vector().shape[0]

    def skew(self, vec):
        """
        Create a skew-symmetric matrix from a 3-element vector.
        """
        x, y, z = vec
        return np.array([
            [0, z, -y],
            [-z, 0, x],
            [y, -x, 0]])

    def get_rotation_matrix(self, q=None):
        if q is None:
            q = self.q
        
        # w, x, y, z = q.flatten()
        # # return Rotation.from_quat([x, y, z, w]).as_matrix()
        # R = np.array([
        #     [1 - 2*(y**2 + z**2),     2*(x*y - w*z),       2*(x*z + w*y)],
        #     [2*(x*y + w*z),           1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
        #     [2*(x*z - w*y),           2*(y*z + w*x),       1 - 2*(x**2 + y**2)]
        # ])
        # return R
        # q = q / np.linalg.norm(q)
        # q = q.flatten()
        # vec = q[1:]
        # w = q[0]

        # R = (2*w*w-1)*np.identity(3) - 2*w*self.skew(vec) + 2*vec[:, None]*vec
        # return R
        return State.get_rotation_matrix_from_quaternion_vector(q.flatten())
    
    def get_euler_angle_from_quaternion(self, q=None):
        if q is None:
            q = self.q.flatten()
            
        # w, x, y, z = q.flatten()
        # return np.array([
        #     [1 - 2*(y**2 + z**2), 2*(x*y - w*z),     2*(x*z + w*y)],
        #     [2*(x*y + w*z),       1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
        #     [2*(x*z - w*y),       2*(y*z + w*x),     1 - 2*(x**2 + y**2)]
        # ])
        return State.get_euler_angle_from_quaternion_vector(q.flatten())

    @staticmethod
    def get_euler_angle_from_quaternion_vector(q) -> np.ndarray:
        w, x, y, z = q
        
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x**2 + y**2)
        phi = np.arctan2(t0, t1)
    
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        theta = np.arcsin(t2)
    
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y**2 + z**2)
        psi = np.arctan2(t3, t4)
        
        return np.array([phi, theta, psi])
        
    @staticmethod
    def get_rotation_matrix_from_quaternion_vector(q) -> np.ndarray:
        q0, q1, q2, q3 = q
        # https://ahrs.readthedocs.io/en/latest/filters/ekf.html
        # https://www.iri.upc.edu/people/jsola/JoanSola/objectes/notes/kinematics.pdf
        return np.array([
            [q0**2 + q1**2 - q2**2 - q3**2, 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)],
            [2*(q1*q2 + q0*q3), q0**2 - q1**2 + q2**2 - q3**2, 2*(q2*q3 - q0*q1)],
            [2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), q0**2 - q1**2 - q2**2 + q3**2]
        ])

    @staticmethod
    def get_quaternion_from_rotation_matrix(R: np.ndarray) -> np.ndarray:
        assert R.shape == np.eye(3).shape, "Please provide 3x3 matrix"
        trace = np.trace(R)

        if trace > 0:
            S = 2.0 * np.sqrt(trace + 1.0)
            w = 0.25 * S
            x = (R[2, 1] - R[1, 2]) / S
            y = (R[0, 2] - R[2, 0]) / S
            z = (R[1, 0] - R[0, 1]) / S
        elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / S
            x = 0.25 * S
            y = (R[0, 1] + R[1, 0]) / S
            z = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / S
            x = (R[0, 1] + R[1, 0]) / S
            y = 0.25 * S
            z = (R[1, 2] + R[2, 1]) / S
        else:
            S = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / S
            x = (R[0, 2] + R[2, 0]) / S
            y = (R[1, 2] + R[2, 1]) / S
            z = 0.25 * S

        return np.array([w, x, y, z]).reshape(-1, 1)
        
        # r11, r12, r13 = R[0, :]
        # r21, r22, r23 = R[1, :]
        # r31, r32, r33 = R[2, :]
        # _q0 = np.sqrt((1+r11+r22+r33)/4)
        # _q1 = np.sqrt((1+r11-r22-r33)/4)
        # _q2 = np.sqrt((1-r11+r22-r33)/4)
        # _q3 = np.sqrt((1-r11-r22+r33)/4)
        # if _q0 > _q1 and _q0 > _q2 and _q0 > _q3:
        #   div = 4*_q0
        #   _q1 = (r31-r23)/div
        #   _q2 = (r13-r31)/div
        #   _q3 = (r21-r12)/div
        # elif _q1 > _q0 and _q1 > _q2 and _q1 > _q3:
        #   div = 4*_q1
        #   _q0 = (r32-r23)/div
        #   _q2 = (r12-r21)/div
        #   _q3 = (r13-r31)/div
        # elif _q2 > _q0 and _q2 > _q1 and _q2 > _q3:
        #   div = 4*_q2
        #   _q0 = (r13-r31)/div
        #   _q1 = (r12-r21)/div
        #   _q3 = (r23-r32)/div
        # else:
        #   div = 4*_q3
        #   _q0 = (r21-r12)/div
        #   _q1 = (r13-r31)/div
        #   _q2 = (r23-r32)/div
            
        # return np.array([_q0, _q1, _q2, _q3]).reshape(-1, 1)
        
    @staticmethod
    def get_euler_angle_from_rotation_matrix(R: np.ndarray) -> np.ndarray:
        assert R.shape == np.eye(3).shape, "Please provide 3x3 matrix"
        
        r = Rotation.from_matrix(R)
        angles = r.as_euler('xyz', degrees=False)
        
        return angles.reshape(-1, 1)
        
    @staticmethod
    def get_quaternion_from_euler_angle(w: np.ndarray) -> np.ndarray:
        w = w.flatten()
        assert w.shape[0] == 3, "Please provide 3d vector"
        
        normalize_euler = lambda angle: (angle + np.pi) % (2 * np.pi) - np.pi
        
        roll, pitch, yaw = w
        
        roll = normalize_euler(roll)
        pitch = normalize_euler(pitch)
        yaw = normalize_euler(yaw)
            
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return np.array([w, x, y, z]).reshape(-1, 1)
    

class Pose:
  """
  3d rigid transform.
  """
  def __init__(self, R, t):
    assert R.shape == np.eye(3).shape, "Please set 3x3 rotation matrix"
    assert t.shape[0] == 3, "Please set 3x1 translation vector"
    
    self.R = R
    self.t = t if len(t.shape) == 1 else t.reshape(-1, )

  def matrix(self, pose_only=False):
    m = np.zeros((3, 4)) if pose_only else np.identity(4)
    m[:3, :3] = self.R
    m[:3, 3] = self.t
    return m

  def inverse(self):
    return Pose(self.R.T, -self.R.T @ self.t)

  def __mul__(self, T1):
    R = self.R @ T1.R
    t = self.R @ T1.t + self.t
    return Pose(R, t)

  @classmethod
  def from_state(cls, state: State):
    R = state.get_rotation_matrix()
    t = state.p.flatten()
    return cls(R=R, t=t)
  
  def from_ned_to_enu(self):
    return Pose(R=np.array([
      [0, 1, 0],
      [1, 0, 0],
      [0, 0, -1]
    ]), t=np.zeros((3, 1))) * Pose(R=self.R, t=self.t)
    
  def from_enu_to_ned(self):
    return Pose(R=np.array([
      [0, 1, 0],
      [1, 0, 0],
      [0, 0, -1]
    ]), t=np.zeros((3, 1))) * Pose(R=self.R, t=self.t)

@dataclass
class SensorConfig:
    name: str
    dropout_ratio: float
    window_size: int
    args: dict

    def __init__(self,
                 name: str,
                 dropout_ratio: float = 0.,
                 window_size: int = 1,
                 args: dict = {}
                 ):
        self.name = name
        self.dropout_ratio = dropout_ratio
        self.window_size = window_size
        self.args = args

@dataclass
class ExtendedSensorConfig(SensorConfig):
    sensor_type: SensorType
    
    def __init__(
        self, 
        sensor_type: SensorType,
        *args,
        **kwargs
        ):
        super().__init__(*args, **kwargs)
        self.sensor_type = sensor_type

@dataclass
class IMUSensorErrors:
    acc_bias: np.ndarray
    gyro_bias: np.ndarray
    acc_noise: np.ndarray
    gyro_noise: np.ndarray

    def __str__(self):
        return f'IMUSensorErrors:\n\
            \tacc_bias: {self.acc_bias.flatten()}\n\
            \tgyro_bias: {self.gyro_bias.flatten()}\n\
            \tacc_noise: {self.acc_noise.flatten()}\n\
            \tgyro_noise: {self.gyro_noise.flatten()}\n'

@dataclass
class ControlInput:
    u: np.ndarray
    dt: float

@dataclass
class TimeUpdateField(ControlInput):
    Q: np.ndarray

    def __init__(self, Q: np.ndarray, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Q = Q

@dataclass
class MeasurementUpdateField:
    z: np.ndarray
    R: np.ndarray
    sensor_type: SensorType

@dataclass
class StereoField:
    left_frame_id: str
    right_frame_id: str

@dataclass
class CustomVOUpdateField:
    delta_pose: Pose
    velocity: np.ndarray
    R: np.ndarray
    sensor_type: SensorType
    
@dataclass
class ImageData:
    image: np.ndarray
    timestamp: float

@dataclass
class VisualOdometryData:
    last_pose: Pose
    relative_pose: Pose
    confidence: float
    timestamp: int
    received_timestamp: int
    processed_timestamp: int

@dataclass
class SensorData:
    z: np.ndarray

@dataclass
class SensorDataField:
    type: SensorType
    timestamp: int
    data: Union[
        ControlInput, 
        SensorData, 
        StereoField, 
        VisualOdometryData]
    coordinate_frame: CoordinateFrame

@dataclass
class InitialState:
    x: State
    P: np.ndarray

class VisualizationDataType(IntEnum):
    IMAGE = auto()
    ESTIMATION = auto()  # Fusion estimation
    GPS = auto()
    VO = auto()
    BEACON = auto()
    ACCELEROMETER = auto()
    GYROSCOPE = auto()
    VELOCITY = auto()
    ANGLE = auto()
    LEICA = auto()


    STATE = auto()

    GROUND_TRUTH = auto()

    @staticmethod
    def get_enum_name_list():
        return [
            s.lower() for s in list(VisualizationDataType.__members__.keys())
        ]

    @classmethod
    def get_type(cls, s: str):
        s = s.lower()
        try:
            index = VisualizationDataType.get_enum_name_list().index(s)
            return cls(index + 1)
        except:
            return None

@dataclass
class VisualizationQueue:
    name: str
    type: VisualizationDataType
    queue: mp.Queue


@dataclass
class VisualizerInterface:
    queues: list[VisualizationQueue]
    num_cols: int
    window_title: str


@dataclass
class VisualizationData:
    data: np.ndarray
    timestamp: int
    extra: str

    def __init__(self, data: np.ndarray, timestamp: int, extra: str = None):
        self.data = data
        self.timestamp = timestamp
        self.extra = extra

@dataclass
class FusionResponse:
    pose: np.ndarray
    timestamp: int
    
    imu_acceleration: np.ndarray
    imu_angular_velocity: np.ndarray
    estimated_angle: np.ndarray
    estimated_linear_velocity: np.ndarray

    vo_data: np.ndarray
    gps_data: np.ndarray
    geo_fencing_data: np.ndarray
    leica_data: np.ndarray


    def __init__(
            self,
            pose: np.ndarray = None,
            timestamp: int = None,
            imu_acceleration: np.ndarray = None,
            imu_angular_velocity: np.ndarray = None,
            estimated_angle: np.ndarray = None,
            estimated_linear_velocity: np.ndarray = None,
            vo_data: np.ndarray = None,
            gps_data: np.ndarray = None,
            geo_fencing_data: np.ndarray = None,
            leica_data: np.ndarray = None
    ):
        self.pose = pose
        self.timestamp = timestamp
        self.imu_acceleration = imu_acceleration
        self.imu_angular_velocity = imu_angular_velocity
        self.estimated_angle = estimated_angle
        self.estimated_linear_velocity = estimated_linear_velocity

        self.vo_data = vo_data
        self.gps_data = gps_data
        self.geo_fencing_data = geo_fencing_data
        self.leica_data = leica_data
