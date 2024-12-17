from enum import (
    Enum, 
    IntEnum,
    auto
)
from typing import Callable, Union
from extendableenum import inheritable_enum

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
    UAV = "UAV"
    # KAGARU = "KAGARU"
    # OAK = "OAK"
    # EUROC = "EUROC"
    # ROSARIO = "ROSARIO"
    # TUMVI = "TUMVI"
    # BLACKBIRD = "BLACKBIRD"
    # KUMAR = "KUMAR"
    
    VIZTRACK = "VIZTRACK"
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
            case DatasetType.UAV:
                return CoordinateSystem.NED
            case DatasetType.VIZTRACK:
                return CoordinateSystem.ENU
            case _:
                return CoordinateSystem.ENU


@inheritable_enum
class KITTI_SensorType(IntEnum):
    OXTS_IMU = auto()
    OXTS_INS = auto()
    OXTS_GPS = auto()
    KITTI_STEREO = auto()
    
    KITTI_CUSTOM_VO = auto()
    KITTI_COLOR_IMAGE = auto()
    
    
    KITTI_CUSTOM_VO_VELOCITY_ONLY = auto()
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

@inheritable_enum
class UAV_SensorType(IntEnum):
    VOXL_IMU0 = auto()
    VOXL_IMU1 = auto()
    VOXL_QVIO = auto()
    VOXL_STEREO = auto()
    VOXL_QVIO_OVERLAY = auto()
    
    PX4_IMU0 = auto()
    PX4_IMU1 = auto()
    PX4_GPS = auto()
    PX4_IMU0_BIAS = auto()
    PX4_IMU1_BIAS = auto()
    PX4_MAG = auto()
    PX4_VO = auto()
    PX4_VEHICLE_ODOM = auto()
    PX4_ACTUATOR_MOTORS = auto()
    PX4_ACTUATOR_OUTPUTS = auto()
    
    UAV_CUSTOM_VO = auto()
    
    UWB = auto()
    LIDAR = auto()
    
    @staticmethod
    def get_enum_name_list():
        return [s.lower() for s in list(UAV_SensorType.__members__.keys())]
    
    @classmethod
    def get_uav_sensor_from_str(cls, sensor_str: str):
        s = sensor_str.lower()
        try: 
            index = UAV_SensorType.get_enum_name_list().index(s)
            return cls(index + 1)
        except:
            return None
        
@inheritable_enum
class Viztrack_SensorType(IntEnum):
    OAK_D_IMU = auto()
    OAK_D_STEREO = auto()
    Viztrack_GPS = auto()
    
    VIZTRACK_UPWARD_LEFTWARD_VELOCITY = auto()
    
    @staticmethod
    def get_enum_name_list():
        return [s.lower() for s in list(Viztrack_SensorType.__members__.keys())]
    
    @classmethod
    def get_viztrack_sensor_from_str(cls, sensor_str: str):
        s = sensor_str.lower()
        try: 
            index = Viztrack_SensorType.get_enum_name_list().index(s)
            return cls(index + 1)
        except:
            return None
    
# @copy_enum_members(KITTI_SensorType, UAV_SensorType)
class SensorType(UAV_SensorType, KITTI_SensorType, Viztrack_SensorType):
    
    GROUND_TRUTH = 100
    
    @staticmethod
    def is_time_update(t):
        return t.name in [
            SensorType.VOXL_IMU0.name, 
            SensorType.VOXL_IMU1.name, 
            SensorType.PX4_IMU0.name, 
            SensorType.PX4_IMU1.name,
            SensorType.OXTS_IMU.name,
            SensorType.OXTS_INS.name,
            SensorType.OAK_D_IMU.name
        ]
    @staticmethod
    def is_stereo_image_data(t):
        return t.name in [
            SensorType.VOXL_STEREO.name,
            SensorType.KITTI_STEREO.name,
            SensorType.OAK_D_STEREO.name
        ]
    
    @staticmethod
    def is_visualization_data(t):
        return t.name in [
            SensorType.VOXL_QVIO_OVERLAY.name,
            SensorType.KITTI_COLOR_IMAGE.name
        ]
    
    @staticmethod
    def is_vo_data(t):
        return t.name in [
            SensorType.PX4_VO.name,
            SensorType.KITTI_CUSTOM_VO.name
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
    def get_sensor_from_str_func(d: str) -> Callable[[str], Union[KITTI_SensorType, UAV_SensorType, None]]:
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
            case DatasetType.UAV:
                return UAV_SensorType.get_uav_sensor_from_str
            case DatasetType.VIZTRACK:
                return Viztrack_SensorType.get_viztrack_sensor_from_str
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
            case DatasetType.UAV:
                return UAV_SensorType(sensor_id).name
            case DatasetType.VIZTRACK:
                return Viztrack_SensorType(sensor_id).name
            case DatasetType.EXPERIMENT:
                return KITTI_SensorType(sensor_id).name
            case _:
                return ""

class MotionModel(Enum):
    KINEMATICS = "KINEMATICS"
    VELOCITY = "VELOCITY"
    
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
            

if __name__ == "__main__":
    from queue import PriorityQueue
    
    s = KITTI_SensorType.get_kitti_sensor_from_str("oxts_imu")
    q = PriorityQueue()
    q.put((1, SensorType.OXTS_IMU))
    q.put((1, SensorType.OXTS_GPS))
    
    print(q.get())
    print(q.get())
    
    q.put((2, SensorType.VOXL_IMU0))
    q.put((2, SensorType.OXTS_IMU))
    q.put((2, SensorType.GROUND_TRUTH))
    
    print(q.get())
    print(q.get())
    print(q.get())
    
    print(SensorType.OXTS_IMU is SensorType.VOXL_IMU0)
    print(SensorType.OXTS_IMU is SensorType.GROUND_TRUTH)
    
    print(SensorType.GROUND_TRUTH)
    print(KITTI_SensorType.KITTI_CUSTOM_VO is SensorType.PX4_IMU0)
    
    print(SensorType.is_time_update(KITTI_SensorType.KITTI_CUSTOM_VO))
    print(UAV_SensorType(1).name)
    name = SensorType.get_sensor_name("kitti", KITTI_SensorType.KITTI_CUSTOM_VO.value)
    print(DatasetType.get_coordinate_system("kitti"))

    print(Viztrack_SensorType.get_viztrack_sensor_from_str("OAK_D_IMU"))