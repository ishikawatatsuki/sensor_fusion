from enum import Enum
class MeasurementDataEnum(Enum):
    """
        MeasurementDataEnum determines how to feed sensor data to a filter in measurement update step of a filtering process.
        - ALL_DATA:
            Feed all measurement data (suppose all data available meaning no tunnel, many features in the trajectory).
        - DROPOUT: 
            Feed measurement data when available (when unreliable measurement, don't feed the data).
        - COVARIANCE: 
            Once a dropout ratio is set for VO and GPS, data loader returns sensor value and different value of error covariance matrix, 
            assuming that tunnel or cycle slip in GPS and less feature points on a frame in VO, which means that the measured sensor data is uncertain. 
    """
    ALL_DATA = 1
    DROPOUT = 2
    COVARIANCE = 3

class SetupEnum(Enum):
    """
        SetupEnum defines the sensor setup.
        - SETUP_1: 
            IMU data is fed to a filter as a control input in time update step, and VO data is fed in measurement update step.
        - SETUP_2:
            IMU data is fed in time update step, VO and GPS data are fed in measurement update step.
        - SETUP_3:
            INS data, especially forward velocity and yaw angle are fed to the filter in time update step, and GPS data from INS is fed in the measurement update step.
    """
    SETUP_1 = 1 # IMU, VO
    SETUP_2 = 2 # IMU, VO + GPS
    SETUP_3 = 3 # INS

    @staticmethod
    def get_names():
        return [e.name for e in SetupEnum]
    
    @staticmethod
    def get_name(setup):
        match setup:
            case SetupEnum.SETUP_1:
                return "Setup1 (IMU, VO)"
            case SetupEnum.SETUP_2:
                return "Setup2 (IMU, VO+GPS)"
            case SetupEnum.SETUP_3:
                return "Setup3 (INS)"
            case _:
                return ""

class FilterEnum(Enum):
    """
        FilterEnum specifies a filter
        - EKF:
            Extended Kalman Filter
        - UKF:
            Unscented Kalman Filter
        - PF:
            Particle Filter
        - EnKF:
            Ensemble Kalman Filter
        - CKF:
            Cubature Kalman Filter
    """
    EKF = 1 
    UKF = 2 
    PF = 3
    EnKF = 4
    CKF = 5

    @staticmethod
    def get_names():
        return [e.name for e in FilterEnum]
    
    @staticmethod
    def list_approximation_based_filters():
        return [FilterEnum.EKF, FilterEnum.UKF, FilterEnum.CKF]
    
    @staticmethod
    def list_sampling_based_filters():
        return [FilterEnum.PF, FilterEnum.EnKF]

class NoiseTypeEnum(Enum):
    """
        NoiseTypeEnum determines which noise 
    """
    DEFAULT = 1
    CURRENT = 2
    OPTIMIZED = 3

    @staticmethod
    def get_names():
        return [e.name for e in NoiseTypeEnum]
    
    @staticmethod
    def get_suffix():
        return [suffix[e] for e in NoiseTypeEnum]

suffix = {
    NoiseTypeEnum.DEFAULT: "_default",
    NoiseTypeEnum.CURRENT: "_current",
    NoiseTypeEnum.OPTIMIZED: "_optimized"
}
class ErrorEnum(Enum):
    """
        ErrorEnum specifies an error metric.
        - MAE:
            Mean Absolute Error
        - RMSE:
            Root Mean Square Error
        - MAX:
            Maximum error in the entire trajectory
    """
    MAE = 1
    RMSE = 2
    MAX = 3

    @staticmethod
    def get_all():
        return [e for e in ErrorEnum]
    
    @staticmethod
    def get_names():
        return [e.name for e in ErrorEnum]

class SamplingEnum(Enum):
    DEFAULT_DATA = 1 
    DOWNSAMPLED_DATA = 2 
    UPSAMPLED_DATA = 3

class Configs:
    decimal_place = 3
    processing_time_decimal_place = 5
    declination_angle = 9.98 # magnetic declination angle in Estonia represented in degree
    
class IMU_Type(Enum):
    OXTS = "oxts"
    IMU0 = "icm_42688_p"
    IMU1 = "icm_20948"
    IMU2 = "icm_20602"
    IMU3 = "icm_42688"
    
    @staticmethod
    def is_voxl(type: str):
        return type in [IMU_Type.IMU0.value, IMU_Type.IMU1.value]
    
    @staticmethod
    def is_px4(type: str):
        return type in [IMU_Type.IMU2.value, IMU_Type.IMU3.value]

class CoordinateSystemEnum(Enum):
    ENU = "ENU"
    NED = "NED"

class SensorType(Enum):
    
    VOXL_IMU0 = "voxl_imu0"
    VOXL_IMU1 = "voxl_imu1"
    VOXL_QVIO = "voxl_qvio"
    VOXL_STEREO = "voxl_stereo"
    
    PX4_IMU0 = "px4_imu0"
    PX4_IMU1 = "px4_imu1"
    PX4_GPS = "px4_gps"
    PX4_IMU0_BIAS = "px4_imu0_bias"
    PX4_IMU1_BIAS = "px4_imu1_bias"
    PX4_MAG = "px4_mag"
    PX4_VO = "px4_vo"
    PX4_VEHICLE_ODOM = "px4_vehicle_odom"
    PX4_ACTUATOR_MOTORS = "px4_actuator_motors"
    PX4_ACTUATOR_OUTPUTS = "px4_actuator_outputs"

class DatasetType(Enum):
    
    KITTI = "KITTI"
    CUSTOM_UAV = "CUSTOM_UAV"
    # EUROC_MAV = "EUROC_MAV"
    