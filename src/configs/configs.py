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
    NORMAL_DATA = 1 
    DOWNSAMPLED_DATA = 2 
    UPSAMPLED_DATA = 3

class Configs:
    decimal_place = 3
    processing_time_decimal_place = 5