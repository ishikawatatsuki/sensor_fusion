from enum import Enum

class MeasurementDataEnum(Enum):
    ALL_DATA = 1 # Feed all measurement data (suppose all data available meaning no tunnel, many features in the trajectory).
    DROPOUT = 2 # Feed measurement data when available (when unreliable measurement, don't feed the data).
    COVARIANCE = 3 # Feed entire measurement data with varying measurement error covariance matrix when dropout.

class SetupEnum(Enum):
    SETUP_1 = 1 
    SETUP_2 = 2 
    SETUP_3 = 3

    @staticmethod
    def get_names():
        return [e.name for e in SetupEnum]

class FilterEnum(Enum):
    EKF = 1 
    UKF = 2 
    PF = 3
    EnKF = 4
    CKF = 5

    @staticmethod
    def get_names():
        return [e.name for e in FilterEnum]

class NoiseTypeEnum(Enum):
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

class SamplingEnum(Enum):
    NORMAL_DATA = 1 
    DOWNSAMPLED_DATA = 2 
    UPSAMPLED_DATA = 3

    LOOSELY_COUPLED = 4

class Configs:
    decimal_place = 3
    processing_time_decimal_place = 5
    
class Dimensions:
    _2D = 2
    _3D = 3

class ErrorEnum(Enum):
    MAE = 1
    RMSE = 2
    MAX = 3

    @staticmethod
    def get_all():
        return [e for e in ErrorEnum]
    
    @staticmethod
    def get_names():
        return [e.name for e in ErrorEnum]