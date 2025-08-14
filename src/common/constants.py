from numpy import pi
from .datatypes import SensorType

DECLINATION_OFFSET_RADIAN_IN_ESTONIA = (9.98 + (1/60 * 59)) * (pi / 180)

MAX_CONSECUTIVE_DROPOUT_RATIO = 0.15


KITTI_SEQUENCE_MAPS = {
    "0027": "00",
    "0042": "01",
    "0034": "02",
    "0067": "03",
    "0016": "04",
    "0018": "05",
    "0020": "06",
    "0027": "07",
    "0028": "08",
    "0033": "09",
    "0034": "10",
}

KITTI_SEQUENCE_TO_DATE = {
    "00": "2011_10_03",
    "01": "2011_10_03",
    "02": "2011_10_03",
    "03": "2011_09_26",
    "04": "2011_09_30",
    "05": "2011_09_30",
    "06": "2011_09_30",
    "07": "2011_09_30",
    "08": "2011_09_30",
    "09": "2011_09_30",
    "10": "2011_09_30",
}

KITTI_SEQUENCE_TO_DRIVE = {
    "00": "0027",
    "01": "0042",
    "02": "0034",
    "03": "0067",
    "04": "0016",
    "05": "0018",
    "06": "0020",
    "07": "0027",
    "08": "0028",
    "09": "0033",
    "10": "0034",
}

KITTI_GEOMETRIC_LIMITATIONS = {
    "0016": {"min": [], "max": []},
    "0033": {"min": [-100, -400, -30], "max": [600, 200, 15]},
    "default": {"min": [0, 0, 0], "max": [100, 100, 20]}
}   

IMU_FREQUENCY_MAP = {
    SensorType.OXTS_IMU.name: 10.0,
    SensorType.OXTS_IMU_UNSYNCED.name: 100.0,
    SensorType.EuRoC_IMU.name: 200.0,
}