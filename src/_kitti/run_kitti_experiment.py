from logging import config
import os
import sys
import logging
from time import sleep
import numpy as np
import pandas as pd
from tqdm import tqdm
from enum import Enum
from typing import List
from itertools import product
from collections import namedtuple
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

from ..misc import setup_logging
from ..internal.error_reporter.kitti_error_reporter import evaluate_kitti_odom
from ..pipeline import SingleThreadedPipeline
from ..internal.extended_common import (
    FusionData,
    KITTI_SensorType,
    CoordinateFrame,
    SensorType,
    SensorConfig,
    ExtendedConfig,
    FilterConfig,
    DatasetConfig
)
from ..utils.geometric_transformer import TransformationField
from ..common.constants import KITTI_SEQUENCE_MAPS

Result = namedtuple('Field', ['gt_position', 'estimated_position', 'vo_position', 'mae', 'ate', 'rpe_m', 'rpe_deg', 'inference_time_update', 'inference_measurement_update'])