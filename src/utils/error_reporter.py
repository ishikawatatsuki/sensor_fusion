import os
import sys
import logging
import datetime
import numpy as np
import pandas as pd
from typing import List
from enum import Enum
from collections import namedtuple
from sklearn.metrics import mean_squared_error, mean_absolute_error
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from config import FilterConfig, DatasetConfig, ReportConfig
from interfaces import State, Pose
from custom_types import DatasetType
from constants import KITTI_SEQUENCE_MAPS

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s > %(message)s', 
                        datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG)

GeneralErrorReport = namedtuple('GeneralErrorReport', ['mae', 'rmse', 'max'])
PoseErrorReport = namedtuple('PoseErrorReport', [])

class ErrorReportType(Enum):
    GENERAL = "GENERAL"
    POSE = "POSE"
    POSE_IN_CAM = "POSE_IN_CAM"
    

class ErrorReporter:
    
    decimal_place = 3
    
    def __init__(
            self, 
            report_config: ReportConfig,
            filter_config: FilterConfig,
            dataset_config: DatasetConfig,
        ):
        
        self.config = report_config
        self.filter_config = filter_config
        self.dataset_config = dataset_config
        self.dataset = DatasetType.get_type_from_str(self.dataset_config.type)
        if self.dataset is DatasetType.KITTI or self.dataset is DatasetType.EXPERIMENT:
            self.seqence = KITTI_SEQUENCE_MAPS.get(self.dataset_config.variant)
        
        self.general_error_path = os.path.join(self.config.error_output_root_path, "errors/general")
        self.pose_error_path = os.path.join(self.config.error_output_root_path, "errors/poses")
        
        #only used in kitti
        self.pose_error_in_cam_path = os.path.join(
            self.config.error_output_root_path, 
            "errors/errors_for_eval",
            self.config.kitti_pose_result_folder
        )
        logger.info(self.pose_error_in_cam_path)
        
        self.estimated_poses = []
        self.referenced_poses = []
        
        self.estimated_poses_in_cam = []
        
        self.estimated_trajectory = []
        self.referenced_trajectory = []
        
    def set_trajectory(
            self, 
            estimated: np.ndarray, 
            expected: np.ndarray, 
            type: ErrorReportType=ErrorReportType.GENERAL
        ):
        
        match(type):
            case ErrorReportType.GENERAL:
                self.estimated_trajectory.append(estimated)
                self.referenced_trajectory.append(expected)
                
                return mean_absolute_error(estimated, expected) > 100.
            case ErrorReportType.POSE:
                self.estimated_poses.append(estimated.flatten())
                self.referenced_poses.append(expected.flatten())
                
                return False
            case ErrorReportType.POSE_IN_CAM:
                self.estimated_poses_in_cam.append(estimated.flatten())
                
                return False
            case _:
                self.estimated_trajectory.append(estimated)
                self.referenced_trajectory.append(expected)
                
                return mean_absolute_error(estimated, expected) > 100.
    
    def export_all_error_report(self, filename: str):
        
        try:
            self._report_general_error(filename)
            self._report_pose_error(filename)
            self._report_pose_error_in_cam_frame()
        except Exception as e:
            logger.error(f"Failed to save error report: {e}")
        
    def compute_error(self, filename: str=None, type: ErrorReportType=ErrorReportType.GENERAL):
        if filename is None:
            filename = datetime.datetime.now().replace(microsecond=0).isoformat()
            
        match (type):
            case ErrorReportType.GENERAL:
                return self._report_general_error(filename)
            case ErrorReportType.POSE:
                return self._report_pose_error(filename)
            case ErrorReportType.POSE_IN_CAM:
                return self._report_pose_error_in_cam_frame()
            case _:
                return self._report_general_error(filename)
            
    def _report_general_error(self, filename: str) -> GeneralErrorReport:
        if len(self.estimated_trajectory) == 0:
            logger.info("General error is not stored. Skipping.")
            return
        
        self.referenced_trajectory = np.array(self.referenced_trajectory)
        self.estimated_trajectory = np.array(self.estimated_trajectory)
        
        estimated_result = self.estimated_trajectory[:, :self.filter_config.dimension]
        ground_truth = self.referenced_trajectory[:, :self.filter_config.dimension]
        
        absolute_errors = np.absolute(np.subtract(ground_truth, estimated_result))
        mae = np.round(np.mean(absolute_errors), self.decimal_place)
        rmse = np.round(np.sqrt(mean_squared_error(ground_truth, estimated_result)), self.decimal_place)
        maximum = np.round(np.max(absolute_errors), self.decimal_place)
        
        if self.config.export_error:
            if not os.path.exists(self.general_error_path):
                os.makedirs(self.general_error_path)
            
            filename = os.path.join(self.general_error_path, filename)
            if not filename.endswith(".csv"):
                filename += ".csv"
                
            error = np.array([mae, rmse, maximum]).reshape(1, -1)
            df = pd.DataFrame(error, columns=["MAE", "RMSE", "MAX"])
            df.to_csv(filename)
            
            logger.info("General error report exported successfully.")
        
        return GeneralErrorReport(mae=mae, rmse=rmse, max=maximum)
    
    def _report_pose_error(self, filename: str) -> PoseErrorReport:
        """ Compute rotational error and translational error of the estimated result for KITTI dataset
        
        """
        # TODO: replace this with error metric provided by https://github.com/Huangying-Zhan/DF-VO?tab=readme-ov-file
        if len(self.estimated_poses) == 0:
            logger.info("Pose error is not stored. Skipping.")
            return
        
        flattened_estimated_pose = np.array([pose.flatten() for pose in self.estimated_poses])
        flattened_referenced_pose = np.array([[pose.flatten() for pose in self.referenced_poses]])
        flattened_estimated_pose = np.squeeze(flattened_estimated_pose)
        flattened_referenced_pose = np.squeeze(flattened_referenced_pose)
        
        if self.config.export_error:
            if not os.path.exists(self.pose_error_path):
                os.makedirs(self.pose_error_path)
            
            filename_estimated = os.path.join(self.pose_error_path, "estimated_" + filename)
            filename_referenced = os.path.join(self.pose_error_path, "referenced_" + filename)
            if not filename_estimated.endswith(".txt"):
                filename_estimated += ".txt"
            if not filename_referenced.endswith(".txt"):
                filename_referenced += ".txt"
            
            np.savetxt(filename_estimated, flattened_estimated_pose, delimiter=" ")
            
            np.savetxt(filename_referenced, flattened_referenced_pose, delimiter=" ")
        
        return PoseErrorReport()
        
    def _report_pose_error_in_cam_frame(self) -> PoseErrorReport:
        if len(self.estimated_poses_in_cam) == 0 or\
            (self.dataset is not DatasetType.KITTI and self.dataset is not DatasetType.EXPERIMENT):
            logger.info("Pose error in camera coordinate is not stored. Skipping.")
            return
        
        flattened_estimated_pose = np.array([pose.flatten() for pose in self.estimated_poses_in_cam])
        flattened_estimated_pose = np.squeeze(flattened_estimated_pose)
        
        if self.config.export_error:
            if not os.path.exists(self.pose_error_in_cam_path):
                os.makedirs(self.pose_error_in_cam_path)
            
            filename = self.seqence+".txt" if self.seqence is not None else "00.txt"
            
            filename_estimated = os.path.join(self.pose_error_in_cam_path, filename)
            np.savetxt(filename_estimated, flattened_estimated_pose, delimiter=" ")
            
        
        return PoseErrorReport()
    
    @staticmethod
    def combine_general_errors(error_list: List[GeneralErrorReport], labels: List[str]) -> pd.DataFrame:
        assert len(error_list) == len(labels), "Please provide same size of error reports and labels."
        
        errors = np.array([np.array([err.mae, err.rmse, err.max]) for err in error_list]).T
        df = pd.DataFrame(errors, index=["MAE", "RMSE", "MAX"], columns=labels)
        return df