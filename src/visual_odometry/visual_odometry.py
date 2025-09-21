import os
import sys
import cv2
import yaml
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
from einops import rearrange
from sklearn.metrics import mean_absolute_error

from .depth_estimator import DepthEstimator
from .object_detection import DynamicObjectDetector, ObjectDetectorType
from .vo_utils import (
    DetectorType, 
    MatcherType, 
    RANSAC_FlagType, 
    draw_reprojection, 
    print_reprojection_error,
    grid_sample_indices,
    get_saved_vo_pose_dir,

    sample_keypoints_uniformly,
    filter_keypoints_by_border,
    filter_keypoints_by_response,
    filter_keypoints_by_size,
    filter_keypoints_by_depth
)


from ..common.config import VisualOdometryConfig
from ..internal.extended_common.extended_config import DatasetConfig
from ..common.datatypes import ImageData, VisualOdometryData, SensorType
from ..common.constants import EUROC_SEQUENCE_MAPS, VO_POSE_ESTIMATION_MAP, KITTI_SEQUENCE_TO_DATE, KITTI_SEQUENCE_TO_DRIVE

class EstimatorType(Enum):
    EpipolarGeometryBased = '2d2d'
    PnP = '2d3d'
    Hybrid = 'hybrid'

    @classmethod
    def from_string(cls, value: str):
        if value.lower() == '2d2d':
            return cls.EpipolarGeometryBased
        elif value.lower() == '2d3d':
            return cls.PnP
        elif value.lower() == 'hybrid':
            return cls.Hybrid
        else:
            raise ValueError(f"Unknown estimator type: {value}")
        

@dataclass
class DebuggingData:
    prev_pts: np.ndarray = None
    next_pts: np.ndarray = None
    mask: np.ndarray = None

    def __str__(self):
        return f"DebuggingData(prev_pts={self.prev_pts.shape if self.prev_pts is not None else None}, next_pts={self.next_pts.shape if self.next_pts is not None else None}, mask={self.mask.shape if self.mask is not None else None})"

@dataclass
class PoseEstimatorArgs:
    prev_pts: np.ndarray
    next_pts: np.ndarray
    mask: np.ndarray
    depth_map: np.ndarray
    def __init__(self, prev_pts=None, next_pts=None, mask=None, depth_map=None):
        self.prev_pts = prev_pts
        self.next_pts = next_pts
        self.mask = mask
        self.depth_map = depth_map


class StaticVisualOdometry:
    """Parse expoerted visual odometry estimates and simulate real visual odometry output."""
    def __init__(
            self, 
            config: VisualOdometryConfig, 
            dataset_config: DatasetConfig, 
            debug: bool = False
        ):
        self.config = config
        self.dataset_config = dataset_config
        self.is_debugging = debug
        self.previous_timestamp = None
        self.debugging_data = DebuggingData()

        self.vo_poses = self._get_exported_vo_poses()

    def _get_kitti_vo_poses(self, vo_pose_dir: str) -> dict[float, np.ndarray]:
        import pykitti
        root_path, variant = self.dataset_config.root_path, self.dataset_config.variant
        date = KITTI_SEQUENCE_TO_DATE.get(variant)
        drive = KITTI_SEQUENCE_TO_DRIVE.get(variant)
        kitti_dataset = pykitti.raw(root_path, date, drive)

        vo_estimates = pd.read_csv(
            vo_pose_dir,
            names=[str(i) for i in range(16)]
        ).values.reshape(-1, 4, 4)[:, :3, :]
        self.previous_timestamp = datetime.timestamp(kitti_dataset.timestamps[0])
        return {datetime.timestamp(ts): pose for ts, pose in zip(kitti_dataset.timestamps[1:], vo_estimates)}


    def _get_euroc_vo_poses(self, vo_pose_dir: str):
        
        sequence = EUROC_SEQUENCE_MAPS.get(self.dataset_config.variant, "mav_01")
        vo_estimates = pd.read_csv(
            vo_pose_dir,
            names=[str(i) for i in range(16)]
        ).values.reshape(-1, 4, 4)[:, :3, :]

        timestamps_file = os.path.join(self.dataset_config.root_path, sequence, "cam0", "data.csv")
        columns = "#timestamp [ns],filename".split(",")
        df = pd.read_csv(timestamps_file, names=columns, skiprows=1)
        timestamps = df["#timestamp [ns]"].values
        self.previous_timestamp = timestamps[0]

        return {ts / 1e9: pose for ts, pose in zip(timestamps[1:], vo_estimates)}

    def _get_exported_vo_poses(self):
        vo_pose_dir = get_saved_vo_pose_dir(
            root_path=self.dataset_config.root_path,
            variant=self.dataset_config.variant,
            dataset_type=self.dataset_config.type,
            estimation_type=self.config.type
        )

        if self.dataset_config.type == "kitti":
            return self._get_kitti_vo_poses(vo_pose_dir)
        elif self.dataset_config.type == "euroc":
            return self._get_euroc_vo_poses(vo_pose_dir)
        else:
            raise ValueError("Unsupported dataset type for static visual odometry")

    def compute_pose(self, data: ImageData):
        dt = data.timestamp - self.previous_timestamp
        self.previous_timestamp = data.timestamp
        pose = self.vo_poses.get(data.timestamp, None)
        processed_time = 0.0

        if pose is None:
            return VisualOdometryData(
                success=False, 
                relative_pose=pose, 
                image_timestamp=data.timestamp, 
                estimate_timestamp=data.timestamp+processed_time, 
                dt=dt
            )

        relative_pose = np.eye(4)
        relative_pose[:3, :] = pose
        return VisualOdometryData(
            success=True, 
            relative_pose=relative_pose, 
            image_timestamp=data.timestamp, 
            estimate_timestamp=data.timestamp+processed_time, 
            dt=dt)
    
    def initialize_frame(self, data: ImageData):
        # Convert to grayscale if the image is in color
        self.prev_timestamp = data.timestamp

    def get_debugging_data(self) -> DebuggingData:
        return self.debugging_data

class MonocularVisualOdometry:

    def __init__(
            self, 
            config: VisualOdometryConfig,
            dataset_config: DatasetConfig,
            debug: bool = False
            ):
        
        self.config = config

        self.confidence = config.params.get('confidence', 0.999)
        self.matching_threshold = config.params.get('matching_threshold', 0.3)
        self.ransac_reproj_threshold = config.params.get('ransac_reproj_threshold', 1.0)

        self.reprojection_error = config.params.get('reprojection_error', 4.0)
        self.itterations_count = config.params.get('itterations_count', 100)

        self.keypoint_border_threshold = config.params.get('border_threshold', 30.0)
        self.keypoint_response_threshold = config.params.get('response_threshold', 0.01)
        self.keypoint_size_min = config.params.get('size_min', 2.0)
        self.keypoint_size_max = config.params.get('size_max', 10.0)
        self.keypoint_depth_min = config.params.get('depth_min', 1.0)
        self.keypoint_depth_max = config.params.get('depth_max', 20.0)
        self.min_number_of_keypoints = config.params.get('min_number_of_keypoints', 100)

        self.flag = RANSAC_FlagType.from_string(config.params.get('flag', 'SOLVEPNP_ITERATIVE'))


        self.dataset_config = dataset_config
        self.motion_estimator = EstimatorType.from_string(config.estimator)
        self.object_detector_type = ObjectDetectorType.from_string(config.object_detector)
        model_path = ObjectDetectorType.get_model_path(config.object_detector)
        conf = config.params.get('object_detection_confidence', 0.8)
        self.object_detector = DynamicObjectDetector(
            type=config.object_detector,
            model_path=model_path,
            dynamic_classes=config.dynamic_objects, 
            conf=conf)

    
        self.advanced_detector = config.use_advanced_detector
        if self.advanced_detector:
            self.detector = DetectorType.create_detector_from_str(config.feature_detector)
            self.matcher = MatcherType.create_matcher_from_str(config.feature_matcher)

            logging.debug(f"Using advanced detector: {self.detector} and matcher: {self.matcher}")


        if self.motion_estimator == EstimatorType.PnP or\
            self.motion_estimator == EstimatorType.Hybrid:
            self.depth_estimator = DepthEstimator(config=config)

        self.K, self.distortion_coeffs = self._get_calibration_parameters()

        self.pose_estimator_fn = self._get_pose_estimator()

        self.is_debugging = debug

        self.prev_kp = None
        self.prev_desc = None
        self.prev_frame = None
        self.prev_timestamp = None

        self.debugging_data = DebuggingData()

    def _get_calibration_parameters(self):
        if self.dataset_config.type == "kitti":
            calibration_file = os.path.join(self.dataset_config.root_path, "vo_calibrations", self.dataset_config.variant, f"calib.txt")
            camera_id ="P0:"
            calib_params = pd.read_csv(calibration_file, delimiter=' ', header=None, index_col=0)
            K, _, _, _, _, _, _  = cv2.decomposeProjectionMatrix(np.array(calib_params.loc[camera_id]).reshape((3,4)))
            distortion_coeffs = np.zeros(4)
            return K, distortion_coeffs
        elif self.dataset_config.type == "euroc":
            sequence = EUROC_SEQUENCE_MAPS.get(self.dataset_config.variant, "mav_01")
            calibration_file = os.path.join(self.dataset_config.root_path, sequence, "cam0", "sensor.yaml")
            with open(calibration_file, 'r') as f:
                calib_params = yaml.safe_load(f)
                fu, fv, cu, cv = calib_params['intrinsics']
                k1, k2, p1, p2 = calib_params['distortion_coefficients']
                resolution = calib_params['resolution']
                K = np.array([
                    [fu,  0, cu],
                    [0,  fv, cv],
                    [0,   0,  1]
                ])
                distortion_coeffs = np.array([k1, k2, p1, p2])

            return K, distortion_coeffs
        else:
            raise ValueError("Unsupported dataset type")

    def _get_pose_estimator(self):
        if self.motion_estimator == EstimatorType.EpipolarGeometryBased:
            return self._estimate_2d2d_pose
        elif self.motion_estimator == EstimatorType.PnP:
            return self._estimate_2d3d_pose
        elif self.motion_estimator == EstimatorType.Hybrid:
            return self._estimate_hybrid_pose
        else:
            raise ValueError(f"Unknown estimator type: {self.motion_estimator}")

    def _backproject_to_3d(self, pts2d, depth_map):
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]

        u = pts2d[:, 0]
        v = pts2d[:, 1]
        z = depth_map[v.astype(int), u.astype(int)]

        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        return np.stack((x, y, z), axis=1)  # shape: (N, 3)
    
    def _create_output(self, pose: np.ndarray, timestamp: float):
        dt = timestamp - self.prev_timestamp
        if pose is None:
            return VisualOdometryData(
                success=False, 
                relative_pose=None, 
                image_timestamp=timestamp, 
                estimate_timestamp=timestamp, 
                dt=dt
            )

        # generate random time delay to simulate real world scenario
        processed_time = 0.0

        return VisualOdometryData(
            success=True, 
            relative_pose=pose, 
            image_timestamp=timestamp, 
            estimate_timestamp=timestamp+processed_time, 
            dt=dt
        )

    def _inlier_reprojection(self, pts3d, rvec, tvec):
        reprojected_points, _ = cv2.projectPoints(pts3d, rvec, tvec, self.K, None)
        if reprojected_points.ndim == 3:
            reprojected_points = rearrange(reprojected_points, 'n 1 s -> n s')
        return reprojected_points

    def _estimate_2d2d_pose(self, args: PoseEstimatorArgs) -> np.ndarray:
        prev_pts, next_pts, mask = args.prev_pts, args.next_pts, args.mask

        # This function finds the essential matrix E that satisfies the epipolar constraint
        E, _mask = cv2.findEssentialMat(prev_pts, next_pts, self.K, method=cv2.RANSAC, prob=self.confidence, threshold=self.ransac_reproj_threshold)
        if E is None:
            logging.debug("Essential matrix estimation failed.")
            return None
        next_pts = next_pts[_mask.ravel() == 1]
        prev_pts = prev_pts[_mask.ravel() == 1]

        if next_pts.ndim == 2:
            next_pts = rearrange(next_pts, 'n s -> n 1 s')
        if prev_pts.ndim == 2:
            prev_pts = rearrange(prev_pts, 'n s -> n 1 s')

        if self.is_debugging:
            self.debugging_data = DebuggingData(
                prev_pts=rearrange(prev_pts, 'n 1 s -> n s'),
                next_pts=rearrange(next_pts, 'n 1 s -> n s'),
                mask=mask
            )
        

        _, R, t, _ = cv2.recoverPose(E, prev_pts, next_pts, self.K)

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.flatten()
        return np.linalg.inv(T)

    def _estimate_2d3d_pose(self, args: PoseEstimatorArgs) -> np.ndarray:
        prev_pts, next_pts, mask, depth_map = args.prev_pts, args.next_pts, args.mask, args.depth_map

        if prev_pts.ndim == 3:
            prev_pts = rearrange(prev_pts, 'n 1 s -> n s')
        if next_pts.ndim == 3:
            next_pts = rearrange(next_pts, 'n 1 s -> n s')

        pts3d = self._backproject_to_3d(prev_pts, depth_map)
        valid = ~np.isnan(pts3d[:, 2]) & (pts3d[:, 2] >= self.keypoint_depth_min) & (pts3d[:, 2] < self.keypoint_depth_max)

        if not np.any(valid):
            logging.debug("No valid 3D points found after filtering.")
            return None

        pts3d, next_pts = pts3d[valid], next_pts[valid]
        pts3d = pts3d.astype(np.float32)
        next_pts = next_pts.astype(np.float32)

        # Copy keypoints for debugging
        next_pts_cp = next_pts.copy()
        prev_pts_cp = prev_pts.copy()

        prev_inliers = np.ones((pts3d.shape[0], 1), dtype=np.uint8)
        success, rvec, tvec, inliers = False, None, None, None
        try:
            for _ in range(5):
                success, rvec, tvec, inliers = cv2.solvePnPRansac(
                    pts3d, next_pts, self.K, None,
                    reprojectionError=self.reprojection_error,
                    confidence=self.confidence,
                    iterationsCount=self.itterations_count,
                    flags=self.flag.value,
                )

                logging.debug(f"rvec: {rvec.shape}, tvec: {tvec.shape} sucess: {success}")
                if not success or inliers is None or np.array_equal(inliers, prev_inliers):
                    break

                pts3d = pts3d[inliers.flatten()]
                next_pts = next_pts[inliers.flatten()]

                # Extract the keypoints, which are actually used for pose estimation
                next_pts_cp = next_pts_cp[inliers.flatten()]
                prev_pts_cp = prev_pts_cp[inliers.flatten()]

                prev_inliers = inliers

                logging.debug(f"Inliers: {len(inliers) if inliers is not None else 0}")
                
        except cv2.error as e:
            logging.debug(f"Error in solvePnPRansac: {e}")
            return

        if not success:
            logging.debug("Pose estimation failed due to insufficient points.")
            return
        
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug(f"Estimated rvec: {rvec}, tvec: {tvec}")
            reprojected_pts = self._inlier_reprojection(pts3d, rvec, tvec)
            draw_reprojection(self.prev_frame, next_pts, reprojected_pts)
            print_reprojection_error(next_pts, reprojected_pts)


        if self.is_debugging:
            if prev_pts_cp.ndim == 3:
                prev_pts_cp = rearrange(prev_pts_cp, 'n 1 s -> n s')
            if next_pts_cp.ndim == 3:
                next_pts_cp = rearrange(next_pts_cp, 'n 1 s -> n s')

            self.debugging_data = DebuggingData( 
                prev_pts=prev_pts_cp,
                next_pts=next_pts_cp,
                mask=mask
            )
            
        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = tvec.flatten()
        return np.linalg.inv(T) 
    

    def _estimate_hybrid_pose(self, args: PoseEstimatorArgs) -> np.ndarray:
        
        T_2d2d = self._estimate_2d2d_pose(args)
        if T_2d2d is None:
            return None
        T_2d3d = self._estimate_2d3d_pose(args)
        if T_2d3d is None:
            return None
        
        T = np.eye(4)
        T[:3, :3] = T_2d2d[:3, :3] # Get rotation from 2d2d
        T[:3, 3] = T_2d3d[:3, 3] # Get translation from 2d3d
        return T
    

    def _compute_keypoints(self, frame: np.ndarray) -> PoseEstimatorArgs:

        response = PoseEstimatorArgs()

        mask = self.object_detector.get_dynamic_mask(frame)
        
        if self.motion_estimator != EstimatorType.EpipolarGeometryBased:
            depth_map = self.depth_estimator.estimate_depth(frame)
            # depth_map = cv2.resize(depth_map, (self.prev_frame.shape[1], self.prev_frame.shape[0]), interpolation=cv2.INTER_LINEAR)
            response.depth_map = depth_map

        if self.advanced_detector:
            current_kp, desc = self.detector.detectAndCompute(frame, mask=mask)

            current_kp, desc = filter_keypoints_by_border(current_kp, desc, frame.shape, border=self.keypoint_border_threshold)
            current_kp, desc = filter_keypoints_by_response(current_kp, desc, threshold=self.keypoint_response_threshold)
            current_kp, desc = filter_keypoints_by_size(current_kp, desc, min_size=self.keypoint_size_min, max_size=self.keypoint_size_max, min_keypoints_size=self.min_number_of_keypoints)
            
            if self.motion_estimator != EstimatorType.EpipolarGeometryBased:
                current_kp, desc = filter_keypoints_by_depth(current_kp, desc, depth_map, min_depth=self.keypoint_depth_min, max_depth=self.keypoint_depth_max, min_keypoints_size=self.min_number_of_keypoints)

            # Balance keypoints across the grid in a given frame
            current_kp, desc = sample_keypoints_uniformly(current_kp, desc, frame.shape, grid_rows=30, grid_cols=30, min_keypoints_size=self.min_number_of_keypoints)

            if self.prev_kp is None or self.prev_desc is None:
                self.prev_kp = current_kp
                self.prev_desc = desc
                return response
    
            matches = self.matcher.knnMatch(self.prev_desc, desc, k=2)

            good_points = []
            for m, n in matches:
                if m.distance < self.matching_threshold * n.distance:
                    good_points.append(m)

            if len(good_points) < 5:
                print("Not enough good matches")
                return response

            prev_pts = np.float32([self.prev_kp[m.queryIdx].pt for m in good_points])
            next_pts = np.float32([current_kp[m.trainIdx].pt for m in good_points])

            # prev_pts = cv2.undistortPoints(np.expand_dims(prev_pts, axis=2), self.K, self.distortion_coeffs)
            # next_pts = cv2.undistortPoints(np.expand_dims(next_pts, axis=2), self.K, self.distortion_coeffs)

            self.prev_kp = current_kp
            self.prev_desc = desc
        else:
            if self.prev_frame is None:
                return response
            
            prev_pts = cv2.goodFeaturesToTrack(self.prev_frame, maxCorners=1000, qualityLevel=0.01, minDistance=7, mask=mask)
            if prev_pts is None or len(prev_pts) < 4:
                logging.warning("Not enough points to track. Skipping frame.")
                return None, None, None
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_frame, frame, prev_pts, None)
            prev_pts, next_pts = prev_pts[status.flatten() == 1], next_pts[status.flatten() == 1]

            # prev_pts = cv2.undistortPoints(prev_pts, self.K, self.distortion_coeffs)
            # next_pts = cv2.undistortPoints(next_pts, self.K, self.distortion_coeffs)

        response.prev_pts = prev_pts
        response.next_pts = next_pts
        response.mask = mask
        return response


    def compute_pose(self, data: ImageData):

        # Convert current frame to grayscale if it's in color
        current_frame = data.image
        
        if len(current_frame.shape) == 3:
            current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        args = self._compute_keypoints(current_frame)

        if self.prev_frame is None:
            self.initialize_frame(data)
            return self._create_output(None, data.timestamp)

        logging.debug(f"prev_pts: {args.prev_pts.shape}, next_pts: {args.next_pts.shape}")

        pose = self.pose_estimator_fn(args)

        # Update previous frame for next iteration
        self.prev_frame = current_frame
        self.prev_timestamp = data.timestamp

        return self._create_output(pose, data.timestamp)

    def initialize_frame(self, data: ImageData):
        # Convert to grayscale if the image is in color
        if len(data.image.shape) == 3:
            self.prev_frame = cv2.cvtColor(data.image, cv2.COLOR_BGR2GRAY)
        else:
            self.prev_frame = data.image
        self.prev_timestamp = data.timestamp

    def get_debugging_data(self) -> DebuggingData:
        return self.debugging_data


class VisualOdometry:
    def __init__(self, config: VisualOdometryConfig, dataset_config: DatasetConfig, debug: bool = False):

        if config.type == "static":
            vo_pose_dir = get_saved_vo_pose_dir(
                root_path=dataset_config.root_path,
                variant=dataset_config.variant,
                dataset_type=dataset_config.type,
                estimation_type=config.type
            )
            
            if os.path.exists(vo_pose_dir):
                self._vo_provider = StaticVisualOdometry(config=config, dataset_config=dataset_config, debug=debug)
            else:
                logging.warning("Static visual odometry data not found. Falling back to MonocularVisualOdometry.")
                self._vo_provider = MonocularVisualOdometry(config=config, dataset_config=dataset_config, debug=debug)
        elif config.type == "monocular":
            self._vo_provider = MonocularVisualOdometry(config=config, dataset_config=dataset_config, debug=debug)
        else:
            raise ValueError(f"Unsupported visual odometry type: {config.type}")
    
    def compute_pose(self, data: ImageData) -> VisualOdometryData:
        return self._vo_provider.compute_pose(data=data)
    
    def get_debugging_data(self) -> DebuggingData:
        return self._vo_provider.get_debugging_data()
    
    @property
    def is_debugging(self) -> bool:
        return self._vo_provider.is_debugging
    
    @property
    def get_datatype(self) -> SensorType:
        if self._vo_provider.dataset_config.type == "kitti":
            return SensorType.KITTI_VO
        elif self._vo_provider.dataset_config.type == "euroc":
            return SensorType.EuRoC_VO
        return None

if __name__ == "__main__":
    import time
    import sys
    import os
    import pykitti
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    from src.internal.visualizers.vo_visualizer import VO_Visualizer, VO_VisualizationData
    from src.common.constants import KITTI_SEQUENCE_TO_DATE, KITTI_SEQUENCE_TO_DRIVE, EUROC_SEQUENCE_MAPS


    logging.basicConfig(level=logging.INFO)
    logging.getLogger('matplotlib.font_manager').disabled = True
    use_kitti = True

    if use_kitti:
        rootpath = "/Volumes/Data_EXT/data/workspaces/sensor_fusion/data/KITTI"
        variant = "09"
        date = KITTI_SEQUENCE_TO_DATE.get(variant, "2011_09_30")
        drive = KITTI_SEQUENCE_TO_DRIVE.get(variant, "0033")
        dataset_config = DatasetConfig(
            type='kitti',
            mode='stream',
            root_path=rootpath,
            variant=variant,
        )
        image_path = os.path.join(rootpath, f"{date}/{date}_drive_{drive}_sync/image_00/data")
        image_files = sorted([f for f in os.listdir(image_path) if f.endswith('.png')])
        date = KITTI_SEQUENCE_TO_DATE.get(variant)
        drive = KITTI_SEQUENCE_TO_DRIVE.get(variant)
        kitti_dataset = pykitti.raw(rootpath, date, drive)
        image_timestamps = np.array([datetime.timestamp(ts) for ts in kitti_dataset.timestamps])
        start = 0
        image_files = image_files[start:]
        image_timestamps = image_timestamps[start:]
        logging.debug(f"Found {len(image_files)} image files.")
        ground_truth_path = os.path.join(rootpath, f"ground_truth/{variant}.txt")
        ground_truth = pd.read_csv(ground_truth_path, sep=' ', header=None, skiprows=1).values
        ground_truth = ground_truth.reshape(-1, 3, 4)
        gt_pos = ground_truth[start:, :3, 3]
    else:
        rootpath = "/Volumes/Data_EXT/data/workspaces/sensor_fusion/data/EuRoC"
        variant = "01"
        sequence = EUROC_SEQUENCE_MAPS.get(variant, "mav_01")
        dataset_config = DatasetConfig(
            type='euroc',
            mode='stream',
            root_path=rootpath,
            variant=variant,
        )
        offset_start = 1000
        end = offset_start + 100
        image_path = os.path.join(rootpath, f"{sequence}/cam0/data")
        image_timestamps = pd.read_csv(os.path.join(rootpath, f"{sequence}/cam0/data.csv"), names=["#timestamp [ns]", "filename"], skiprows=1)['#timestamp [ns]'].values[offset_start:end]
        image_files = sorted([f for f in os.listdir(image_path) if f.endswith('.png')])[offset_start:end]
        logging.debug(f"Found {len(image_files)} image files.")

        ground_truth_path = os.path.join(rootpath, f"{sequence}/state_groundtruth_estimate0/data.csv")
        columns = "#timestamp, p_RS_R_x [m], p_RS_R_y [m], p_RS_R_z [m], q_RS_w [], q_RS_x [], q_RS_y [], q_RS_z [], v_RS_R_x [m s^-1], v_RS_R_y [m s^-1], v_RS_R_z [m s^-1], b_w_RS_S_x [rad s^-1], b_w_RS_S_y [rad s^-1], b_w_RS_S_z [rad s^-1], b_a_RS_S_x [m s^-2], b_a_RS_S_y [m s^-2], b_a_RS_S_z [m s^-2]".split(", ")
        df = pd.read_csv(ground_truth_path, names=columns, skiprows=1)
        initial_timestamp = image_timestamps[0]
        gt_pos = df[df["#timestamp [ns]"] > initial_timestamp][["p_RS_R_x [m]", "p_RS_R_y [m]", "p_RS_R_z [m]"]].values

    config_filepath = "/Volumes/Data_EXT/data/workspaces/sensor_fusion/configs/kitti_config.yaml"
    with open(config_filepath, 'r') as f:
        config = yaml.safe_load(f)
        vo_json_config = config.get("visual_odometry", None)
    
    if vo_json_config is None:
        logging.error("Visual Odometry configuration not found in the config file.")
        exit(1)
    
    config = VisualOdometryConfig.from_json(vo_json_config)
    config.type = "monocular"
    config.estimator = "2d3d"

    # config = VisualOdometryConfig(
    #     type='monocular',
    #     estimator='hybrid',
    #     camera_id='left',
    #     depth_estimator='dino_v2',
    #     object_detector='segformer',
    #     use_advanced_detector=True,
    #     feature_detector='SIFT',
    #     feature_matcher='BF',
    #     export_vo_data=False,
    #     export_vo_data_path='./data/EuRoC',
    #     params={
    #         'confidence': 0.90,
    #         'ransac_reproj_threshold': 0.999,
    #         'matching_threshold': 0.45,
    #         'max_features': 1000,
    #         'object_detection_confidence': 0.7,
    #         'border_threshold': 30.0,
    #         'response_threshold': 0.01,
    #         'size_min': 3.0,
    #         'size_max': 15.0,
    #         'depth_min': 1.0,
    #         'depth_max': 35.0,
    #         'min_number_of_keypoints': 100,
    #     },
    #     dynamic_objects=['sky'],
    # )

    vo = VisualOdometry(config=config, dataset_config=dataset_config, debug=True)
    logging.debug("Visual Odometry initialized.")

    logging.debug(gt_pos.shape)
    logging.debug(f"Loaded ground truth data with {len(gt_pos)} entries.")

    # frame1 = cv2.imread(os.path.join(image_path, image_files[0]))
    # frame2 = cv2.imread(os.path.join(image_path, image_files[1]))

    # vo.compute_pose(ImageData(image=frame1, timestamp=time.time()))
    # time.sleep(0.1)  # Simulate a delay between frames
    # vo.compute_pose(ImageData(image=frame2, timestamp=time.time()))

    ground_truth_position = []
    estimated_position = []
    estimated_pose = []
    current_pose = np.eye(4)
    if use_kitti:
        current_pose[:3, 3] = np.array([gt_pos[0, 0], gt_pos[0, 1], gt_pos[0, 2]])
    else:
        current_pose[:3, 3] = np.array([-gt_pos[0, 1], -gt_pos[0, 2], gt_pos[0, 0]])
    max_points = 200
    if vo.is_debugging:
        vis = VO_Visualizer(
            save_path=f"/Volumes/Data_EXT/data/workspaces/sensor_fusion/outputs/vo_estimates/vo_debug/2d2d_{variant}",
            save_frame=True
        )
        vis.start()

    for i, (image_file, timestamp) in enumerate(tqdm(zip(image_files, image_timestamps), total=len(image_files))):
        idx = i + 1
        frame_path = os.path.join(image_path, image_file)
        frame = cv2.imread(frame_path)
        vo_data = vo.compute_pose(ImageData(image=frame, timestamp=timestamp))
        if vo_data.success:
            if idx >= len(gt_pos):
                continue

            pose = vo_data.relative_pose
            current_pose = current_pose @ pose
            estimated_pose.append(current_pose[:3, :].flatten())
            estimated_position.append(current_pose[:3, 3])
            ground_truth_position.append(gt_pos[idx])
            debugging_data = vo.get_debugging_data()
            if debugging_data.prev_pts is not None:
                x, y, z = current_pose[:3, 3]
                _est_pose = np.array([x, z])
                px, py, pz = gt_pos[idx]
                if use_kitti:
                    _gt_pose = np.array([px, pz])
                else:
                    _gt_pose = np.array([-py, px])
                vis_data = VO_VisualizationData(
                    frame=frame, 
                    mask=debugging_data.mask,
                    pts_prev=debugging_data.prev_pts[:max_points], 
                    pts_curr=debugging_data.next_pts[:max_points], 
                    estimated_pose=_est_pose, 
                    gt_pose=_gt_pose
                )
                vis.send(vis_data)
                time.sleep(0.05)

    estimated_pose = np.array(estimated_pose)
    
    ground_truth_position = np.array(ground_truth_position)
    estimated_position = np.array(estimated_position)
    min_len = min(len(ground_truth_position), len(estimated_position))
    ground_truth_position = ground_truth_position[:min_len]
    estimated_position = estimated_position[:min_len]


    # logging.info(f"Estimated Positions shape: {ground_truth_position.shape}")
    # logging.info(f"Estimated Pose shape: {estimated_position.shape}")

    # mae = mean_absolute_error(ground_truth_position, estimated_position)
    # logging.info(f"Mean Absolute Error (MAE) of the estimated positions: {mae:.4f}m")

    vis.stop()

    # # Plot the estimated trajectory
    plt.figure(figsize=(10, 8))
    px, py, pz = gt_pos.T
    plt.plot(px, py, marker='o', markersize=1, label='Ground Truth Trajectory', color='black')
    px, py, pz = estimated_position.T
    plt.plot(pz, -px, marker='o', markersize=1, label='Estimated Trajectory', color='blue')
    plt.title('Estimated Trajectory from Visual Odometry')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.axis('equal')
    plt.grid()
    plt.legend()
    # plt.show()
    plt.savefig('estimated_trajectory.png')
