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
from ..common.constants import EUROC_SEQUENCE_MAPS, VO_POSE_ESTIMATION_MAP, KITTI_SEQUENCE_TO_DATE, KITTI_SEQUENCE_TO_DRIVE, UAV_SEQUENCE_MAPS

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
        
        sequence = EUROC_SEQUENCE_MAPS.get(self.dataset_config.variant, "MH_01_easy")
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

        self.reprojection_error = config.params.get('reprojection_error', 5.0)
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
            sequence = EUROC_SEQUENCE_MAPS.get(self.dataset_config.variant, "MH_01_easy")
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
        elif self.dataset_config.type == "uav":
            sequence = UAV_SEQUENCE_MAPS.get(self.dataset_config.variant, "log0001")
            calibration_file = os.path.join(self.dataset_config.root_path, sequence, "data/modalai/opencv_tracking_intrinsics.yml")
            
            fs = cv2.FileStorage(calibration_file, cv2.FILE_STORAGE_READ)
            K = fs.getNode("M").mat()  # Intrinsic camera matrix
            D = fs.getNode("D").mat()  # Distortion coefficients
            fs.release()
            return K, D
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
        """Backproject 2D points to 3D using depth map.
        
        For distorted cameras (EuRoC), undistort points first to get correct 3D coordinates.
        For rectified cameras (KITTI), use pinhole model directly.
        """
        # Get depth values at pixel locations
        u = pts2d[:, 0]
        v = pts2d[:, 1]
        z = depth_map[v.astype(int), u.astype(int)]
        
        # For cameras with distortion, undistort to normalized coordinates
        if np.any(self.distortion_coeffs != 0):
            # Undistort to normalized coordinates
            pts2d_reshaped = np.expand_dims(pts2d, axis=1).astype(np.float32)
            pts_normalized = cv2.undistortPoints(pts2d_reshaped, self.K, self.distortion_coeffs)
            pts_normalized = pts_normalized.reshape(-1, 2)
            
            # Scale by depth to get 3D points
            x = pts_normalized[:, 0] * z
            y = pts_normalized[:, 1] * z
        else:
            # No distortion - use pinhole model directly
            fx, fy = self.K[0, 0], self.K[1, 1]
            cx, cy = self.K[0, 2], self.K[1, 2]
            
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
        # Project 3D points back to 2D with distortion for accurate reprojection error
        reprojected_points, _ = cv2.projectPoints(pts3d, rvec, tvec, self.K, self.distortion_coeffs)
        if reprojected_points.ndim == 3:
            reprojected_points = rearrange(reprojected_points, 'n 1 s -> n s')
        return reprojected_points

    def _estimate_2d2d_pose(self, args: PoseEstimatorArgs) -> np.ndarray:
        prev_pts, next_pts, mask = args.prev_pts, args.next_pts, args.mask

        # For EuRoC and datasets with lens distortion, we need to undistort points manually
        # since findEssentialMat doesn't accept distCoeffs parameter in OpenCV 4.x
        if np.any(self.distortion_coeffs != 0):
            # Undistort to normalized coordinates, then use with Identity matrix
            prev_pts_norm = cv2.undistortPoints(
                np.expand_dims(prev_pts, axis=1), 
                self.K, 
                self.distortion_coeffs
            )
            next_pts_norm = cv2.undistortPoints(
                np.expand_dims(next_pts, axis=1), 
                self.K, 
                self.distortion_coeffs
            )
            
            # Find essential matrix in normalized space (use Identity as K)
            E, _mask = cv2.findEssentialMat(
                prev_pts_norm, next_pts_norm, np.eye(3), 
                method=cv2.RANSAC, 
                prob=self.confidence, 
                threshold=self.ransac_reproj_threshold / self.K[0, 0]  # Scale threshold to normalized space
            )
            
            if E is None:
                logging.debug("Essential matrix estimation failed.")
                return None
                
            # Filter points based on inliers
            next_pts_norm = next_pts_norm[_mask.ravel() == 1]
            prev_pts_norm = prev_pts_norm[_mask.ravel() == 1]
            
            # Recover pose using normalized coordinates with Identity matrix
            _, R, t, _ = cv2.recoverPose(E, prev_pts_norm, next_pts_norm, np.eye(3))
        else:
            # No distortion (KITTI case) - use original implementation
            E, _mask = cv2.findEssentialMat(
                prev_pts, next_pts, self.K, 
                method=cv2.RANSAC, 
                prob=self.confidence, 
                threshold=self.ransac_reproj_threshold
            )
            
            if E is None:
                logging.debug("Essential matrix estimation failed.")
                return None
                
            next_pts = next_pts[_mask.ravel() == 1]
            prev_pts = prev_pts[_mask.ravel() == 1]
            
            if next_pts.ndim == 2:
                next_pts = rearrange(next_pts, 'n s -> n 1 s')
            if prev_pts.ndim == 2:
                prev_pts = rearrange(prev_pts, 'n s -> n 1 s')
            
            # Recover pose using pixel coordinates with camera matrix
            _, R, t, _ = cv2.recoverPose(E, prev_pts, next_pts, self.K)
            
            if self.is_debugging:
                self.debugging_data = DebuggingData(
                    prev_pts=rearrange(prev_pts, 'n 1 s -> n s') if prev_pts.ndim == 3 else prev_pts,
                    next_pts=rearrange(next_pts, 'n 1 s -> n s') if next_pts.ndim == 3 else next_pts,
                    mask=mask
                )

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

        # Filter ALL arrays by valid mask to maintain correspondence
        pts3d, next_pts, prev_pts = pts3d[valid], next_pts[valid], prev_pts[valid]
        pts3d = pts3d.astype(np.float32)
        next_pts = next_pts.astype(np.float32)
        prev_pts = prev_pts.astype(np.float32)

        # Copy keypoints for debugging - now they have matching lengths
        next_pts_cp = next_pts.copy()
        prev_pts_cp = prev_pts.copy()

        prev_inliers = np.ones((pts3d.shape[0], 1), dtype=np.uint8)
        success, rvec, tvec, inliers = False, None, None, None
        try:
            for _ in range(5):
                # Pass distortion coefficients for accurate reprojection during RANSAC
                # Critical for EuRoC and other datasets with lens distortion
                success, rvec, tvec, inliers = cv2.solvePnPRansac(
                    pts3d, next_pts, self.K, self.distortion_coeffs,
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
            return None

        if not success:
            logging.debug("Pose estimation failed due to insufficient points.")
            return None
        
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
            
            # Check if current descriptors are valid
            if desc is None or len(desc) == 0:
                logging.warning("No descriptors found in current frame. Skipping frame.")
                response.prev_pts = None
                response.next_pts = None
                return response
            
            # Ensure descriptor types match (required for knnMatch)
            if self.prev_desc.dtype != desc.dtype:
                desc = desc.astype(self.prev_desc.dtype)
    
            matches = self.matcher.knnMatch(self.prev_desc, desc, k=2)

            good_points = []
            for match in matches:
                # knnMatch may return fewer than k matches for some descriptors
                if len(match) == 2:
                    m, n = match
                    if m.distance < self.matching_threshold * n.distance:
                        good_points.append(m)

            if len(good_points) < 5:
                logging.warning(f"Not enough good matches: {len(good_points)} matches found")
                response.prev_pts = None
                response.next_pts = None
                return response

            prev_pts = np.float32([self.prev_kp[m.queryIdx].pt for m in good_points])
            next_pts = np.float32([current_kp[m.trainIdx].pt for m in good_points])

            self.prev_kp = current_kp
            self.prev_desc = desc
        else:
            if self.prev_frame is None:
                return response
            
            prev_pts = cv2.goodFeaturesToTrack(self.prev_frame, maxCorners=1000, qualityLevel=0.01, minDistance=7, mask=mask)
            if prev_pts is None or len(prev_pts) < 4:
                logging.warning("Not enough points to track. Skipping frame.")
                response.prev_pts = None
                response.next_pts = None
                return response
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_frame, frame, prev_pts, None)
            prev_pts, next_pts = prev_pts[status.flatten() == 1], next_pts[status.flatten() == 1]
            
            # Check if optical flow tracking succeeded
            if len(prev_pts) < 4 or len(next_pts) < 4:
                logging.warning(f"Optical flow tracking failed: only {len(prev_pts)} points tracked")
                response.prev_pts = None
                response.next_pts = None
                return response

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

        # Check if keypoint computation failed or insufficient points
        if args.prev_pts is None or args.next_pts is None:
            logging.warning("Failed to compute keypoints or insufficient matches. Skipping frame.")
            self.prev_timestamp = data.timestamp
            return self._create_output(None, data.timestamp)
        
        if len(args.prev_pts) < 5 or len(args.next_pts) < 5:
            logging.warning(f"Insufficient keypoints ({len(args.prev_pts)} points). Skipping frame.")
            self.prev_timestamp = data.timestamp
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
        elif self._vo_provider.dataset_config.type == "uav":
            return SensorType.UAV_VO
        return None
