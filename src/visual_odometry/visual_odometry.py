import os
import sys
import cv2
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
from einops import rearrange
from sklearn.metrics import mean_absolute_error

from .depth_estimator import DepthEstimator
from .object_detection import DynamicObjectDetector
from .vo_utils import DetectorType, MatcherType, RANSAC_FlagType, draw_reprojection, print_reprojection_error

from ..common.config import VisualOdometryConfig
from ..internal.extended_common.extended_config import DatasetConfig
from ..common.datatypes import ImageData


class EstimatorType(Enum):
    EpipolarGeometryBased = '2d2d'
    PnP = '2d3d'

    @classmethod
    def from_string(cls, value: str):
        if value.lower() == '2d2d':
            return cls.EpipolarGeometryBased
        elif value.lower() == '2d3d':
            return cls.PnP
        else:
            raise ValueError(f"Unknown estimator type: {value}")
        

@dataclass
class DebuggingData:
    prev_pts: np.ndarray = None
    next_pts: np.ndarray = None
    mask: np.ndarray = None

    def __str__(self):
        return f"DebuggingData(prev_pts={self.prev_pts.shape if self.prev_pts is not None else None}, next_pts={self.next_pts.shape if self.next_pts is not None else None}, mask={self.mask.shape if self.mask is not None else None})"

"""
    [x] Make 2d3d monocular visual odometry work with depth estimation
    [] Measure optical flow on a segment detected by object detector and when the vector is aligning with the camera motion, remove the mask on that segment since it is static
"""

class VisualOdometry:

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

        self.min_depth_threshold = config.params.get('min_depth_threshold', 1)
        self.max_depth_threshold = config.params.get('max_depth_threshold', 50)
        self.reprojection_error = config.params.get('reprojection_error', 4.0)
        self.itterations_count = config.params.get('itterations_count', 100)
        self.flag = RANSAC_FlagType.from_string(config.params.get('flag', 'SOLVEPNP_ITERATIVE'))


        self.dataset_config = dataset_config
        self.motion_estimator = EstimatorType.from_string(config.estimator)
        self.object_detector = DynamicObjectDetector(model_path="src/visual_odometry/yolo11n-seg.pt", dynamic_classes=["person", "car", "bicycle", "motorbike", "bus", "truck"], conf=0.6)

    
        self.advanced_detector = config.use_advanced_detector
        if self.advanced_detector:
            self.detector = DetectorType.create_detector_from_str(config.feature_detector)
            self.matcher = MatcherType.create_matcher_from_str(config.feature_matcher)

            logging.debug(f"Using advanced detector: {self.detector} and matcher: {self.matcher}")


        if self.motion_estimator == EstimatorType.PnP:
            self.depth_estimator = DepthEstimator(config=config)

        self.K = self._get_intrinsic_matrix()

        self.pose_estimator_fn = self._get_pose_estimator()

        self.is_debugging = debug

        self.prev_kp = None
        self.prev_desc = None
        self.prev_frame = None
        self.prev_timestamp = None

        self.debugging_data = DebuggingData()

    def _get_calib_path(self):
        if self.dataset_config.type == "kitti":
            return os.path.join(self.dataset_config.root_path, "vo_calibrations", self.dataset_config.variant, f"calib.txt"), "P0:"
        else:
            raise ValueError("Unsupported dataset type")

    def _get_intrinsic_matrix(self):
        calibration_file, camera_id = self._get_calib_path()
        calib_params = pd.read_csv(calibration_file, delimiter=' ', header=None, index_col=0)
        K, _, _, _, _, _, _  = cv2.decomposeProjectionMatrix(np.array(calib_params.loc[camera_id]).reshape((3,4)))
        return K

    def _get_pose_estimator(self):
        if self.motion_estimator == EstimatorType.EpipolarGeometryBased:
            return self._estimate_2d2d_pose
        elif self.motion_estimator == EstimatorType.PnP:
            return self._estimate_2d3d_pose
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
        return pose
    
    def _inlier_reprojection(self, pts3d, rvec, tvec):
        reprojected_points, _ = cv2.projectPoints(pts3d, rvec, tvec, self.K, None)
        if reprojected_points.ndim == 3:
            reprojected_points = rearrange(reprojected_points, 'n 1 s -> n s')
        return reprojected_points

    def _estimate_2d2d_pose(
            self, 
            prev_pts: np.ndarray, 
            next_pts: np.ndarray,
            mask: np.ndarray
        ) -> np.ndarray:

        E, _mask = cv2.findEssentialMat(next_pts, prev_pts, self.K, method=cv2.RANSAC, prob=self.confidence, threshold=self.ransac_reproj_threshold)
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
        

        _, R, t, _ = cv2.recoverPose(E, next_pts, prev_pts, self.K)

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.flatten()
        return T


    def _estimate_2d3d_pose(
            self, 
            prev_pts: np.ndarray, 
            next_pts: np.ndarray, 
            mask: np.ndarray
        ) -> np.ndarray:
        if prev_pts.ndim == 3:
            prev_pts = rearrange(prev_pts, 'n 1 s -> n s')
        if next_pts.ndim == 3:
            next_pts = rearrange(next_pts, 'n 1 s -> n s')

        depth_map = self.depth_estimator.estimate_depth(self.prev_frame)
        # depth_map = cv2.resize(depth_map, (self.prev_frame.shape[1], self.prev_frame.shape[0]), interpolation=cv2.INTER_LINEAR)
        pts3d = self._backproject_to_3d(prev_pts, depth_map)
        valid = ~np.isnan(pts3d[:, 2]) & (pts3d[:, 2] > self.min_depth_threshold) & (pts3d[:, 2] < self.max_depth_threshold)

        if not np.any(valid):
            logging.debug("No valid 3D points found after filtering.")
            return None

        pts3d, next_pts = pts3d[valid], next_pts[valid]
        pts3d = pts3d.astype(np.float32)
        next_pts = next_pts.astype(np.float32)

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
            _prev_pts = prev_pts.copy()
            _next_pts = next_pts.copy()
            if _prev_pts.ndim == 3:
                _prev_pts = rearrange(_prev_pts, 'n 1 s -> n s')
            if _next_pts.ndim == 3:
                _next_pts = rearrange(_next_pts, 'n 1 s -> n s')

            self.debugging_data = DebuggingData( 
                prev_pts=_prev_pts,
                next_pts=_next_pts,
                mask=mask
            )
            
        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = tvec.flatten()
        return np.linalg.inv(T) 
    
    def _compute_keypoints(self, frame: np.ndarray) -> tuple:

        mask = self.object_detector.get_dynamic_mask(frame)

        if self.advanced_detector:
            current_kp, desc = self.detector.detectAndCompute(frame, mask=mask)

            if self.prev_kp is None or self.prev_desc is None:
                self.prev_kp = current_kp
                self.prev_desc = desc
                return None, None, None

            matches = self.matcher.knnMatch(self.prev_desc, desc, k=2)

            good_points = []
            for m, n in matches:
                if m.distance < self.matching_threshold * n.distance:
                    good_points.append(m)

            if len(good_points) < 5:
                print("Not enough good matches")
                return None, None, None

            prev_pts = np.float32([self.prev_kp[m.queryIdx].pt for m in good_points])
            next_pts = np.float32([current_kp[m.trainIdx].pt for m in good_points])

            self.prev_kp = current_kp
            self.prev_desc = desc
        else:
            if self.prev_frame is None:
                return None, None, None
            
            prev_pts = cv2.goodFeaturesToTrack(self.prev_frame, maxCorners=1000, qualityLevel=0.01, minDistance=7, mask=mask)
            if prev_pts is None or len(prev_pts) < 4:
                logging.warning("Not enough points to track. Skipping frame.")
                return None, None, None
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_frame, frame, prev_pts, None)
            prev_pts, next_pts = prev_pts[status.flatten() == 1], next_pts[status.flatten() == 1]
        
        return prev_pts, next_pts, mask


    def compute_pose(self, data: ImageData):

        # Convert current frame to grayscale if it's in color
        current_frame = data.image
        
        if len(current_frame.shape) == 3:
            current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        prev_pts, next_pts, mask = self._compute_keypoints(current_frame)

        if self.prev_frame is None:
            self.initialize_frame(data)
            return None


        logging.debug(f"prev_pts: {prev_pts.shape}, next_pts: {next_pts.shape}")

        pose = self.pose_estimator_fn(prev_pts, next_pts, mask)

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

if __name__ == "__main__":
    import time
    import sys
    import os
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    from src.internal.visualizers.vo_visualizer import VO_Visualizer, VO_VisualizationData
    from src.common.constants import KITTI_SEQUENCE_TO_DATE, KITTI_SEQUENCE_TO_DRIVE


    logging.basicConfig(level=logging.INFO)
    logging.getLogger('matplotlib.font_manager').disabled = True

    # rootpath = "/gpfs/mariana/home/taishi/workspace/researches/sensor_fusion/data/KITTI"
    rootpath = "/Volumes/Data_EXT/data/workspaces/sensor_fusion/data/KITTI"
    variant = "07"
    date = KITTI_SEQUENCE_TO_DATE.get(variant, "2011_09_30")
    drive = KITTI_SEQUENCE_TO_DRIVE.get(variant, "0033")

    dataset_config = DatasetConfig(
        type='kitti',
        mode='stream',
        root_path=rootpath,
        variant=variant,
    )
    
    config = VisualOdometryConfig(
        type='monocular',
        estimator='2d3d',
        camera_id='left',
        depth_estimator='zoe_depth',
        use_advanced_detector=True,
        feature_detector='SIFT',
        feature_matcher='BF',
        params={
            'confidence': 0.90,
            'ransac_reproj_threshold': 0.99,
            'matching_threshold': 0.5,
            'max_features': 1000,
        }
    )

    vo = VisualOdometry(config=config, dataset_config=dataset_config, debug=True)
    logging.debug("Visual Odometry initialized.")

    image_path = os.path.join(rootpath, f"{date}/{date}_drive_{drive}_sync/image_00/data")
    image_files = sorted([f for f in os.listdir(image_path) if f.endswith('.png')])

    logging.debug(f"Found {len(image_files)} image files.")

    ground_truth_path = os.path.join(rootpath, f"ground_truth/{variant}.txt")
    ground_truth = pd.read_csv(ground_truth_path, sep=' ', header=None, skiprows=1).values
    ground_truth = ground_truth.reshape(-1, 3, 4)
    gt_pos = ground_truth[:, :3, 3]

    logging.debug(gt_pos.shape)
    logging.debug(f"Loaded ground truth data with {len(ground_truth)} entries.")

    # frame1 = cv2.imread(os.path.join(image_path, image_files[0]))
    # frame2 = cv2.imread(os.path.join(image_path, image_files[1]))

    # vo.compute_pose(ImageData(image=frame1, timestamp=time.time()))
    # time.sleep(0.1)  # Simulate a delay between frames
    # vo.compute_pose(ImageData(image=frame2, timestamp=time.time()))

    ground_truth_position = []
    estimated_position = []
    estimated_pose = []
    current_pose = np.eye(4)
    current_pose[:3, 3] = gt_pos[0]

    max_points = 200
    if vo.is_debugging:
        vis = VO_Visualizer(
            save_path=f"/Volumes/Data_EXT/data/workspaces/sensor_fusion/outputs/vo_estimates/vo_debug/2d3d_{variant}",
            save_frame=True
        )
        vis.start()

    for i, image_file in enumerate(tqdm(image_files)):
        idx = (i + 1) % len(gt_pos)
        frame_path = os.path.join(image_path, image_file)
        frame = cv2.imread(frame_path)
        pose = vo.compute_pose(ImageData(image=frame, timestamp=time.time()))
        if pose is not None:
            current_pose = current_pose @ pose
            estimated_pose.append(current_pose[:3, :].flatten())
            estimated_position.append(current_pose[:3, 3])
            ground_truth_position.append(gt_pos[idx])

            debugging_data = vo.get_debugging_data()
            if debugging_data.prev_pts is not None:
                x, y, z = current_pose[:3, 3]
                _est_pose = np.array([x, z])
                px, py, pz = gt_pos[idx, 0], gt_pos[idx, 1], gt_pos[idx, 2]
                _gt_pose = np.array([px, pz])
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

    logging.info(f"Estimated Positions shape: {ground_truth_position.shape}")
    logging.info(f"Estimated Pose shape: {estimated_position.shape}")

    mae = mean_absolute_error(ground_truth_position, estimated_position)
    logging.info(f"Mean Absolute Error (MAE) of the estimated positions: {mae:.4f}m")

    vis.stop()

    # # Plot the estimated trajectory
    # plt.figure(figsize=(10, 8))
    # px, py, pz = ground_truth_position.T
    # plt.plot(px, pz, marker='o', markersize=1, label='Ground Truth Trajectory', color='black')
    # px, py, pz = estimated_position.T
    # plt.plot(px, pz, marker='o', markersize=1, label='Estimated Trajectory', color='blue')
    # plt.title('Estimated Trajectory from Visual Odometry')
    # plt.xlabel('X Position (m)')
    # plt.ylabel('Y Position (m)')
    # plt.axis('equal')
    # plt.grid()
    # plt.legend()
    # # plt.show()
    # plt.savefig('estimated_trajectory.png')
