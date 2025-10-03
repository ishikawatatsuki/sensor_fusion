import os
import cv2
import logging
import numpy as np
from einops import rearrange
from typing import Tuple
from enum import Enum

from ..common.constants import EUROC_SEQUENCE_MAPS, VO_POSE_ESTIMATION_MAP, KITTI_SEQUENCE_TO_DATE, KITTI_SEQUENCE_TO_DRIVE


class RANSAC_FlagType(Enum):
    SOLVEPNP_ITERATIVE = cv2.SOLVEPNP_ITERATIVE
    SOLVEPNP_EPNP = cv2.SOLVEPNP_EPNP
    SOLVEPNP_P3P = cv2.SOLVEPNP_P3P
    SOLVEPNP_AP3P = cv2.SOLVEPNP_AP3P

    @staticmethod
    def get_names():
        return [e.name for e in RANSAC_FlagType]

    @classmethod
    def from_string(cls, flag_str: str):
        try:
            return cls[flag_str.upper()]
        except KeyError:
            raise ValueError(f"Unknown RANSAC flag: {flag_str}")

class BASE_Detector:
    def __init__(self):
        self.detector = None
        self.descriptor = cv2.xfeatures2d.BriefDescriptorExtractor().create()

    def detectAndCompute(self, image: np.ndarray, mask: np.ndarray = None):
        if self.detector is None:
            raise NotImplementedError("Detector not implemented.")
        keypoints = self.detector.detect(image, mask=mask)
        return self.descriptor.compute(image, keypoints)

class FAST(BASE_Detector):
    def __init__(self):
        super().__init__()
        self.detector = cv2.xfeatures2d.StarDetector().create()

    def detectAndCompute(self, image: np.ndarray, mask: np.ndarray = None):
        keypoints = self.detector.detect(image, mask=mask)
        return self.descriptor.compute(image, keypoints)

class MSER(BASE_Detector):
    def __init__(self):
        super().__init__()
        self.detector = cv2.MSER().create()

    def detectAndCompute(self, image: np.ndarray, mask: np.ndarray = None):
        keypoints = self.detector.detect(image, mask=mask)
        return self.descriptor.compute(image, keypoints)

class FFD(BASE_Detector):
    def __init__(self):
        super().__init__()
        self.detector = cv2.FastFeatureDetector().create()

    def detectAndCompute(self, image: np.ndarray, mask: np.ndarray = None):
        keypoints = self.detector.detect(image, mask=mask)
        return self.descriptor.compute(image, keypoints)

class AFD(BASE_Detector):
    def __init__(self):
        super().__init__()
        self.detector = cv2.AgastFeatureDetector().create()

    def detectAndCompute(self, image: np.ndarray, mask: np.ndarray = None):
        keypoints = self.detector.detect(image, mask=mask)
        return self.descriptor.compute(image, keypoints)
    
class GFTT(BASE_Detector):
    def __init__(self):
        super().__init__()
        self.detector = cv2.GFTTDetector().create()

    def detectAndCompute(self, image: np.ndarray, mask: np.ndarray = None):
        keypoints = self.detector.detect(image, mask=mask)
        return self.descriptor.compute(image, keypoints)
    
class SimpleBlobDetector(BASE_Detector):
    def __init__(self):
        super().__init__()
        self.detector = cv2.SimpleBlobDetector().create()

    def detectAndCompute(self, image: np.ndarray, mask: np.ndarray = None):
        keypoints = self.detector.detect(image, mask=mask)
        return self.descriptor.compute(image, keypoints)

class DetectorType(Enum):
    
    SIFT = "SIFT"
    ORB = "ORB"
    KAZE = "KAZE"
    AKAZE = "AKAZE"
    SURF = "SURF"
    FAST = "FAST"
    BRISK = "BRISK"
    MSER = "MSER"
    FFD = "FFD"
    AFD = "AFD"
    GFTT = "GFTT"
    SIMPLE_BLOB = "SIMPLE_BLOB"

    @staticmethod
    def get_names():
        return [e.name for e in DetectorType]

    @classmethod
    def get_type_from_str(cls, detector_str: str):
        d = detector_str.upper()
        try: 
            return cls(d)
        except:
            return None
    
    @staticmethod
    def create_detector_from_str(detector_str: str):
        d = DetectorType.get_type_from_str(detector_str=detector_str)
        assert d is not None, "Please provide proper detector algorithm."
        
        match (d):
            case DetectorType.SIFT:
                detector = cv2.SIFT()
                detector = detector.create()
            case DetectorType.ORB:
                detector = cv2.ORB()
                detector = detector.create()
            case DetectorType.AKAZE:
                detector = cv2.AKAZE()
                detector = detector.create()
            case DetectorType.SURF:
                detector = cv2.xfeatures2d.SURF()
                detector = detector.create()
            case DetectorType.FAST:
                detector = FAST()
            case DetectorType.BRISK:
                detector = cv2.BRISK()
                detector = detector.create()
            case DetectorType.MSER:
                detector = MSER()
            case DetectorType.FFD:
                detector = FFD()
            case DetectorType.AFD:
                detector = AFD()
            case DetectorType.GFTT:
                detector = GFTT()
            case DetectorType.KAZE:
                detector = cv2.KAZE()
                detector = detector.create()
            case DetectorType.SIMPLE_BLOB:
                detector = SimpleBlobDetector()
            case _:
                logging.warning("No available detector is specified. Setting SIFT.")
                detector = cv2.SIFT()
                detector = detector.create()
        return detector

class MatcherType(Enum):
    
    BF = "BF"
    FLANN = "FLANN"

    @staticmethod
    def get_names():
        return [e.name for e in MatcherType]

    @classmethod
    def get_type_from_str(cls, matcher_str: str):
        d = matcher_str.upper()
        try: 
            return cls(d)
        except:
            return None
    
    @staticmethod
    def create_matcher_from_str(matcher_str: str):
        match (MatcherType.get_type_from_str(matcher_str)):
            case MatcherType.BF:
                matcher = cv2.BFMatcher()
            case MatcherType.FLANN:
                matcher = cv2.FlannBasedMatcher()
            case _:
                logging.warning("No available matcher algorithm is specified. ")
                matcher = cv2.BFMatcher()
            
        return matcher
    
def filter_keypoints_by_border(keypoints, descriptors, img_shape, border=10):
    h, w = img_shape[:2]
    mask = [
        (border < kp.pt[0] < w - border) and 
        (border < kp.pt[1] < h - border)
        for kp in keypoints
    ]
    filtered_kp = [kp for kp, keep in zip(keypoints, mask) if keep]
    filtered_desc = descriptors[np.array(mask)]
    return filtered_kp, filtered_desc

def filter_keypoints_by_response(keypoints, descriptors, threshold=0.01):
    mask = [kp.response > threshold for kp in keypoints]
    filtered_kp = [kp for kp, keep in zip(keypoints, mask) if keep]
    filtered_desc = descriptors[np.array(mask)]
    return filtered_kp, filtered_desc

def filter_keypoints_by_size(keypoints, descriptors, min_size=1, max_size=50, min_keypoints_size=100):
    if len(keypoints) <= min_keypoints_size:
        logging.warning("Not enough keypoints to filter by size. Returning original keypoints.")
        return keypoints, descriptors
    
    mask = [min_size < kp.size < max_size for kp in keypoints]
    filtered_kp = [kp for kp, keep in zip(keypoints, mask) if keep]
    filtered_desc = descriptors[np.array(mask)]
    return filtered_kp, filtered_desc

def filter_keypoints_by_depth(keypoints, descriptors, depth_image, min_depth=0.1, max_depth=50.0, min_keypoints_size=100):
    """
    keypoints: list of cv2.KeyPoint
    descriptors: np.ndarray (N x 128 for SIFT)
    depth_image: np.ndarray (same width/height as input image)
    min_depth, max_depth: valid depth range (in meters or dataset units)
    """
    if len(keypoints) <= min_keypoints_size:
        logging.warning("Not enough keypoints to filter by depth. Returning original keypoints.")
        return keypoints, descriptors
    
    filtered_kp = []
    filtered_desc = []

    for kp, desc in zip(keypoints, descriptors):
        x, y = int(round(kp.pt[0])), int(round(kp.pt[1]))
        if x < 0 or y < 0 or y >= depth_image.shape[0] or x >= depth_image.shape[1]:
            continue
        depth = depth_image[y, x]
        
        # skip invalid or too far/close depths
        if np.isnan(depth) or depth <= 0:
            continue
        if depth < min_depth or depth > max_depth:
            continue
        
        filtered_kp.append(kp)
        filtered_desc.append(desc)

    return filtered_kp, np.array(filtered_desc)

def grid_sample_indices(image_shape: Tuple[int, int], keypoints: list[cv2.KeyPoint], grid_rows=4, grid_cols=4, max_per_cell=10):
    """
    Args:
        image_shape: (H, W) tuple
        keypoints: list of cv2.KeyPoint
        grid_rows: number of rows in the image grid
        grid_cols: number of cols in the image grid
        max_per_cell: max keypoints to select per cell

    Returns:
        selected_indices: list of indices of selected keypoints
    """
    h, w = image_shape
    cell_h, cell_w = h // grid_rows, w // grid_cols
    selected_indices = []

    keypoints_with_idx = [(i, kp) for i, kp in enumerate(keypoints)]

    for i in range(grid_rows):
        for j in range(grid_cols):
            x0, y0 = j * cell_w, i * cell_h
            x1, y1 = (j + 1) * cell_w, (i + 1) * cell_h

            # Get indices of keypoints in the current cell
            cell_kps = [
                (idx, kp) for idx, kp in keypoints_with_idx
                if x0 <= kp.pt[0] < x1 and y0 <= kp.pt[1] < y1
            ]

            # Sort by response and select top max_per_cell
            cell_kps = sorted(cell_kps, key=lambda x: -x[1].response)[:max_per_cell]
            selected_indices.extend([idx for idx, _ in cell_kps])

    return selected_indices

def sample_keypoints_uniformly(keypoints, descriptors, img_shape, grid_rows=4, grid_cols=4, max_per_cell=10, min_keypoints_size=100):
    if len(keypoints) <= min_keypoints_size:
        logging.warning("Not enough keypoints to sample uniformly. Returning original keypoints.")
        return keypoints, descriptors

    uniformly_distributed_samples_index = grid_sample_indices(img_shape, keypoints, grid_rows, grid_cols, max_per_cell)
    keypoints = [keypoints[i] for i in uniformly_distributed_samples_index]
    descriptors = descriptors[uniformly_distributed_samples_index]
    return keypoints, descriptors

def draw_reprojection(image, measured_pts, projected_pts):
    img_vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image.copy()
    
    print(f"Measured points: {measured_pts.shape}, Projected points: {projected_pts.shape}")
    for (x1, y1), (x2, y2) in zip(measured_pts, projected_pts):
        # Measured point (from optical flow or feature match)
        cv2.circle(img_vis, (int(x1), int(y1)), 3, (0, 255, 0), -1)  # green
        
        # Reprojected point (from estimated pose)
        cv2.circle(img_vis, (int(x2), int(y2)), 3, (0, 0, 255), -1)  # red
        
        # Line between them
        cv2.line(img_vis, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1)

    cv2.imshow("Reprojection Debug", img_vis)
    cv2.waitKey(1)

def print_reprojection_error(measured_pts, projected_pts):
    if measured_pts.shape != projected_pts.shape:
        raise ValueError("Measured and projected points must have the same shape.")
    
    errors = np.linalg.norm(measured_pts - projected_pts, axis=1)
    rmse = np.sqrt(np.mean(errors ** 2))
    print(f"Reprojection RMSE: {rmse:.4f} pixels")
    print(f"Max error: {np.max(errors)}")
    print(f"Min error: {np.min(errors)}")

def get_saved_vo_pose_dir(root_path: str, variant: str, dataset_type: str, estimation_type: str) -> bool:

    def _kitti_vo_pose_dir() -> str:
        date = KITTI_SEQUENCE_TO_DATE.get(variant)
        drive = KITTI_SEQUENCE_TO_DRIVE.get(variant)
        estimation_dir = VO_POSE_ESTIMATION_MAP.get(estimation_type, "vo_pose_estimates_2d3d")
        vo_pose_dir = os.path.join(root_path, estimation_dir, date, drive, "data.csv")
        return vo_pose_dir

    def _euroc_vo_pose_dir() -> str:
        sequence = EUROC_SEQUENCE_MAPS.get(variant, "MH_01_easy")
        estimation_dir = VO_POSE_ESTIMATION_MAP.get(estimation_type, "vo_pose_estimates_2d3d")
        vo_pose_dir = os.path.join(root_path, estimation_dir, sequence, "data.csv")
        return vo_pose_dir
    
    if dataset_type == "kitti":
        return _kitti_vo_pose_dir()
    elif dataset_type == "euroc":
        return _euroc_vo_pose_dir()
    else:
        return None

if __name__ == "__main__":
    import cv2
    import time
    import numpy as np
    import pandas as pd
    base_dir = "/Volumes/Data_EXT/data/workspaces/sensor_fusion/data/KITTI/2011_09_30/2011_09_30_drive_0020_sync/image_00/data/0000000006.png"
    image = cv2.imread(base_dir)

    back_list = [DetectorType.SURF.name]
    detectors = [ detector for detector in DetectorType.get_names() if detector not in back_list]
    # print("Available detectors:", detectors)
    # image_save_base_dir = "feature_detection_test"
    # if not os.path.exists(image_save_base_dir):
    #     os.makedirs(image_save_base_dir)

    # statistics = []
    # for detector_name in detectors:
    #     print(f"Testing detector: {detector_name}")
    #     detector = DetectorType.create_detector_from_str(detector_name)
    #     start = time.time()
    #     keypoints, descriptors = detector.detectAndCompute(image, None)
    #     end = time.time()
    #     duration = end - start
    #     statistics.append({
    #         'detector': detector_name,
    #         'keypoints': len(keypoints),
    #         'time': duration
    #     })
        
    #     img_with_keypoints = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0), flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
    #     save_path = os.path.join(image_save_base_dir, f"{detector_name}.png")
    #     cv2.imwrite(save_path, img_with_keypoints)
    #     print(f"{detector_name}: {len(keypoints)} keypoints detected in {duration:.4f} seconds.")

    # df = pd.DataFrame(statistics, columns=['detector', 'keypoints', 'time'])
    # df = df.sort_values(by='keypoints', ascending=False)
    # df.to_csv("detected_keypoints.csv", index=False)
    
    matcgers = [matcher for matcher in MatcherType.get_names()]
    frame1 = cv2.imread("/Volumes/Data_EXT/data/workspaces/sensor_fusion/data/KITTI/2011_09_30/2011_09_30_drive_0020_sync/image_00/data/0000000005.png")
    frame2 = cv2.imread("/Volumes/Data_EXT/data/workspaces/sensor_fusion/data/KITTI/2011_09_30/2011_09_30_drive_0020_sync/image_00/data/0000000006.png")
    success_pairs = []
    for detector_name in detectors:
        for matcher_name in matcgers:
            print(f"Testing detector: {detector_name} with matcher: {matcher_name}")
            detector = DetectorType.create_detector_from_str(detector_name)
            matcher = MatcherType.create_matcher_from_str(matcher_name)

            kp1, desc1 = detector.detectAndCompute(frame1, None)
            kp2, desc2 = detector.detectAndCompute(frame2, None)
            try:
                matches = matcher.knnMatch(desc1, desc2, k=2)
                success_pairs.append((detector_name, matcher_name, len(matches)))
            except Exception as e:
                print(f"Error with matcher {matcher_name}: {e}")
                continue

    print("Successful pairs:")
    for pair in success_pairs:
        print(f"Detector: {pair[0]}, Matcher: {pair[1]}, Matches: {pair[2]}")

    # flag = RANSAC_FlagType.from_string("SOLVEPNP_EPNP")
    # print(f"Selected RANSAC flag: {flag.name} with value {flag.value}")