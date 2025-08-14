import os
import sys
import optuna
import cv2
import pandas as pd
from tqdm import tqdm
import numpy as np
import argparse
import matplotlib.pyplot as plt

from src.visual_odometry.visual_odometry import VisualOdometry
from src.visual_odometry.vo_utils import DetectorType, MatcherType
from src.common.config import VisualOdometryConfig
from src.internal.extended_common.extended_config import DatasetConfig
from src.common.datatypes import ImageData

from time import time
from sklearn.metrics import mean_absolute_error

def parse_args():

    parser = argparse.ArgumentParser(description="Visual Odometry Experiment")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save output results")
    parser.add_argument("--log_dir", type=str, default="./logs", help="Directory to save logs")
    parser.add_argument("--num_trials", type=int, default=200, help="Number of trials for hyperparameter optimization")
    
    return parser.parse_args()

detector_matcher_pairs = [
    ("SIFT", "FLANN"),
    ("SIFT", "BF"),
    ("ORB", "BF"),
    ("KAZE", "BF"),
    ("KAZE", "FLANN"),
    ("AKAZE", "BF"),
    ("FAST", "BF"),
    ("BRISK", "BF"),
    ("MSER", "BF"),
    ("FFD", "BF"),
    ("AFD", "BF"),
    ("GFTT", "BF"),
    ("SIMPLE_BLOB", "BF")
]

class VisualOdometryObjective2d2D:
    def __init__(
        self, 
        dataset_dir: str, 
        output_dir: str,
    ):  
        self.dataset_dir = dataset_dir
        self.output_dir = output_dir
        self.image_output_dir = os.path.join(output_dir, "images")
        if not os.path.exists(self.image_output_dir):
            os.makedirs(self.image_output_dir, exist_ok=True)
            
        back_list = [DetectorType.SURF.name]
        self.detector_list = [desc for desc in DetectorType.get_names() if desc not in back_list]
        self.matcher_list = MatcherType.get_names()
        print(f"Available detectors: {self.detector_list}")
        print(f"Available matchers: {self.matcher_list}")


        ground_truth_path = os.path.join(self.dataset_dir, "ground_truth/09.txt")
        ground_truth = pd.read_csv(ground_truth_path, sep=' ', header=None, skiprows=1).values
        self.ground_truth = ground_truth.reshape(-1, 3, 4)[:, :3, 3]

    def __call__(self, trial: optuna.Trial):
        # Hyperparameters

        (feature_detector, feature_matcher) = trial.suggest_categorical("detector_matcher", detector_matcher_pairs)
        essential_matrix_prob = trial.suggest_float("essential_matrix_prob", 0.8, 0.9999, step=0.01)
        essential_matrix_threshold = trial.suggest_float("essential_matrix_threshold", 1.0, 3.0, step=0.5)
        matching_threshold = trial.suggest_float("matching_threshold", 0.1, 0.5, step=0.05)

        dataset_config = DatasetConfig(
            type='kitti',
            mode='stream',
            root_path=self.dataset_dir,
            variant='0033',
        )
                
        config = VisualOdometryConfig(
            type='monocular',
            estimator='2d2d',
            camera_id='left',
            depth_estimator='zoe_depth',
            use_advanced_detector=True,
            feature_detector=feature_detector,
            feature_matcher=feature_matcher,
            params={
                'max_features': 1000,
                'ransac_reproj_threshold': essential_matrix_threshold,
                'confidence': essential_matrix_prob,
                'matching_threshold': matching_threshold
            }
        )

        vo = VisualOdometry(config=config, dataset_config=dataset_config, debug=False)

        image_path = os.path.join(self.dataset_dir, "2011_09_30/2011_09_30_drive_0033_sync/image_00/data")
        image_files = sorted([f for f in os.listdir(image_path) if f.endswith('.png')])
        
        current_pose = np.eye(4)
        current_pose[:3, 3] = self.ground_truth[0]
        
        estimated_position = []
        ground_truth_position = []

        try:
            for i, image_file in enumerate(tqdm(image_files)):
                idx = (i+1) % len(self.ground_truth)
                frame_path = os.path.join(image_path, image_file)
                frame = cv2.imread(frame_path)
                pose = vo.compute_pose(ImageData(image=frame, timestamp=time()))
                if pose is not None:
                    current_pose = current_pose @ pose
                    estimated_position.append(current_pose[:3, 3])
                    ground_truth_position.append(self.ground_truth[idx])
        except Exception as e:
            print(f"Error during processing: {e}")
            return float('inf')

        ground_truth_position = np.array(ground_truth_position)
        estimated_position = np.array(estimated_position)

        filename = f"{feature_detector}_{feature_matcher}_essential_matrix_prob_{essential_matrix_prob}_threshold_{essential_matrix_threshold}.png"

        plt.figure(figsize=(10, 8))
        px, py, pz = ground_truth_position.T
        plt.plot(px, pz, marker='o', markersize=1, label='Ground Truth Trajectory', color='black')
        px, py, pz = estimated_position.T
        plt.plot(px, pz, marker='o', markersize=1, label='Estimated Trajectory', color='blue')
        plt.title('Estimated Trajectory from Visual Odometry')
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.axis('equal')
        plt.grid()
        plt.legend()
        plt.savefig(os.path.join(self.image_output_dir, filename))

        return mean_absolute_error(ground_truth_position, estimated_position)  # Mean Absolute Error (MAE) of the estimated positions

if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    obj = VisualOdometryObjective2d2D(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
    )

    study = optuna.create_study(direction="minimize")
    study.optimize(obj, n_trials=args.num_trials)
    
    print(study.trials)

    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # save the study results to a CSV file
    output_file = os.path.join(args.output_dir, "vo_2d2d_study.csv")
    df = study.trials_dataframe()
    df.to_csv(output_file, index=False)