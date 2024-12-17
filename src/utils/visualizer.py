import os
import sys
import cv2
import logging
import numpy as np
import pandas as pd
import seaborn as sns
from typing import List
from enum import IntEnum, auto
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import namedtuple

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../interfaces'))

from config import FilterConfig, VisualizationConfig, DatasetConfig
from custom_types import FilterType, DatasetType, SensorType
from interfaces import Pose

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(
        format='[%(levelname)s] %(asctime)s > [%(name)20s] %(message)s', 
        datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG
    )
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)


BaseVisualizationField = namedtuple('BaseVisualizationField', ['x', 'y', 'z', 'lw', 'color'])

PoseVisualizationField = namedtuple('PoseVisualizationField', ['type', 'pose', 'lw', 'color'])
ScatterPoints = namedtuple('ScatterPoints', ['positions', 'sizes'])

# TODO: Delete this type and merge with the pose visualization filed.
FilterVisualizationField = namedtuple('FilterVisualizationField', ['type', 'x', 'y', 'z'])

class VisualizationDataType(IntEnum):
    GROUND_TRUTH = auto()
    ESTIMATION = auto() # Fusion estimation
    VO = auto()
    IMAGE = auto()
    QUALITY = auto()

class VIO_Visualizer:
    
    def __init__(
        self,
        config: VisualizationConfig
    ):
        self.config = config
        
        self.show_vio_frame = config.show_vio_frame
        
        self.pf_scatter_points = None
        self.enkf_scatter_points = None
        
        self.figure = None
        self.image_window = None
        self.ekf_window = None
        self.ukf_window = None
        self.pf_window = None
        self.enkf_window = None
        self.ckf_window = None
        
        self.prev_trajectories = {
            FilterType.EKF: None,
            FilterType.UKF: None,
            FilterType.PF: None,
            FilterType.EnKF: None,
            FilterType.CKF: None,
        }
        self.prev_gt_trajectory = None
        
        self.gt_color = "black"
        self.filter_color = "red"
        self.legend_loc = "upper right"
        self.gt_label = "GPS"
        self.frame = 0
        self.output_frame_path = os.path.join(self.config.output_filepath, "frames1")
        if self.config.save_frames and not os.path.exists(self.output_frame_path):
            os.makedirs(self.output_frame_path)
            
    def _get_geo_limit(self, axis=0):
        if axis==0: # x-axis
            return (self.config.limits.min[0], self.config.limits.max[0])
        elif axis==1: # y-axis
            return (self.config.limits.min[1], self.config.limits.max[1])
        else: # z-axis
            return (self.config.limits.min[2], self.config.limits.max[2])
        
    def _save_current_frame(self):
        if self.config.save_frames and self.figure is not None:
            self.figure.savefig(os.path.join(self.output_frame_path, f"{str(self.frame)}.png"))
            self.frame += 1
        
    def start(self):
        if not self.show_vio_frame:
            return
        plt.ion()  # Enable interactive mode
        
        self.figure = plt.figure(figsize=(16, 10))
        gs = self.figure.add_gridspec(ncols=3, nrows=3)
        
        self.image_window = self.figure.add_subplot(gs[:2, :2]) 
        
        self.ekf_window = self.figure.add_subplot(gs[2, 0]) 
        self.ekf_window.title.set_text("EKF estimation")
        self.ekf_window.set_xlabel("X[m]")
        self.ekf_window.set_ylabel("Y[m]")
        
        self.ukf_window = self.figure.add_subplot(gs[2, 1]) 
        self.ukf_window.title.set_text("UKF estimation")
        self.ukf_window.set_xlabel("X[m]")
        self.ukf_window.set_ylabel("Y[m]")
        
        self.pf_window = self.figure.add_subplot(gs[0, 2]) 
        self.pf_window.title.set_text("PF estimation")
        self.pf_window.set_xlabel("X[m]")
        self.pf_window.set_ylabel("Y[m]")
        
        self.enkf_window = self.figure.add_subplot(gs[1, 2]) 
        self.enkf_window.title.set_text("EnKF estimation")
        self.enkf_window.set_xlabel("X[m]")
        self.enkf_window.set_ylabel("Y[m]")
        
        self.ckf_window = self.figure.add_subplot(gs[2, 2]) 
        self.ckf_window.title.set_text("CKF estimation")
        self.ckf_window.set_xlabel("X[m]")
        self.ckf_window.set_ylabel("Y[m]")

        initial_pose = [0, 0]
        self.ekf_window.plot(
            initial_pose, 
            initial_pose, 
            color=self.gt_color,
            label=self.gt_label
        )
        self.ukf_window.plot(
            initial_pose, 
            initial_pose, 
            color=self.gt_color,
            label=self.gt_label
        )
        self.pf_window.plot(
            initial_pose, 
            initial_pose, 
            color=self.gt_color,
            label=self.gt_label
        )
        self.enkf_window.plot(
            initial_pose, 
            initial_pose, 
            color=self.gt_color,
            label=self.gt_label
        )
        self.ckf_window.plot(
            initial_pose, 
            initial_pose, 
            color=self.gt_color,
            label=self.gt_label
        )
        
        self.ekf_window.plot(
            initial_pose, 
            initial_pose, 
            color=self.filter_color,
            label="EKF"
        )
        self.ukf_window.plot(
            initial_pose, 
            initial_pose, 
            color=self.filter_color,
            label="UKF"
        )
        self.pf_window.plot(
            initial_pose, 
            initial_pose, 
            color=self.filter_color,
            label="PF"
        )
        self.enkf_window.plot(
            initial_pose, 
            initial_pose, 
            color=self.filter_color,
            label="EnKF"
        )
        self.ckf_window.plot(
            initial_pose, 
            initial_pose, 
            color=self.filter_color,
            label="CKF"
        )
        self.ekf_window.legend(loc=self.legend_loc)
        self.ukf_window.legend(loc=self.legend_loc)
        self.pf_window.legend(loc=self.legend_loc)
        self.enkf_window.legend(loc=self.legend_loc)
        self.ckf_window.legend(loc=self.legend_loc)
        
        if self.config.set_lim_in_plot:
            self.ekf_window.set_xlim(self._get_geo_limit(0))
            self.ekf_window.set_ylim(self._get_geo_limit(1))
            self.ukf_window.set_xlim(self._get_geo_limit(0))
            self.ukf_window.set_ylim(self._get_geo_limit(1))
            self.pf_window.set_xlim(self._get_geo_limit(0))
            self.pf_window.set_ylim(self._get_geo_limit(1))
            self.enkf_window.set_xlim(self._get_geo_limit(0))
            self.enkf_window.set_ylim(self._get_geo_limit(1))
            self.ckf_window.set_xlim(self._get_geo_limit(0))
            self.ckf_window.set_ylim(self._get_geo_limit(1))
        
        plt.tight_layout()
        logger.debug("Creating VIO plot frame.")
    
    def stop(self):
        if not self.show_vio_frame:
            return
        
        self.image_window = None
        self.ekf_window = None
        self.ukf_window = None
        self.pf_window = None
        self.enkf_window = None
        self.ckf_window = None
        plt.close(self.figure)
        
    def set_frame(self, image_path):
        image_path = os.path.abspath(image_path)
        if os.path.exists(image_path):
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            self.image_window.clear()
            self.image_window.imshow(image)
            self.image_window.axis("off")  # Hide axes for better viewing
            plt.draw()
            plt.pause(0.1)
            self._save_current_frame()
    
    def set_particles(self, particles: np.ndarray, weights: np.ndarray = None, filter_type: FilterType = FilterType.PF):
        """ Set particles to visualize realtime.

        Args:
            particles (np.ndarray): < 100 to save computation
            weights (np.ndarray): same size as particles
        """
        if not self.config.show_particles:
            return
        
        max_particle_size = 10
        if filter_type is FilterType.PF:
            mean, std = np.mean(weights), np.std(weights)
            q1 = mean - std
            q3 = mean + std
            sizes = np.tile(max_particle_size, particles.shape[0])
            
            sizes[weights < q1] = np.round(0.1 * max_particle_size, 2)
            sizes[(q1 <= weights) & (weights < mean)] = np.round(0.33 * max_particle_size, 2)
            sizes[(mean <= weights) * (weights < q3)] = np.round(0.66 * max_particle_size, 2)
            
            positions = particles[:, :3]
            
            self.pf_scatter_points = ScatterPoints(positions=positions, sizes=sizes)
        elif filter_type is FilterType.EnKF:
            
            sizes = np.tile(max_particle_size, particles.shape[0])
            positions = particles[:, :3]
            
            self.pf_scatter_points = ScatterPoints(positions=positions, sizes=sizes)
        
    def set_ground_truth_trajectory(self, data: BaseVisualizationField):
        if self.prev_gt_trajectory is None:
            self.prev_gt_trajectory = data
            return
        
        self.ekf_window.plot(
            [self.prev_gt_trajectory.x, data.x], 
            [self.prev_gt_trajectory.y, data.y], color=self.gt_color,
            label=self.gt_label
        )
        self.ukf_window.plot(
            [self.prev_gt_trajectory.x, data.x], 
            [self.prev_gt_trajectory.y, data.y], color=self.gt_color,
            label=self.gt_label
        )
        self.pf_window.plot(
            [self.prev_gt_trajectory.x, data.x], 
            [self.prev_gt_trajectory.y, data.y], color=self.gt_color,
            label=self.gt_label
        )
        self.enkf_window.plot(
            [self.prev_gt_trajectory.x, data.x], 
            [self.prev_gt_trajectory.y, data.y], color=self.gt_color,
            label=self.gt_label
        )
        self.ckf_window.plot(
            [self.prev_gt_trajectory.x, data.x], 
            [self.prev_gt_trajectory.y, data.y], color=self.gt_color,
            label=self.gt_label
        )
        
        self.prev_gt_trajectory = data
        
    def set_filter_estimation(self, data: List[FilterVisualizationField]):
        for current_data in data:
            prev_data = self.prev_trajectories[current_data.type]
            self.prev_trajectories[current_data.type] = BaseVisualizationField(x=current_data.x, y=current_data.y, z=current_data.z, lw=1)
            if prev_data is None:
                continue
            
            match (current_data.type):
                case FilterType.EKF:
                    self.ekf_window.plot(
                        [prev_data.x, current_data.x], 
                        [prev_data.y, current_data.y], color=self.filter_color,
                        label="EKF"
                    )
                case FilterType.UKF:
                    self.ukf_window.plot(
                        [prev_data.x, current_data.x], 
                        [prev_data.y, current_data.y], color=self.filter_color,
                        label="UKF"
                    )
                case FilterType.PF:
                    self.pf_window.plot(
                        [prev_data.x, current_data.x], 
                        [prev_data.y, current_data.y], color=self.filter_color,
                        label="PF"
                    )
                    if self.config.show_particles and\
                        self.pf_scatter_points is not None:
                        self.pf_window.scatter(
                            self.pf_scatter_points.positions[:, 0],
                            self.pf_scatter_points.positions[:, 1],
                            s=self.pf_scatter_points.sizes
                        )
                case FilterType.EnKF:
                    self.enkf_window.plot(
                        [prev_data.x, current_data.x], 
                        [prev_data.y, current_data.y], color=self.filter_color,
                        label="EnKF"
                    )
                    if self.config.show_particles and\
                        self.enkf_scatter_points is not None:
                        self.pf_window.scatter(
                            self.enkf_scatter_points.positions[:, 0],
                            self.enkf_scatter_points.positions[:, 1],
                            s=self.enkf_scatter_points.sizes
                        )
                case FilterType.CKF:
                    self.ckf_window.plot(
                        [prev_data.x, current_data.x], 
                        [prev_data.y, current_data.y], color=self.filter_color,
                        label="CKF"
                    )
        
        plt.pause(interval=0.005)
        self._save_current_frame()
        
# TODO: Implement visual odometory visual debugger
class VisualOdometryVisualizer:
    
    def __init__(
        self,
        config: VisualizationConfig,
        dataset_config: DatasetConfig,
        vo_config: VisualizationConfig
    ):
        self.config = config
        self.dataset_config = dataset_config
        self.vo_config = vo_config

        self.output_filename = os.path.join(self.config.output_filepath, "vo_estimation.py")
        
        logger.info("Configuring Visual Odometry visualizer.")
        self._create_window(dataset_type=DatasetType.get_type_from_str(self.dataset_config.type))
        
    def _create_window(self, dataset_type: DatasetType):
        self.figure = plt.figure(figsize = (7, 7))
        self.window = self.figure.add_subplot()
        self.window.set_xlabel('X (km)')
        self.window.set_ylabel('Y (km)')
        
        match (dataset_type):
            case DatasetType.KITTI:
                self.window.set_title('VO estimated trajectory VS Ground truth')
            case _: # UAV, KAGARU, OAK
                self.window.set_title('VO estimated trajectory')
                
        plt.grid()
        
    def _image_rect_helper(
        self,
        image1:np.ndarray, 
        image2:np.ndarray, 
        image3:np.ndarray, 
        image4:np.ndarray, 
        image5:np.ndarray
    ):
        W = 2160  # Total width
        H = 1080   # Total height (for 3 rows)
        
        final_image = np.zeros((H, W, 3), dtype=np.uint8)
        
        # Calculate widths and heights based on percentages
        w1 = int(W * 0.35)
        w2 = W - w1
        h1 = h2 = h3 = int(H / 3)
        
        # Resize images to fit the layout
        img1_resized = cv2.resize(image1, (w1, h1))
        img2_resized = cv2.resize(image2, (w2, h1))
        img3_resized = cv2.resize(image3, (w2, h2))
        img4_resized = cv2.resize(image4, (w2, h3))
        img5_resized = cv2.resize(image5, (w1, h2 + h3))
        
        final_image[0:h1, 0:w1] = img1_resized
        final_image[0:h1, w1:W] = img2_resized
        final_image[h1:h1+h2, 0:w2] = img3_resized
        final_image[h1+h2:H, 0:w2] = img4_resized
        final_image[h1:H, w2:W] = img5_resized

        return final_image 
    
    def set_estimated_point(self, point):
        if not self.config.show_vio_frame:
            ...
        pass
        
    def set_reference_point(self, point):
        if not self.config.show_vio_frame:
            ...
        pass
        
    def export_image(self):
        ...
        
class Visualizer:

    colors = ["black", "red", "blue", "green", "pink", "yellow"]
    
    def __init__(
            self,
            config: VisualizationConfig,
            filter_config: FilterConfig,
        ):
        
        self.config = config
        self.dimension = filter_config.dimension
        self.filter_type = filter_config.type
        
        if self.config.save_trajectory and not os.path.exists(self.config.output_filepath):
            os.makedirs(self.config.output_filepath)
        
        self.output_realtime_figure_filename = os.path.join(self.config.output_filepath, "estimated_trajectory_new.png")
        
        self.labels = self._get_labels()
        self.num_of_trajectories = len(self.labels)
        
        self.realtime_window = None
        self.realtime_fig = None
        self.prev_data = None
        self.scatter_points = None
        
        self.all_trajectories = []
        
        self.angle_window = None
        self.angle_fig = None
        self.angle_index = 0
        self.initial_poses = None
        
        self.gt_angles = []
        self.estimated_angles = []
        
    def start(self, initial_poses: List[PoseVisualizationField]):
        
        self.initial_poses = initial_poses
        self.prev_data = self.initial_poses
            
        if not self.config.realtime:
            return
        
        if self.dimension == 2:
            self.realtime_fig, self.realtime_window = plt.subplots(1, 1, figsize=(8, 6))
            self.realtime_window.set_title("Trajectory comparison", size=18)
            
            
            for i, initial_pose in enumerate(self.initial_poses):
                x, y, z = initial_pose.pose.t.flatten()
                self.realtime_window.plot(x, y, color=initial_pose.color)
            
            self.realtime_window.set_xlabel('X [m]', fontsize=16)
            self.realtime_window.set_ylabel('Y [m]', fontsize=16)
            
            self.realtime_window.grid()
            
        else:
            self.realtime_fig = plt.figure()

            self.realtime_window = self.realtime_fig.add_subplot(111, projection='3d')
            self.realtime_window.set_title("Trajectory comparison")
            
            for i, initial_pose in enumerate(self.initial_poses):
                x, y, z = initial_pose.pose.t.flatten()
                self.realtime_window.plot(x, y, z, color=initial_pose.color)
                

            self.realtime_window.set_xlabel('X [m]', fontsize=16)
            self.realtime_window.set_ylabel('Y [m]', fontsize=16)
            self.realtime_window.set_zlabel('Z [m]', fontsize=16)
        
        if self.config.set_lim_in_plot:
            self.realtime_window.set_xlim([self.config.limits.min[0], self.config.limits.max[0]])
            self.realtime_window.set_ylim([self.config.limits.min[1], self.config.limits.max[1]])
            if self.dimension == 3:
                self.realtime_window.set_zlim([self.config.limits.min[2], self.config.limits.max[2]])
                
        self.realtime_window.legend(
            handles=self.labels, 
            loc='upper right', 
            # bbox_to_anchor=(1.1, 0., 0.2, 0.9)
        )
        self.realtime_window.tick_params(axis='both', which='major', labelsize=12)
        self.realtime_fig.tight_layout()
        
    def stop(self):
        if self.config.save_trajectory and self.realtime_fig is not None:
            self.realtime_fig.savefig(self.output_realtime_figure_filename, bbox_inches = None)
        
        self.realtime_window = None
        self.realtime_fig = None
        plt.close('all')
        
    def show_final_estimation(self, filename=None):
        self.all_trajectories = np.array(self.all_trajectories).T
        
        if self.dimension == 2:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            ax.set_title("Trajectory comparison", size=18)
            
            for i, trajectory in enumerate(self.all_trajectories):
                xs, ys, _ = trajectory
                lw = 2 if i < 2 else 1
                ax.plot(xs, ys, color=self.colors[i], lw=lw)

                ax.set_xlabel('X [m]', fontsize=16)
                ax.set_ylabel('Y [m]', fontsize=16)
                ax.tick_params(axis='both', which='major', labelsize=12)
                
            ax.legend(handles=self.labels, prop={'size': 12})
            ax.grid()
        else:
            fig = plt.figure()

            ax = fig.add_subplot(111, projection='3d')
            ax.set_title("Trajectory comparison", size=18)
            
            for i, trajectory in enumerate(self.all_trajectories):
                lw = 2 if i < 2 else 1
                xs, ys, zs = trajectory
                ax.plot(xs, ys, zs, color=self.colors[i], lw=lw)

            ax.set_xlabel('X [m]', fontsize=16)
            ax.set_ylabel('Y [m]', fontsize=16)
            ax.set_zlabel('Z [m]', fontsize=16)
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.legend(handles=self.labels, loc='best', bbox_to_anchor=(1.1, 0., 0.2, 0.9))
            fig.tight_layout()
        
        if self.config.save_trajectory:
            filename = filename if filename is not None else "final_result.png"
            if filename.endswith(".png"):
                filename += ".png"
            
            fig.savefig(os.path.join(self.config.output_filepath, filename))
        
        if self.config.show_end_result:
            plt.show()

    def show_realtime_estimation(self, data: List[PoseVisualizationField]):
        if  self.prev_data is None or\
            self.config.show_vo_trajectory and\
            len(data) != len(self.prev_data):
            self.prev_data = data
            return
        
        trajectories = []
        for current_data in data:
            initial_pose_list = [initial_pose for initial_pose in self.initial_poses if initial_pose.type == current_data.type]
            if len(initial_pose_list) == 0:
                continue
            prev_data_list = [prev for prev in self.prev_data if prev.type == current_data.type]
            if len(prev_data_list) == 0:
                continue
            
            initial_data = initial_pose_list[0]
            prev_data = prev_data_list[0]
            
            prev_pose_T = (initial_data.pose.inverse() * prev_data.pose).t.flatten()
            current_pose_T = (initial_data.pose.inverse() * current_data.pose).t.flatten()
            
            
            if self.config.realtime:
                if self.dimension == 2:
                    self.realtime_window.plot(
                        [prev_pose_T[0], current_pose_T[0]], 
                        [prev_pose_T[1], current_pose_T[1]], 
                        color=current_data.color,
                        lw=current_data.lw,
                        )
                else:
                    self.realtime_window.plot(
                        [prev_pose_T[0], current_pose_T[0]],
                        [prev_pose_T[1], current_pose_T[1]], 
                        [prev_pose_T[2], current_pose_T[2]], 
                        color=current_data.color,
                        lw=current_data.lw
                        )
            
            trajectories.append(current_pose_T)
        
        if len(trajectories) == self.num_of_trajectories:
            self.all_trajectories.append(np.array(trajectories).T)
            
        if self.config.realtime:
            plt.pause(interval=0.005)
            
        self.prev_data = data
        
    def set_particles(self, particles: np.ndarray, weights: np.ndarray = None, filter_type: FilterType = FilterType.PF):
        """ Set particles to visualize realtime.

        Args:
            particles (np.ndarray): < 100 to save computation
            weights (np.ndarray): same size as particles
        """
        if not self.config.show_particles or self.realtime_window is None:
            return
        
        if not FilterType.is_probabilistic_filter(filter_type=filter_type.value):
            return
        
        if filter_type is FilterType.PF:
            max_particle_size = 10
            mean, std = np.mean(weights), np.std(weights)
            q1 = mean - std
            q3 = mean + std
            sizes = np.tile(max_particle_size, particles.shape[0])
            
            sizes[weights < q1] = np.round(0.1 * max_particle_size, 2)
            sizes[(q1 <= weights) & (weights < mean)] = np.round(0.33 * max_particle_size, 2)
            sizes[(mean <= weights) * (weights < q3)] = np.round(0.66 * max_particle_size, 2)
            
            positions = particles[:, :3]
            
        elif filter_type is FilterType.EnKF:
            sizes = np.tile(2, particles.shape[0])
            positions = particles[:, :3]
            
        # NOTE: Visualize particles for sampling based filters
        self.realtime_window.scatter(
            positions[:, 0],
            positions[:, 1],
            s=sizes
        )
    
    def set_angle_estimation(self, data: List[BaseVisualizationField]):
        if not self.config.show_angle_estimation:
            return
        
        for i, current_angle in enumerate(data):
            angle = np.array([current_angle.x, current_angle.y, current_angle.z])
            if i == 0:
                self.gt_angles.append(angle)
            else:
                self.estimated_angles.append(angle)

    def show_angle_estimation(self):
        if not self.config.show_angle_estimation:
            return
        
        self.gt_angles = np.array(self.gt_angles)
        self.estimated_angles = np.array(self.estimated_angles)
        
        fig, axs = plt.subplots(3, 1, figsize=(8, 6))
        plt.suptitle("Angle comparison (ground-truth vs estimation)")
        
        for i, axis in enumerate(["x", "y", "z"]):
            axs[i].plot(
                self.gt_angles[:, i], 
                label=f"Ground-truth angle in {axis}-axis", 
                color="black"
            )
            axs[i].plot(
                self.estimated_angles[:, i], 
                label=f"Estimated angle in {axis}-axis", 
                color="red"
            )
            axs[i].legend(loc='best', bbox_to_anchor=(1.1, 0., 0.2, 0.9))
            
            axs[i].set_xlabel('index', fontsize=14)
            axs[i].set_ylabel('rad', fontsize=14)
            
            axs[i].grid()
            
        plt.tight_layout()
        plt.show()
        
    
    def show_innovation(self, innovations):
        if self.config.show_innovation_history:
            innovations = np.array(innovations)
            plt.plot(np.arange(innovations.shape[0]), innovations)
            plt.tight_layout()
            plt.show()
            
    def _get_labels(self):
        
        gt_label = mpatches.Patch(color='black', label='Ground-truth trajectory')
        estimation_label = mpatches.Patch(color='red', label='Estimated trajectory')
        labels = [gt_label, estimation_label]
        if self.config.show_vo_trajectory:
            vo_label = mpatches.Patch(color='blue', label='VO estimation')
            labels.append(vo_label)
        return labels
        
        
    @staticmethod
    def show_filter_comparison(
            df: pd.DataFrame,
            title: str="Filter results", 
            is_saving: bool=False, 
            path: str=None, 
            labels: List[str] = None
        ):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
        xy_labels = labels if labels is not None else ["Filter types", "Error metrics"]
        
        sns.heatmap(df,
                    ax=ax,
                    cmap="crest",
                    annot=True,
                    linewidths=1,
                    fmt='.3g')
        ax.set_title(title)
        ax.set(xlabel=xy_labels[0], ylabel=xy_labels[1])
        
        if is_saving:
            fig.savefig(path)
            logger.info("The result is saved successfully.")
                    
if __name__ == "__main__":
    from tqdm import tqdm
    from PIL import Image

    filter_config = FilterConfig(
        type="ekf",
        dimension=3,
        motion_model="kinematics",
        noise_type=False,
        params=None
    )
    
    config = VisualizationConfig(
        realtime=True,
        output_filepath="./",
        save_trajectory=True,
        show_end_result=True,
    )
    
    visualizer = Visualizer(
        config=config,
        filter_config=filter_config
    )
    
    visualizer.start(labels=["Ground truth trajectory", "Estimated Trajectory"])
    
    
    zline = np.linspace(0, 15, 50)
    xline = np.sin(zline)
    yline = np.cos(zline)
    
    xline2 = np.cos(zline)
    yline2 = np.sin(zline)
        
    zdata = 15 * np.random.random(100)
    xdata = np.sin(zdata) + 0.1 * np.random.randn(100)
    ydata = np.cos(zdata) + 0.1 * np.random.randn(100)
    
    for i, z in tqdm(enumerate(zline)):
        data = PoseVisualizationField(
            type=VisualizationDataType.GROUND_TRUTH,
            pose=Pose(R=np.eye(3), t=np.array([xline[i], yline[i], z]).reshape(-1, 1)),
            lw=2,
            color='black'
        )
        data2 = PoseVisualizationField(
            type=VisualizationDataType.GROUND_TRUTH,
            pose=Pose(R=np.eye(3), t=np.array([xline2[i], yline2[i], z]).reshape(-1, 1)),
            lw=2,
            color='red'
        )
        visualizer.show_realtime_estimation(data=[data, data2])
    
    visualizer.show_final_estimation()
    visualizer.stop()
