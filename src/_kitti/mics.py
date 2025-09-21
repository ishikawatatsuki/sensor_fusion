import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any
from collections import namedtuple

SENSOR_MAPS = {
    'imu_vo_pos': {
        'oxts_imu': {'fields': ['linear_acceleration', 'angular_velocity']},
        'kitti_vo': {'fields': ['position']}
    },
    'imu_vo_vel': {
        'oxts_imu': {'fields': ['linear_acceleration', 'angular_velocity']},
        'kitti_vo': {'fields': ['linear_velocity']}
    },
    'imu_gps': {
        'oxts_imu': {'fields': ['linear_acceleration', 'angular_velocity']},
        'oxts_gps': {'fields': ['position']}
    },
    'imu_gps_vo_pos': {
        'oxts_imu': {'fields': ['linear_acceleration', 'angular_velocity']},
        'oxts_gps': {'fields': ['position']},
        'kitti_vo': {'fields': ['position']}
    },
    'imu_gps_vo_vel': {
        'oxts_imu': {'fields': ['linear_acceleration', 'angular_velocity']},
        'oxts_gps': {'fields': ['position']},
        'kitti_vo': {'fields': ['linear_velocity']}
    },
}

Result = namedtuple('Field', ['gt_pose', 'estimated_pose', 'gt_position', 'estimated_position', 'vo_position', 'mae', 'ate', 'rpe_m', 'rpe_deg', 'inference_time_update', 'inference_measurement_update'])

def save_trajectory_plots(
        gt_positions: np.ndarray, 
        estimated_positions: np.ndarray, 
        vo_positions: np.ndarray, 
        title: str = "Trajectory",
        output_dir: str = "."
    ):
    """ Visualize the trajectory of the estimated positions against the ground truth positions.

    Args:
        gt_positions (np.ndarray): Ground truth positions (N x 3).
        estimated_positions (np.ndarray): Estimated positions (N x 3).
        vo_positions (np.ndarray): Visual Odometry positions (N x 3).
        title (str): Title of the plot.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(gt_positions[:, 0], gt_positions[:, 1], label='Ground Truth trajectory', color='black', linestyle='--')
    ax.plot(estimated_positions[:, 0], estimated_positions[:, 1], label='Fusion estimate', color='blue')

    if vo_positions is not None:
        ax.plot(vo_positions[:, 0], vo_positions[:, 1], label='Visual Odometry estimate', color='red', alpha=0.5)

    ax.set_title(title)
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.legend()
    ax.grid()
    plt.axis('equal')
    plt.savefig(f"{output_dir}/{title}_trajectory.png")
    plt.close(fig)
