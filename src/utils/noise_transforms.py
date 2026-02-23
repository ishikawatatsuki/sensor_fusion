"""
Noise transformation functions for converting sensor specifications 
to process/measurement noise covariance matrices.

These functions are referenced by name in dataset configuration files
and can be reused across different datasets with the same sensor types.
"""

import numpy as np
from typing import Dict, Any
import logging


def imu_noise_to_process_noise(
    gyroscope_noise_density: float,
    accelerometer_noise_density: float,
    gyroscope_random_walk: float,
    accelerometer_random_walk: float,
    dt: float,
    tuning_multipliers: Dict[str, float] = None
) -> np.ndarray:
    """
    Convert IMU noise parameters to process noise covariance diagonal.
    
    Derives process noise from IMU specifications using noise propagation theory:
    - Position noise: from double integration of acceleration
    - Velocity noise: from integration of acceleration
    - Orientation noise: from integration of gyroscope
    - Bias noise: from random walk specifications
    
    Args:
        gyroscope_noise_density: Gyroscope noise density (rad/s/√Hz)
        accelerometer_noise_density: Accelerometer noise density (m/s²/√Hz)
        gyroscope_random_walk: Gyroscope bias random walk (rad/s²/√Hz)
        accelerometer_random_walk: Accelerometer bias random walk (m/s³/√Hz)
        dt: Time step (seconds)
        tuning_multipliers: Optional multipliers for [p, v, q] to account for 
                          model mismatch. Defaults to [1e6, 1e5, 1e3]
    
    Returns:
        Process noise variance array: [p, p, p, v, v, v, q, q, q, q, b_a, b_a, b_a, b_w, b_w, b_w]
    """
    # Theoretical noise propagation
    sigma_v = accelerometer_noise_density * dt
    sigma_p = sigma_v * dt + 0.5 * accelerometer_noise_density * dt**2
    sigma_theta = gyroscope_noise_density * dt
    sigma_q = 0.5 * sigma_theta
    
    # Bias evolution from random walk
    sigma_b_a = accelerometer_random_walk * np.sqrt(dt)
    sigma_b_w = gyroscope_random_walk * np.sqrt(dt)
    
    # Apply tuning multipliers (account for model mismatch, unmodeled dynamics)
    if tuning_multipliers is None:
        tuning_multipliers = {'p': 1e6, 'v': 1e5, 'q': 1e3}
    
    k_p = tuning_multipliers.get('p', 1e6)
    k_v = tuning_multipliers.get('v', 1e5)
    k_q = tuning_multipliers.get('q', 1e3)
    
    # Build variance vector (will be diag of Q matrix)
    process_noise = np.concatenate([
        np.repeat((k_p * sigma_p)**2, 3),  # position [x, y, z]
        np.repeat((k_v * sigma_v)**2, 3),  # velocity [vx, vy, vz]
        np.repeat((k_q * sigma_q)**2, 4),  # quaternion [qw, qx, qy, qz]
        np.repeat(sigma_b_a**2, 3),        # accel bias [bax, bay, baz]
        np.repeat(sigma_b_w**2, 3)         # gyro bias [bwx, bwy, bwz]
    ])
    
    logging.debug(f"IMU Process Noise - σ_p: {k_p * sigma_p:.2e}, σ_v: {k_v * sigma_v:.2e}, "
                  f"σ_q: {k_q * sigma_q:.2e}, σ_b_a: {sigma_b_a:.2e}, σ_b_w: {sigma_b_w:.2e}")
    
    return process_noise


def position_noise_to_measurement_noise(
    position_noise: float,
    heading_noise: float = None
) -> np.ndarray:
    """
    Convert GPS noise parameters to measurement noise covariance diagonal.
    
    Args:
        position_noise: GPS position uncertainty (meters)
        heading_noise: Optional GPS heading uncertainty (degrees)
    
    Returns:
        Measurement noise variance array for GPS measurements
    """
    measurement_noise = np.repeat(position_noise**2, 3)  # [x, y, z]
    
    if heading_noise is not None:
        heading_rad = np.deg2rad(heading_noise)
        measurement_noise = np.append(measurement_noise, heading_rad**2)
    
    return measurement_noise


def vo_noise_to_measurement_noise(
    vo_position_noise: float,
    vo_velocity_noise: float = None,
    vo_orientation_noise: float = None,
    fields: list = None
) -> np.ndarray:
    """
    Convert VO noise parameters to measurement noise covariance diagonal.
    
    Args:
        vo_position_noise: VO position uncertainty (meters)
        vo_velocity_noise: Optional VO velocity uncertainty (m/s)
        vo_orientation_noise: Optional VO orientation uncertainty (degrees)
        fields: List of fields being measured (e.g., ['position', 'orientation'])
    
    Returns:
        Measurement noise variance array for VO measurements
    """
    measurement_noise = np.array([])
    
    # Build based on which fields are actually being measured
    if fields is None or 'position' in fields:
        measurement_noise = np.concatenate([
            measurement_noise,
            np.repeat(vo_position_noise**2, 3)
        ])
    
    if vo_velocity_noise is not None and (fields is None or 'linear_velocity' in fields):
        measurement_noise = np.concatenate([
            measurement_noise,
            np.repeat(vo_velocity_noise**2, 3)
        ])
    
    if vo_orientation_noise is not None and (fields is None or 'orientation' in fields):
        orientation_rad = np.deg2rad(vo_orientation_noise)
        measurement_noise = np.concatenate([
            measurement_noise,
            np.repeat(orientation_rad**2, 4)  # quaternion
        ])
    
    return measurement_noise


def lateral_velocity_noise_to_measurement_noise(
    upward_velocity_noise: float,
    leftward_velocity_noise: float
) -> np.ndarray:
    """
    Convert lateral velocity noise to measurement noise covariance diagonal.
    
    Args:
        upward_velocity_noise: Upward velocity uncertainty (m/s)
        leftward_velocity_noise: Leftward velocity uncertainty (m/s)
    
    Returns:
        Measurement noise variance array [upward, leftward]
    """
    return np.array([
        upward_velocity_noise**2,
        leftward_velocity_noise**2
    ])


# Registry of available transformation functions
NOISE_TRANSFORMS = {
    'imu_noise_to_process_noise': imu_noise_to_process_noise,
    'position_noise_to_measurement_noise': position_noise_to_measurement_noise,
    'vo_noise_to_measurement_noise': vo_noise_to_measurement_noise,
    'lateral_velocity_noise_to_measurement_noise': lateral_velocity_noise_to_measurement_noise,
}


def get_transform_function(name: str):
    """
    Get a noise transformation function by name.
    
    Args:
        name: Name of the transformation function
    
    Returns:
        Callable transformation function
    
    Raises:
        ValueError: If transformation function not found
    """
    if name not in NOISE_TRANSFORMS:
        available = ', '.join(NOISE_TRANSFORMS.keys())
        raise ValueError(f"Unknown transformation '{name}'. Available: {available}")
    
    return NOISE_TRANSFORMS[name]