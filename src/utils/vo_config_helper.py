"""
Visual Odometry Configuration Helper

This module provides utilities for handling flexible visual odometry configuration:
- Detecting pre-computed VO data availability
- Managing fallback to runtime VO computation
- Ensuring camera data is available when needed
"""

import os
import logging
from typing import Optional, Dict, Any, Tuple
from pathlib import Path


class VOConfigHelper:
    """Helper class for managing visual odometry configuration and availability."""
    
    # Mapping of VO estimation types to their directory suffixes
    VO_SUFFIX_MAP = {
        '2d2d': '',
        '2d3d': '_2d3d',
        'hybrid': '_hybrid',
        'stereo': '_stereo',
        'epipolar': '',
        'pnp': '_2d3d'
    }
    
    @staticmethod
    def get_vo_path(
        root_path: str,
        variant: str,
        dataset_type: str,
        estimation_type: str = '2d3d',
        vo_suffix: Optional[str] = None
    ) -> str:
        """
        Construct the path to pre-computed VO data.
        
        Args:
            root_path: Dataset root path
            variant: Dataset variant/sequence
            dataset_type: Dataset type ('euroc', 'kitti', etc.)
            estimation_type: VO estimation type ('2d2d', '2d3d', 'hybrid', etc.)
            vo_suffix: Custom suffix override (if None, uses estimation_type)
            
        Returns:
            Path to VO data file
        """
        if vo_suffix is None:
            vo_suffix = VOConfigHelper.VO_SUFFIX_MAP.get(estimation_type.lower(), '')
        
        if dataset_type.lower() == 'euroc':
            vo_dir = f"vo_pose_estimates{vo_suffix}"
            vo_file = f"MH_{variant}_easy_poses.txt"
            return os.path.join(root_path, vo_dir, vo_file)
            
        elif dataset_type.lower() == 'kitti':
            vo_dir = f"vo_pose_estimates{vo_suffix}"
            vo_file = f"{variant}.txt"
            return os.path.join(root_path, vo_dir, vo_file)
            
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")
    
    @staticmethod
    def check_vo_data_exists(
        root_path: str,
        variant: str,
        dataset_type: str,
        estimation_type: str = '2d3d',
        vo_suffix: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        Check if pre-computed VO data exists.
        
        Args:
            root_path: Dataset root path
            variant: Dataset variant/sequence
            dataset_type: Dataset type
            estimation_type: VO estimation type
            vo_suffix: Custom suffix override
            
        Returns:
            Tuple of (exists: bool, path: str)
        """
        vo_path = VOConfigHelper.get_vo_path(
            root_path=root_path,
            variant=variant,
            dataset_type=dataset_type,
            estimation_type=estimation_type,
            vo_suffix=vo_suffix
        )
        
        exists = os.path.exists(vo_path) and os.path.isfile(vo_path)
        
        if exists:
            file_size = os.path.getsize(vo_path)
            if file_size == 0:
                logging.warning(f"Pre-computed VO file exists but is empty: {vo_path}")
                return False, vo_path
        
        return exists, vo_path
    
    @staticmethod
    def determine_vo_mode(
        sensor_config: Dict[str, Any],
        root_path: str,
        variant: str,
        dataset_type: str
    ) -> Dict[str, Any]:
        """
        Determine the VO mode based on configuration and data availability.
        
        Args:
            sensor_config: Sensor configuration dictionary
            root_path: Dataset root path
            variant: Dataset variant/sequence
            dataset_type: Dataset type
            
        Returns:
            Dictionary with VO mode information:
            {
                'mode': 'precomputed' or 'runtime',
                'path': path to VO data (if precomputed),
                'estimation_type': estimation type to use,
                'requires_camera': bool
            }
        """
        args = sensor_config.get('args', {})
        use_precomputed = args.get('use_precomputed', True)
        estimation_type = args.get('estimation_type', '2d3d')
        vo_suffix = args.get('vo_suffix', None)
        
        result = {
            'mode': 'runtime',
            'path': None,
            'estimation_type': estimation_type,
            'requires_camera': True  # Always need camera for runtime fallback
        }
        
        if use_precomputed:
            exists, vo_path = VOConfigHelper.check_vo_data_exists(
                root_path=root_path,
                variant=variant,
                dataset_type=dataset_type,
                estimation_type=estimation_type,
                vo_suffix=vo_suffix
            )
            
            if exists:
                result['mode'] = 'precomputed'
                result['path'] = vo_path
                logging.info(f"Pre-computed VO data found: {vo_path}")
            else:
                logging.info(
                    f"Pre-computed VO data not found at {vo_path}. "
                    f"Will compute VO at runtime using {estimation_type} method."
                )
        else:
            logging.info(f"Configured to compute VO at runtime using {estimation_type} method.")
        
        return result
    
    @staticmethod
    def validate_camera_availability(
        sensors: Dict[str, Any],
        dataset_type: str,
        vo_mode: str
    ) -> bool:
        """
        Validate that camera sensors are available when needed for VO.
        
        Args:
            sensors: Dictionary of sensor configurations
            dataset_type: Dataset type
            vo_mode: VO mode ('precomputed' or 'runtime')
            
        Returns:
            True if camera is available or not needed, False otherwise
        """
        # Camera is always needed for runtime VO
        if vo_mode == 'runtime':
            camera_sensors = {
                'euroc': ['euroc_stereo'],
                'kitti': ['kitti_stereo'],
                'uav': ['voxl_stereo', 'px4_stereo']
            }
            
            required_cameras = camera_sensors.get(dataset_type.lower(), [])
            
            for camera_sensor in required_cameras:
                if camera_sensor in sensors and sensors[camera_sensor].get('selected', False):
                    logging.info(f"Camera sensor '{camera_sensor}' is available for runtime VO computation.")
                    return True
            
            logging.error(
                f"Runtime VO computation requires a camera sensor, but none found. "
                f"Expected one of: {required_cameras}"
            )
            return False
        
        # Pre-computed VO doesn't strictly need camera, but it's recommended
        if vo_mode == 'precomputed':
            logging.info("Using pre-computed VO data. Camera sensor not strictly required.")
            return True
        
        return True
    
    @staticmethod
    def get_vo_config_summary(
        sensor_name: str,
        sensor_config: Dict[str, Any],
        root_path: str,
        variant: str,
        dataset_type: str
    ) -> str:
        """
        Generate a human-readable summary of the VO configuration.
        
        Args:
            sensor_name: Name of the VO sensor
            sensor_config: Sensor configuration
            root_path: Dataset root path
            variant: Dataset variant
            dataset_type: Dataset type
            
        Returns:
            Summary string
        """
        vo_mode_info = VOConfigHelper.determine_vo_mode(
            sensor_config=sensor_config,
            root_path=root_path,
            variant=variant,
            dataset_type=dataset_type
        )
        
        summary_lines = [
            f"\n{'='*60}",
            f"Visual Odometry Configuration: {sensor_name}",
            f"{'='*60}",
            f"Dataset: {dataset_type.upper()}, Variant: {variant}",
            f"Mode: {vo_mode_info['mode'].upper()}",
            f"Estimation Type: {vo_mode_info['estimation_type']}",
        ]
        
        if vo_mode_info['mode'] == 'precomputed':
            summary_lines.append(f"Data Path: {vo_mode_info['path']}")
        else:
            summary_lines.append("Data Path: N/A (computing at runtime)")
        
        summary_lines.append(f"Requires Camera: {vo_mode_info['requires_camera']}")
        summary_lines.append(f"{'='*60}\n")
        
        return '\n'.join(summary_lines)


def check_and_configure_vo(
    sensors: Dict[str, Any],
    root_path: str,
    variant: str,
    dataset_type: str
) -> Dict[str, Any]:
    """
    Main function to check and configure visual odometry based on availability.
    
    This function:
    1. Identifies VO sensors in the configuration
    2. Checks for pre-computed VO data
    3. Ensures camera sensors are available if needed
    4. Returns updated configuration
    
    Args:
        sensors: Dictionary of sensor configurations
        root_path: Dataset root path
        variant: Dataset variant/sequence
        dataset_type: Dataset type
        
    Returns:
        Dictionary with VO configuration status:
        {
            'vo_sensor': name of VO sensor or None,
            'vo_mode': 'precomputed', 'runtime', or None,
            'vo_path': path to VO data (if precomputed),
            'camera_available': bool,
            'estimation_type': estimation type
        }
    """
    # Identify VO sensors
    vo_sensors = [
        name for name, config in sensors.items()
        if 'vo' in name.lower() and config.get('selected', False)
    ]
    
    if not vo_sensors:
        logging.info("No visual odometry sensors selected.")
        return {
            'vo_sensor': None,
            'vo_mode': None,
            'vo_path': None,
            'camera_available': False,
            'estimation_type': None
        }
    
    # Use the first VO sensor found
    vo_sensor_name = vo_sensors[0]
    vo_sensor_config = sensors[vo_sensor_name]
    
    if len(vo_sensors) > 1:
        logging.warning(
            f"Multiple VO sensors selected: {vo_sensors}. "
            f"Using: {vo_sensor_name}"
        )
    
    # Determine VO mode
    vo_mode_info = VOConfigHelper.determine_vo_mode(
        sensor_config=vo_sensor_config,
        root_path=root_path,
        variant=variant,
        dataset_type=dataset_type
    )
    
    # Validate camera availability
    camera_available = VOConfigHelper.validate_camera_availability(
        sensors=sensors,
        dataset_type=dataset_type,
        vo_mode=vo_mode_info['mode']
    )
    
    # Print summary
    summary = VOConfigHelper.get_vo_config_summary(
        sensor_name=vo_sensor_name,
        sensor_config=vo_sensor_config,
        root_path=root_path,
        variant=variant,
        dataset_type=dataset_type
    )
    logging.info(summary)
    
    return {
        'vo_sensor': vo_sensor_name,
        'vo_mode': vo_mode_info['mode'],
        'vo_path': vo_mode_info['path'],
        'camera_available': camera_available,
        'estimation_type': vo_mode_info['estimation_type']
    }


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Example 1: EuRoC dataset with pre-computed VO
    euroc_sensors = {
        'euroc_stereo': {'selected': True, 'window_size': 1},
        'euroc_vo_precomputed': {
            'selected': True,
            'args': {
                'estimation_type': '2d3d',
                'use_precomputed': True,
                'vo_suffix': '_2d3d'
            }
        }
    }
    
    result = check_and_configure_vo(
        sensors=euroc_sensors,
        root_path='./data/EuRoC',
        variant='01',
        dataset_type='euroc'
    )
    print(f"Result: {result}")
    
    # Example 2: KITTI dataset with runtime VO
    kitti_sensors = {
        'kitti_stereo': {'selected': True, 'window_size': 1},
        'kitti_vo_precomputed': {
            'selected': True,
            'args': {
                'estimation_type': '2d3d',
                'use_precomputed': False  # Force runtime computation
            }
        }
    }
    
    result = check_and_configure_vo(
        sensors=kitti_sensors,
        root_path='./data/KITTI',
        variant='09',
        dataset_type='kitti'
    )
    print(f"Result: {result}")
