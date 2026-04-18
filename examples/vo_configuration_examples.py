"""
Visual Odometry Configuration Examples

This script demonstrates how to use the flexible VO configuration system.
"""

import os
import sys
import logging

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.vo_config_helper import (
    VOConfigHelper,
    check_and_configure_vo
)
from src.internal.extended_common.extended_config import ExtendedConfig


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)


def example_1_check_vo_availability():
    """Example 1: Check if pre-computed VO data exists."""
    print("\n" + "="*70)
    print("Example 1: Checking VO Data Availability")
    print("="*70)
    
    # EuRoC dataset
    exists, path = VOConfigHelper.check_vo_data_exists(
        root_path='./data/EuRoC',
        variant='01',
        dataset_type='euroc',
        estimation_type='2d3d'
    )
    
    if exists:
        print(f"✓ Pre-computed VO found: {path}")
    else:
        print(f"✗ Pre-computed VO not found at: {path}")
        print("  → Will use runtime computation")
    
    # KITTI dataset
    exists, path = VOConfigHelper.check_vo_data_exists(
        root_path='./data/KITTI',
        variant='09',
        dataset_type='kitti',
        estimation_type='2d3d'
    )
    
    if exists:
        print(f"✓ Pre-computed VO found: {path}")
    else:
        print(f"✗ Pre-computed VO not found at: {path}")
        print("  → Will use runtime computation")


def example_2_determine_vo_mode():
    """Example 2: Determine VO mode from sensor configuration."""
    print("\n" + "="*70)
    print("Example 2: Determining VO Mode")
    print("="*70)
    
    # Configuration with pre-computed VO
    sensor_config = {
        'args': {
            'use_precomputed': True,
            'estimation_type': '2d3d',
            'vo_suffix': '_2d3d'
        }
    }
    
    vo_mode_info = VOConfigHelper.determine_vo_mode(
        sensor_config=sensor_config,
        root_path='./data/EuRoC',
        variant='01',
        dataset_type='euroc'
    )
    
    print(f"VO Mode: {vo_mode_info['mode']}")
    print(f"VO Path: {vo_mode_info['path']}")
    print(f"Estimation Type: {vo_mode_info['estimation_type']}")
    print(f"Requires Camera: {vo_mode_info['requires_camera']}")


def example_3_validate_camera():
    """Example 3: Validate camera sensor availability."""
    print("\n" + "="*70)
    print("Example 3: Validating Camera Availability")
    print("="*70)
    
    # Configuration with camera
    sensors_with_camera = {
        'euroc_stereo': {'selected': True},
        'euroc_vo_precomputed': {'selected': True}
    }
    
    is_valid = VOConfigHelper.validate_camera_availability(
        sensors=sensors_with_camera,
        dataset_type='euroc',
        vo_mode='runtime'
    )
    
    print(f"Camera available for runtime VO: {is_valid}")
    
    # Configuration without camera
    sensors_without_camera = {
        'euroc_imu': {'selected': True},
        'euroc_vo_precomputed': {'selected': True}
    }
    
    is_valid = VOConfigHelper.validate_camera_availability(
        sensors=sensors_without_camera,
        dataset_type='euroc',
        vo_mode='runtime'
    )
    
    print(f"Camera available for runtime VO: {is_valid}")


def example_4_full_vo_configuration():
    """Example 4: Complete VO configuration check."""
    print("\n" + "="*70)
    print("Example 4: Full VO Configuration")
    print("="*70)
    
    sensors = {
        'euroc_stereo': {
            'selected': True,
            'dropout_ratio': 0,
            'window_size': 1,
            'args': {'frequency': 30}
        },
        'euroc_vo_precomputed': {
            'selected': True,
            'dropout_ratio': 0,
            'window_size': 1,
            'args': {
                'frequency': 30,
                'vo_suffix': '_2d3d',
                'estimation_type': '2d3d',
                'use_precomputed': True
            }
        }
    }
    
    result = check_and_configure_vo(
        sensors=sensors,
        root_path='./data/EuRoC',
        variant='01',
        dataset_type='euroc'
    )
    
    print(f"\nConfiguration Result:")
    print(f"  VO Sensor: {result['vo_sensor']}")
    print(f"  VO Mode: {result['vo_mode']}")
    print(f"  VO Path: {result['vo_path']}")
    print(f"  Camera Available: {result['camera_available']}")
    print(f"  Estimation Type: {result['estimation_type']}")


def example_5_load_config_file():
    """Example 5: Load configuration from YAML file."""
    print("\n" + "="*70)
    print("Example 5: Loading Configuration from File")
    print("="*70)
    
    config_path = './configs/euroc_config.yaml'
    
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        return
    
    try:
        config = ExtendedConfig(config_filepath=config_path)
        
        print(f"Dataset Type: {config.dataset.type}")
        print(f"Dataset Variant: {config.dataset.variant}")
        print(f"Run Visual Odometry: {config.dataset.run_visual_odometry}")
        print(f"VO Mode: {config.dataset.vo_mode}")
        print(f"VO Estimation Type: {config.dataset.vo_estimation_type}")
        
        if config.dataset.use_precomputed_vo:
            print(f"Using pre-computed VO from: {config.dataset.vo_path}")
        elif config.dataset.use_runtime_vo:
            print(f"Will compute VO at runtime using: {config.dataset.vo_estimation_type}")
        else:
            print("Visual odometry not configured")
            
    except Exception as e:
        print(f"Error loading config: {e}")
        import traceback
        traceback.print_exc()


def example_6_get_vo_path():
    """Example 6: Construct VO paths for different configurations."""
    print("\n" + "="*70)
    print("Example 6: Constructing VO Paths")
    print("="*70)
    
    configs = [
        ('euroc', '01', '2d2d', None),
        ('euroc', '02', '2d3d', None),
        ('euroc', '03', 'hybrid', None),
        ('kitti', '09', '2d3d', '_2d3d'),
        ('kitti', '10', 'hybrid', '_hybrid'),
    ]
    
    for dataset_type, variant, estimation_type, vo_suffix in configs:
        root_path = f'./data/{dataset_type.upper()}'
        
        path = VOConfigHelper.get_vo_path(
            root_path=root_path,
            variant=variant,
            dataset_type=dataset_type,
            estimation_type=estimation_type,
            vo_suffix=vo_suffix
        )
        
        exists = os.path.exists(path)
        status = "✓" if exists else "✗"
        
        print(f"{status} {dataset_type.upper()}-{variant} ({estimation_type}): {path}")


def example_7_compare_estimation_types():
    """Example 7: Compare different VO estimation types."""
    print("\n" + "="*70)
    print("Example 7: Comparing VO Estimation Types")
    print("="*70)
    
    estimation_types = ['2d2d', '2d3d', 'hybrid', 'stereo']
    
    print("\nEstimation Type Comparison:")
    print("-" * 70)
    print(f"{'Type':<12} {'Speed':<15} {'Accuracy':<15} {'Suffix':<15}")
    print("-" * 70)
    
    characteristics = {
        '2d2d': ('Fast', 'Good', ''),
        '2d3d': ('Moderate', 'Very Good', '_2d3d'),
        'hybrid': ('Slow', 'Excellent', '_hybrid'),
        'stereo': ('Fast', 'Very Good', '_stereo')
    }
    
    for est_type in estimation_types:
        speed, accuracy, suffix = characteristics.get(est_type, ('Unknown', 'Unknown', ''))
        print(f"{est_type:<12} {speed:<15} {accuracy:<15} {suffix:<15}")
    
    print("\nRecommendation: Use '2d3d' for best balance of speed and accuracy")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("Visual Odometry Configuration Examples")
    print("="*70)
    
    try:
        example_1_check_vo_availability()
        example_2_determine_vo_mode()
        example_3_validate_camera()
        example_4_full_vo_configuration()
        example_5_load_config_file()
        example_6_get_vo_path()
        example_7_compare_estimation_types()
        
        print("\n" + "="*70)
        print("All examples completed!")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
