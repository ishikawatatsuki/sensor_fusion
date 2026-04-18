"""
Sensor Registry Loader and Configuration
=========================================

Loads sensor configurations from YAML and populates the sensor registry.
Provides utilities for creating SensorMetadata from configuration.
"""

import os
import yaml
from typing import Dict, List, Optional
from pathlib import Path

from .sensor_metadata import (
    SensorCategory,
    SensorMetadata,
    SensorRegistry,
    get_sensor_registry
)


def load_sensor_registry_config(config_path: Optional[str] = None) -> Dict:
    """
    Load sensor registry configuration from YAML file.
    
    Args:
        config_path: Path to sensor registry YAML file.
                    If None, uses default path.
    
    Returns:
        Dictionary containing the configuration
    """
    if config_path is None:
        # Default to configs/sensor_registry.yaml
        current_dir = Path(__file__).parent
        config_path = current_dir.parent.parent / "configs" / "sensor_registry.yaml"
    
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Sensor registry config not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def populate_registry_from_config(
    config: Optional[Dict] = None,
    config_path: Optional[str] = None,
    registry: Optional[SensorRegistry] = None
) -> SensorRegistry:
    """
    Populate sensor registry from configuration.
    
    Args:
        config: Configuration dictionary (if already loaded)
        config_path: Path to config file (if not loaded)
        registry: SensorRegistry to populate (creates new if None)
    
    Returns:
        Populated SensorRegistry
    """
    if config is None:
        config = load_sensor_registry_config(config_path)
    
    if registry is None:
        registry = get_sensor_registry()
    
    # We don't auto-register readers here because they need to be imported
    # This is just for metadata. Readers are registered explicitly.
    
    return registry


def create_sensor_metadata_from_config(
    dataset: str,
    sensor_id: str,
    config: Optional[Dict] = None,
    config_path: Optional[str] = None,
    dropout_ratio: float = 0.0,
    window_size: int = 1
) -> SensorMetadata:
    """
    Create SensorMetadata from registry configuration.
    
    Args:
        dataset: Dataset name (e.g., "KITTI")
        sensor_id: Sensor ID (e.g., "OXTS_IMU")
        config: Pre-loaded configuration dict
        config_path: Path to config file
        dropout_ratio: Override dropout ratio
        window_size: Override window size
    
    Returns:
        SensorMetadata object
    """
    if config is None:
        config = load_sensor_registry_config(config_path)
    
    # Find sensor in config
    dataset_config = config.get('datasets', {}).get(dataset)
    if dataset_config is None:
        raise ValueError(f"Dataset '{dataset}' not found in sensor registry")
    
    sensor_config = None
    for sensor in dataset_config.get('sensors', []):
        if sensor['id'] == sensor_id:
            sensor_config = sensor
            break
    
    if sensor_config is None:
        raise ValueError(f"Sensor '{sensor_id}' not found in dataset '{dataset}'")
    
    # Import CoordinateFrame from datatypes
    from .datatypes import CoordinateFrame
    
    # Parse category
    category_str = sensor_config['category']
    try:
        category = SensorCategory[category_str.upper().replace(' ', '_')]
    except KeyError:
        raise ValueError(f"Unknown sensor category: {category_str}")
    
    # Parse coordinate frame
    frame_str = sensor_config['coordinate_frame']
    try:
        coordinate_frame = CoordinateFrame[frame_str.upper()]
    except KeyError:
        raise ValueError(f"Unknown coordinate frame: {frame_str}")
    
    # Create metadata
    metadata = SensorMetadata(
        category=category,
        dataset=dataset,
        sensor_id=sensor_id,
        frequency=sensor_config['frequency'],
        priority=sensor_config['priority'],
        coordinate_frame=coordinate_frame,
        dropout_ratio=dropout_ratio,
        window_size=window_size,
        noise_profile=sensor_config.get('noise_profile'),
        reader_config={
            'reader_class': sensor_config.get('reader_class'),
        }
    )
    
    return metadata


def get_all_dataset_sensors(
    dataset: str,
    config: Optional[Dict] = None,
    config_path: Optional[str] = None
) -> List[str]:
    """
    Get list of all sensor IDs for a dataset.
    
    Args:
        dataset: Dataset name
        config: Pre-loaded configuration
        config_path: Path to config file
    
    Returns:
        List of sensor IDs
    """
    if config is None:
        config = load_sensor_registry_config(config_path)
    
    dataset_config = config.get('datasets', {}).get(dataset, {})
    sensors = dataset_config.get('sensors', [])
    
    return [s['id'] for s in sensors]


def get_reader_class_name(
    dataset: str,
    sensor_id: str,
    config: Optional[Dict] = None
) -> Optional[str]:
    """
    Get the reader class name for a sensor.
    
    Args:
        dataset: Dataset name
        sensor_id: Sensor ID
        config: Pre-loaded configuration
    
    Returns:
        Reader class name or None
    """
    if config is None:
        config = load_sensor_registry_config()
    
    dataset_config = config.get('datasets', {}).get(dataset, {})
    for sensor in dataset_config.get('sensors', []):
        if sensor['id'] == sensor_id:
            return sensor.get('reader_class')
    
    return None
