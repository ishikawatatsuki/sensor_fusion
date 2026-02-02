"""
Adapter to integrate generic configuration-driven data readers
with the existing dataset infrastructure.
"""

import os
import yaml
from typing import Dict, Any
from .generic_reader import (
    SensorReaderFactory, 
    TransformRegistry,
    SensorSchema
)


class ConfigurableDatasetAdapter:
    """
    Adapter that bridges configuration-driven generic readers
    with the existing dataset classes (KITTIDataset, EuRoCDataset, etc.)
    """
    
    def __init__(
        self,
        dataset_type: str,
        schema_config_path: str = None
    ):
        """
        Args:
            dataset_type: Type of dataset (kitti, euroc, uav)
            schema_config_path: Path to sensor_schemas.yaml file
        """
        self.dataset_type = dataset_type
        
        # Load schema configuration
        if schema_config_path is None:
            # Default location
            schema_config_path = os.path.join(
                os.path.dirname(__file__),
                '../../../configs/sensor_schemas.yaml'
            )
        
        with open(schema_config_path, 'r') as f:
            self.schema_config = yaml.safe_load(f)
        
        # Register custom transforms
        self._register_custom_transforms()
        
        # Get dataset-specific sensor mapping
        self.sensor_mapping = self.schema_config.get('dataset_sensor_mapping', {}).get(
            dataset_type, {}
        )
    
    def _register_custom_transforms(self):
        """Register custom transformation functions from config"""
        custom_transforms = self.schema_config.get('custom_transforms', {})
        
        for transform_name, transform_spec in custom_transforms.items():
            # Execute the code to define the function
            code = transform_spec.get('code', '')
            local_scope = {}
            exec(code, globals(), local_scope)
            
            # Register the function
            if transform_name in local_scope:
                TransformRegistry.register(transform_name, local_scope[transform_name])
    
    def create_sensor_reader(
        self,
        sensor_config_name: str,
        root_path: str,
        variant: str = None,
        starttime: float = -float('inf'),
        window_size: int = None,
        **kwargs
    ):
        """
        Create a data reader for a specific sensor.
        
        Args:
            sensor_config_name: Name from dataset config (e.g., 'euroc_imu', 'oxts_imu')
            root_path: Root path to dataset
            variant: Dataset variant (e.g., '01' for EuRoC, '0033' for KITTI)
            starttime: Filter data after this timestamp
            window_size: Rolling window size for averaging
            **kwargs: Additional parameters (e.g., date, drive for KITTI)
        
        Returns:
            GenericDataReader instance
        """
        # Map sensor config name to schema name
        schema_name = self.sensor_mapping.get(sensor_config_name)
        
        if schema_name is None:
            raise ValueError(
                f"Sensor '{sensor_config_name}' not found in mapping for dataset '{self.dataset_type}'"
            )
        
        # Get sensor schema
        sensor_config = self.schema_config['sensor_schemas'].get(schema_name)
        
        if sensor_config is None:
            raise ValueError(f"Sensor schema '{schema_name}' not found in config")
        
        # Create reader
        return SensorReaderFactory.create_reader(
            sensor_config=sensor_config,
            root_path=root_path,
            variant=variant,
            starttime=starttime,
            window_size=window_size,
            **kwargs
        )
    
    def get_available_sensors(self):
        """Get list of available sensors for this dataset type"""
        return list(self.sensor_mapping.keys())


def create_data_reader_from_config(
    dataset_config: Dict[str, Any],
    sensor_name: str,
    **override_params
):
    """
    Convenience function to create a data reader from dataset configuration.
    
    Args:
        dataset_config: Dataset configuration dict (from YAML config)
        sensor_name: Name of sensor (e.g., 'euroc_imu', 'oxts_imu')
        **override_params: Override parameters (starttime, window_size, etc.)
    
    Returns:
        GenericDataReader instance
    
    Example:
        >>> dataset_config = {
        ...     'type': 'euroc',
        ...     'root_path': './data/EuRoC',
        ...     'variant': '01'
        ... }
        >>> reader = create_data_reader_from_config(
        ...     dataset_config,
        ...     'euroc_imu',
        ...     window_size=5
        ... )
    """
    dataset_type = dataset_config['type']
    root_path = dataset_config['root_path']
    variant = dataset_config.get('variant')
    
    # Get sensor-specific config
    sensor_config = dataset_config.get('sensors', {}).get(sensor_name, {})
    
    # Merge parameters
    params = {
        'root_path': root_path,
        'variant': variant,
        'starttime': sensor_config.get('starttime', -float('inf')),
        'window_size': sensor_config.get('window_size'),
    }
    params.update(sensor_config.get('args', {}))
    params.update(override_params)
    
    # Create adapter and reader
    adapter = ConfigurableDatasetAdapter(dataset_type)
    return adapter.create_sensor_reader(sensor_name, **params)


# ============================================================================
# Example: Drop-in replacements for existing reader classes
# ============================================================================

class ConfigurableEuRoCIMUReader:
    """
    Drop-in replacement for EuRoC_IMUDataReader that uses config-driven approach.
    Maintains same interface for backward compatibility.
    """
    def __init__(self, root_path: str, starttime=-float('inf'), window_size=None):
        adapter = ConfigurableDatasetAdapter('euroc')
        
        # Extract variant from root_path (e.g., MH_01_easy -> 01)
        variant = self._extract_variant(root_path)
        
        self._reader = adapter.create_sensor_reader(
            'euroc_imu',
            root_path=os.path.dirname(root_path) if 'MH_' in root_path else root_path,
            variant=variant,
            starttime=starttime,
            window_size=window_size
        )
    
    def _extract_variant(self, root_path: str) -> str:
        """Extract variant from path like 'MH_01_easy'"""
        import re
        match = re.search(r'MH_(\d+)_', root_path)
        if match:
            return match.group(1)
        return '01'  # default
    
    def __iter__(self):
        return iter(self._reader)
    
    def start_time(self):
        return self._reader.start_time()
    
    def set_starttime(self, starttime):
        self._reader.set_starttime(starttime)


class ConfigurableKITTIOXTSIMUReader:
    """
    Drop-in replacement for OXTS_IMUDataReader using config-driven approach.
    """
    def __init__(self, root_path, date, drive, starttime=-float('inf')):
        adapter = ConfigurableDatasetAdapter('kitti')
        
        self._reader = adapter.create_sensor_reader(
            'oxts_imu',
            root_path=root_path,
            variant=drive,
            starttime=starttime,
            date=date,
            drive=drive
        )
    
    def __iter__(self):
        return iter(self._reader)
    
    def start_time(self):
        return self._reader.start_time()
    
    def set_starttime(self, starttime):
        self._reader.set_starttime(starttime)
    
    @staticmethod
    def get_noise_vector():
        # Can be retrieved from schema config
        return 8.333333333333333e-05, 5.817764173314432e-05
