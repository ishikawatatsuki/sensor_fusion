"""
Backward Compatibility Layer
=============================

Provides compatibility between the old SensorType enum system and the new
SensorMetadata system. This allows gradual migration of existing code.

Usage:
    # Convert old sensor type to new metadata
    old_type = KITTI_SensorType.OXTS_IMU
    metadata = legacy_sensor_type_to_metadata(old_type, "KITTI")
    
    # Get old-style checks with new metadata
    if is_imu_data_legacy(metadata):
        # Handle IMU data
        pass
"""

from typing import Union, Optional
from enum import IntEnum

from .sensor_metadata import SensorCategory, SensorMetadata
from .sensor_config_loader import create_sensor_metadata_from_config


def legacy_sensor_type_to_metadata(
    sensor_type: IntEnum,
    dataset: str,
    dropout_ratio: float = 0.0,
    window_size: int = 1
) -> SensorMetadata:
    """
    Convert legacy SensorType enum to SensorMetadata.
    
    Args:
        sensor_type: Old IntEnum sensor type (KITTI_SensorType, etc.)
        dataset: Dataset name
        dropout_ratio: Dropout ratio
        window_size: Window size
    
    Returns:
        SensorMetadata object
    """
    # Get sensor ID from enum name
    sensor_id = sensor_type.name
    
    try:
        metadata = create_sensor_metadata_from_config(
            dataset=dataset,
            sensor_id=sensor_id,
            dropout_ratio=dropout_ratio,
            window_size=window_size
        )
        metadata.legacy_sensor_type = sensor_type
        return metadata
    except (ValueError, KeyError) as e:
        raise ValueError(
            f"Could not convert legacy sensor type {sensor_type.name} "
            f"from dataset {dataset}: {e}"
        )


def metadata_to_legacy_priority(metadata: SensorMetadata) -> int:
    """
    Convert metadata to legacy-style integer priority (for PriorityQueue).
    This was previously the IntEnum value.
    
    Args:
        metadata: SensorMetadata object
    
    Returns:
        Integer priority value
    """
    return metadata.priority


# =============================================================================
# Legacy-style sensor type checking functions
# These mimic the static methods from the old SensorType class
# =============================================================================

def is_imu_data_legacy(metadata: Union[SensorMetadata, SensorCategory]) -> bool:
    """Check if sensor is IMU data (legacy compatibility)"""
    if isinstance(metadata, SensorMetadata):
        return metadata.category == SensorCategory.IMU
    return metadata == SensorCategory.IMU


def is_motor_data_legacy(metadata: Union[SensorMetadata, SensorCategory]) -> bool:
    """Check if sensor is motor/actuator data (legacy compatibility)"""
    if isinstance(metadata, SensorMetadata):
        return metadata.category == SensorCategory.ACTUATOR
    return metadata == SensorCategory.ACTUATOR


def is_time_update_legacy(metadata: Union[SensorMetadata, SensorCategory]) -> bool:
    """Check if sensor is used for time updates (legacy compatibility)"""
    if isinstance(metadata, SensorMetadata):
        return metadata.is_time_update()
    return metadata.is_time_update() if hasattr(metadata, 'is_time_update') else False


def is_gps_data_legacy(metadata: Union[SensorMetadata, SensorCategory]) -> bool:
    """Check if sensor is GPS data (legacy compatibility)"""
    if isinstance(metadata, SensorMetadata):
        return metadata.category == SensorCategory.GPS
    return metadata == SensorCategory.GPS


def is_leica_data_legacy(metadata: Union[SensorMetadata, SensorCategory]) -> bool:
    """Check if sensor is Leica data (legacy compatibility)"""
    if isinstance(metadata, SensorMetadata):
        return metadata.category == SensorCategory.LEICA
    return metadata == SensorCategory.LEICA


def is_positioning_data_legacy(metadata: Union[SensorMetadata, SensorCategory]) -> bool:
    """Check if sensor provides positioning data (legacy compatibility)"""
    if isinstance(metadata, SensorMetadata):
        return metadata.category.is_positioning_data()
    return metadata.is_positioning_data() if hasattr(metadata, 'is_positioning_data') else False


def is_stereo_image_data_legacy(metadata: Union[SensorMetadata, SensorCategory]) -> bool:
    """Check if sensor is stereo camera (legacy compatibility)"""
    if isinstance(metadata, SensorMetadata):
        return metadata.category == SensorCategory.STEREO_CAMERA
    return metadata == SensorCategory.STEREO_CAMERA


def is_camera_image_data_legacy(metadata: Union[SensorMetadata, SensorCategory]) -> bool:
    """Check if sensor is any camera (legacy compatibility)"""
    if isinstance(metadata, SensorMetadata):
        return metadata.category.is_camera_data()
    return metadata.is_camera_data() if hasattr(metadata, 'is_camera_data') else False


def is_visualization_data_legacy(metadata: Union[SensorMetadata, SensorCategory]) -> bool:
    """Check if sensor is visualization data (legacy compatibility)"""
    if isinstance(metadata, SensorMetadata):
        return metadata.category.is_visualization_data()
    return metadata.is_visualization_data() if hasattr(metadata, 'is_visualization_data') else False


def is_vo_data_legacy(metadata: Union[SensorMetadata, SensorCategory]) -> bool:
    """Check if sensor is visual odometry (legacy compatibility)"""
    if isinstance(metadata, SensorMetadata):
        return metadata.category == SensorCategory.VISUAL_ODOMETRY
    return metadata == SensorCategory.VISUAL_ODOMETRY


def is_motor_output_legacy(metadata: Union[SensorMetadata, SensorCategory]) -> bool:
    """Check if sensor is motor output (legacy compatibility)"""
    if isinstance(metadata, SensorMetadata):
        return metadata.category == SensorCategory.ACTUATOR
    return metadata == SensorCategory.ACTUATOR


def is_magnetometer_data_legacy(metadata: Union[SensorMetadata, SensorCategory]) -> bool:
    """Check if sensor is magnetometer (legacy compatibility)"""
    if isinstance(metadata, SensorMetadata):
        return metadata.category == SensorCategory.MAGNETOMETER
    return metadata == SensorCategory.MAGNETOMETER


def is_constraint_data_legacy(metadata: Union[SensorMetadata, SensorCategory]) -> bool:
    """Check if sensor is constraint data (legacy compatibility)"""
    if isinstance(metadata, SensorMetadata):
        return metadata.category == SensorCategory.VELOCITY_CONSTRAINT
    return metadata == SensorCategory.VELOCITY_CONSTRAINT


def is_reference_data_legacy(metadata: Union[SensorMetadata, SensorCategory]) -> bool:
    """Check if sensor is reference/ground truth (legacy compatibility)"""
    if isinstance(metadata, SensorMetadata):
        return metadata.category.is_reference_data()
    return metadata.is_reference_data() if hasattr(metadata, 'is_reference_data') else False


def is_measurement_update_legacy(metadata: Union[SensorMetadata, SensorCategory]) -> bool:
    """Check if sensor provides measurement updates (legacy compatibility)"""
    if isinstance(metadata, SensorMetadata):
        return metadata.is_measurement_update()
    return metadata.is_measurement_update() if hasattr(metadata, 'is_measurement_update') else False


# =============================================================================
# Mapping utilities
# =============================================================================

def get_sensor_name_legacy(metadata: SensorMetadata) -> str:
    """
    Get sensor name in legacy format.
    
    Args:
        metadata: SensorMetadata object
    
    Returns:
        Sensor name string
    """
    return metadata.sensor_id


def get_dataset_from_metadata(metadata: SensorMetadata) -> str:
    """
    Get dataset name from metadata.
    
    Args:
        metadata: SensorMetadata object
    
    Returns:
        Dataset name
    """
    return metadata.dataset


# =============================================================================
# Helper class for gradual migration
# =============================================================================

class LegacySensorTypeAdapter:
    """
    Adapter class that makes SensorMetadata behave like the old SensorType enum.
    This allows using new metadata with old code patterns.
    
    Usage:
        metadata = create_sensor_metadata_from_config("KITTI", "OXTS_IMU")
        adapter = LegacySensorTypeAdapter(metadata)
        
        # Use like old SensorType
        if SensorType.is_imu_data(adapter):
            pass
    """
    
    def __init__(self, metadata: SensorMetadata):
        self.metadata = metadata
        self._metadata = metadata
    
    @property
    def name(self) -> str:
        """Get sensor name (mimics enum .name)"""
        return self.metadata.sensor_id
    
    @property
    def value(self) -> int:
        """Get sensor value (mimics enum .value)"""
        return self.metadata.priority
    
    def __eq__(self, other):
        """Equality comparison"""
        if isinstance(other, LegacySensorTypeAdapter):
            return self.metadata.full_name == other.metadata.full_name
        if isinstance(other, SensorMetadata):
            return self.metadata.full_name == other.full_name
        return False
    
    def __hash__(self):
        """Hash for use in sets/dicts"""
        return hash(self.metadata.full_name)
    
    def __repr__(self):
        """String representation"""
        return f"<{self.metadata.dataset}.{self.metadata.sensor_id}: {self.metadata.category.name}>"
