"""
Refactored Sensor System
========================

This module provides a generalized sensor metadata system that decouples
sensor categories from dataset-specific implementations. This allows:
- Easy addition of new datasets without modifying core enums
- Support for multiple instances of the same sensor type (e.g., IMU0, IMU1)
- Configuration-driven sensor definitions
- Clear separation between sensor category and sensor instance
"""

from enum import Enum, auto
from dataclasses import dataclass, field
import logging
from typing import Optional, Dict, Any, Callable


class SensorCategory(Enum):
    """
    Generic sensor categories that apply across all datasets.
    This replaces dataset-specific sensor type enums.
    """
    # Time update sensors (propagation step)
    IMU = auto()
    ACTUATOR = auto()
    ROTOR_SPEED = auto()
    
    # Measurement update sensors
    GPS = auto()
    MAGNETOMETER = auto()
    VISUAL_ODOMETRY = auto()
    POSITIONING_SENSOR = auto()  # High-precision positioning
    WHEEL_ODOMETRY = auto()
    LIDAR_ODOMETRY = auto()
    
    # Image sensors
    STEREO_CAMERA = auto()
    MONO_CAMERA = auto()
    
    # Velocity constraints
    VELOCITY_CONSTRAINT = auto()
    
    # Reference/visualization
    GROUND_TRUTH = auto()
    VISUALIZATION = auto()
    
    # Other
    LIDAR = auto()
    RADAR = auto()
    RANGE_FINDER = auto()
    
    def is_time_update(self) -> bool:
        """Check if this sensor is used for time update (propagation)"""
        return self in {SensorCategory.IMU, SensorCategory.ROTOR_SPEED, SensorCategory.ACTUATOR}
    
    def is_measurement_update(self) -> bool:
        """Check if this sensor provides measurement updates"""
        return self in {
            SensorCategory.GPS,
            SensorCategory.VISUAL_ODOMETRY,
            SensorCategory.MAGNETOMETER,
            SensorCategory.POSITIONING_SENSOR,
            SensorCategory.VELOCITY_CONSTRAINT,
            SensorCategory.WHEEL_ODOMETRY,
            SensorCategory.LIDAR_ODOMETRY,
            SensorCategory.LIDAR,
            SensorCategory.RADAR,
            SensorCategory.RANGE_FINDER
        }
    
    def is_positioning_data(self) -> bool:
        """Check if this sensor provides absolute positioning"""
        return self in {SensorCategory.GPS, SensorCategory.POSITIONING_SENSOR}
    
    def is_camera_data(self) -> bool:
        """Check if this sensor is a camera"""
        return self in {
            SensorCategory.STEREO_CAMERA,
            SensorCategory.MONO_CAMERA
        }
    
    def is_reference_data(self) -> bool:
        """Check if this is reference/ground truth data"""
        return self == SensorCategory.GROUND_TRUTH
    
    def is_visualization_data(self) -> bool:
        """Check if this is visualization-only data"""
        return self == SensorCategory.VISUALIZATION


@dataclass
class SensorMetadata:
    """
    Metadata for a specific sensor instance.
    
    This contains all information needed to identify, configure, and process
    data from a particular sensor, independent of the dataset it comes from.
    
    Attributes:
        category: The generic category of this sensor
        dataset: Dataset name (e.g., "KITTI", "EuRoC", "UAV")
        sensor_id: Unique identifier within the dataset (e.g., "IMU0", "OXTS_IMU")
        frequency: Sensor sampling frequency in Hz
        priority: Priority for ordering in queue (lower = higher priority)
        coordinate_frame: The coordinate frame this sensor measures in
        dropout_ratio: Probability of dropping sensor readings (for testing)
        window_size: Number of readings to buffer
        noise_profile: Optional noise characteristics
        reader_config: Configuration dict for the data reader
    """
    category: SensorCategory
    dataset: str
    sensor_id: str
    frequency: float
    priority: int
    coordinate_frame: 'CoordinateFrame'
    
    # Optional attributes
    dropout_ratio: float = 0.0
    window_size: int = 1
    noise_profile: Optional[Dict[str, Any]] = None
    reader_config: Optional[Dict[str, Any]] = field(default_factory=dict)
    
    # Legacy support
    legacy_sensor_type: Optional[Any] = None  # For backward compatibility
    
    def __post_init__(self):
        """Validate metadata after initialization"""
        assert 0.0 <= self.dropout_ratio <= 1.0, \
            f"Dropout ratio must be in [0, 1], got {self.dropout_ratio}"
        assert self.frequency > 0, \
            f"Frequency must be positive, got {self.frequency}"
        assert self.window_size > 0, \
            f"Window size must be positive, got {self.window_size}"
    
    @property
    def full_name(self) -> str:
        """Get the full unique name for this sensor"""
        return f"{self.dataset}_{self.sensor_id}"
    
    def __str__(self) -> str:
        return f"SensorMetadata({self.full_name}, {self.category.name}, {self.frequency}Hz)"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def is_time_update(self) -> bool:
        """Check if this sensor is used for time updates"""
        return self.category.is_time_update()
    
    def is_measurement_update(self) -> bool:
        """Check if this sensor provides measurement updates"""
        return self.category.is_measurement_update()


@dataclass
class SensorQueueItem:
    """
    Item for the sensor data priority queue.
    
    This replaces the old system that relied on IntEnum values for ordering.
    Now we have explicit control over ordering based on timestamp and priority.
    
    Ordering:
        1. Primary: timestamp (earlier timestamps first)
        2. Secondary: priority (lower priority value = higher priority)
        3. This ensures deterministic ordering when timestamps are equal
    """
    timestamp: float
    priority: int
    metadata: SensorMetadata
    data: Any
    
    def __lt__(self, other):
        """Custom comparison for priority queue ordering"""
        if not isinstance(other, SensorQueueItem):
            return NotImplemented
        
        # Primary sort by timestamp
        if abs(self.timestamp - other.timestamp) > 1e-9:  # Handle floating point
            return self.timestamp < other.timestamp
        
        # Secondary sort by priority (lower value = higher priority)
        return self.priority < other.priority
    
    def __le__(self, other):
        """Less than or equal comparison"""
        return self == other or self < other
    
    def __gt__(self, other):
        """Greater than comparison"""
        if not isinstance(other, SensorQueueItem):
            return NotImplemented
        return not (self <= other)
    
    def __ge__(self, other):
        """Greater than or equal comparison"""
        if not isinstance(other, SensorQueueItem):
            return NotImplemented
        return not (self < other)
    
    def __eq__(self, other):
        """Equality comparison"""
        if not isinstance(other, SensorQueueItem):
            return NotImplemented
        return (abs(self.timestamp - other.timestamp) < 1e-9 and 
                self.priority == other.priority)
    
    @property
    def sensor_type(self):
        """For backward compatibility - returns category"""
        return self.metadata.category
    
    @property
    def sensor_name(self) -> str:
        """Get the full sensor name"""
        return self.metadata.full_name


class SensorRegistry:
    """
    Registry for sensor data readers and processors.
    
    This allows dynamic registration of sensor readers and processors,
    decoupling sensor types from their implementations.
    
    Usage:
        registry = SensorRegistry()
        
        # Register readers
        registry.register_reader("KITTI", "OXTS_IMU", OXTS_IMUDataReader)
        registry.register_reader("UAV", "VOXL_IMU0", VOXL_IMUDataReader)
        
        # Register processors
        registry.register_processor(SensorCategory.IMU, process_imu_data)
        
        # Get reader/processor
        reader_class = registry.get_reader_class("KITTI", "OXTS_IMU")
        processor = registry.get_processor(SensorCategory.IMU)
    """
    
    def __init__(self):
        self._readers: Dict[str, type] = {}
        self._processors: Dict[SensorCategory, Callable] = {}
        self._metadata_cache: Dict[str, SensorMetadata] = {}
    
    def register_reader(
        self,
        dataset: str,
        sensor_id: str,
        reader_class: type
    ) -> None:
        """
        Register a data reader class for a specific sensor.
        
        Args:
            dataset: Dataset name (e.g., "KITTI")
            sensor_id: Sensor identifier (e.g., "OXTS_IMU")
            reader_class: Class that reads data for this sensor
        """
        key = f"{dataset}_{sensor_id}"
        if key in self._readers:
            logging.warning(f"Key {key} already registered.")
            return
        self._readers[key] = reader_class
    
    def register_processor(
        self,
        category: SensorCategory,
        processor_func: Callable
    ) -> None:
        """
        Register a processing function for a sensor category.
        
        Args:
            category: Sensor category
            processor_func: Function that processes data for this category
        """
        if category in self._processors:
            raise ValueError(f"Processor for {category} already registered")
        self._processors[category] = processor_func
    
    def get_reader_class(self, dataset: str, sensor_id: str) -> Optional[type]:
        """Get the reader class for a sensor"""
        key = f"{dataset}_{sensor_id}"
        return self._readers.get(key)
    
    def get_reader_class_from_metadata(self, metadata: SensorMetadata) -> Optional[type]:
        """Get the reader class from sensor metadata"""
        return self.get_reader_class(metadata.dataset, metadata.sensor_id)
    
    def get_processor(self, category: SensorCategory) -> Optional[Callable]:
        """Get the processor function for a sensor category"""
        return self._processors.get(category)
    
    def cache_metadata(self, metadata: SensorMetadata) -> None:
        """Cache sensor metadata for quick lookup"""
        self._metadata_cache[metadata.full_name] = metadata
    
    def get_metadata(self, full_name: str) -> Optional[SensorMetadata]:
        """Retrieve cached metadata"""
        return self._metadata_cache.get(full_name)
    
    def list_readers(self) -> list:
        """List all registered readers"""
        return list(self._readers.keys())
    
    def list_processors(self) -> list:
        """List all registered processors"""
        return [cat.name for cat in self._processors.keys()]


# Global registry instance
_global_registry = None


def get_sensor_registry() -> SensorRegistry:
    """Get the global sensor registry instance"""
    global _global_registry
    if _global_registry is None:
        _global_registry = SensorRegistry()
    return _global_registry


def reset_sensor_registry() -> None:
    """Reset the global registry (useful for testing)"""
    global _global_registry
    _global_registry = None
