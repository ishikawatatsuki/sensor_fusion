"""
Refactored Sensor System - Usage Examples
==========================================

This file demonstrates how to use the new sensor metadata system.
"""

import numpy as np
from queue import PriorityQueue

# =============================================================================
# Example 1: Creating sensor metadata from configuration
# =============================================================================

def example_create_metadata_from_config():
    """Show how to create sensor metadata from YAML config"""
    from src.common.sensor_config_loader import create_sensor_metadata_from_config
    
    # Create metadata for KITTI IMU
    metadata = create_sensor_metadata_from_config(
        dataset="KITTI",
        sensor_id="OXTS_IMU",
        dropout_ratio=0.1,  # Optional override
        window_size=1
    )
    
    print(f"Created metadata: {metadata}")
    print(f"Category: {metadata.category}")
    print(f"Frequency: {metadata.frequency}Hz")
    print(f"Priority: {metadata.priority}")
    print(f"Is time update: {metadata.is_time_update()}")
    
    return metadata


# =============================================================================
# Example 2: Using the sensor registry
# =============================================================================

def example_use_registry():
    """Show how to use the sensor registry"""
    from src.common.sensor_config_loader import get_sensor_registry, create_sensor_metadata_from_config
    from src.internal.dataset.refactored_sensor import register_all_readers
    
    # Get global registry and populate it
    registry = get_sensor_registry()
    register_all_readers(registry)
    
    # Create metadata and get reader
    metadata = create_sensor_metadata_from_config("KITTI", "OXTS_IMU")
    reader_class = registry.get_reader_class_from_metadata(metadata)
    
    print(f"Reader class for {metadata.full_name}: {reader_class}")
    print(f"All registered readers: {registry.list_readers()}")
    
    return registry


# =============================================================================
# Example 3: Using SensorQueueItem for priority queue
# =============================================================================

def example_priority_queue():
    """Show how the new priority queue system works"""
    from src.common.sensor_metadata import (
        SensorQueueItem,
        SensorCategory
    )
    from src.common.sensor_config_loader import create_sensor_metadata_from_config
    
    # Create different sensor metadata
    imu_metadata = create_sensor_metadata_from_config("KITTI", "OXTS_IMU")
    gps_metadata = create_sensor_metadata_from_config("KITTI", "OXTS_GPS")
    camera_metadata = create_sensor_metadata_from_config("KITTI", "KITTI_STEREO")
    
    # Create queue items
    queue = PriorityQueue()
    
    # Add items at different timestamps
    queue.put(SensorQueueItem(
        timestamp=1.0,
        priority=imu_metadata.priority,
        metadata=imu_metadata,
        data={'a': np.array([0, 0, 9.81]), 'w': np.array([0, 0, 0])}
    ))
    
    queue.put(SensorQueueItem(
        timestamp=1.0,  # Same timestamp as IMU
        priority=gps_metadata.priority,
        metadata=gps_metadata,
        data={'lat': 51.0, 'lon': 0.0, 'alt': 100.0}
    ))
    
    queue.put(SensorQueueItem(
        timestamp=0.9,  # Earlier timestamp
        priority=camera_metadata.priority,
        metadata=camera_metadata,
        data={'left': 'frame_001.png', 'right': 'frame_001.png'}
    ))
    
    # Items come out ordered by timestamp, then priority
    print("\nQueue ordering:")
    while not queue.empty():
        item = queue.get()
        print(f"  t={item.timestamp:.1f}, prio={item.priority}, "
              f"sensor={item.metadata.sensor_id}")


# =============================================================================
# Example 4: Backward compatibility with legacy code
# =============================================================================

def example_backward_compatibility():
    """Show how to use new system with legacy code patterns"""
    from src.common.sensor_config_loader import (
        create_sensor_metadata_from_config,
    )
    from src.common.sensor_legacy import (
        is_imu_data_legacy,
        is_time_update_legacy,
        is_measurement_update_legacy,
        LegacySensorTypeAdapter
    )
    
    # Create new metadata
    metadata = create_sensor_metadata_from_config("KITTI", "OXTS_IMU")
    
    # Use with legacy-style functions
    print(f"\nLegacy compatibility:")
    print(f"  is_imu_data: {is_imu_data_legacy(metadata)}")
    print(f"  is_time_update: {is_time_update_legacy(metadata)}")
    print(f"  is_measurement_update: {is_measurement_update_legacy(metadata)}")
    
    # Use adapter for complete compatibility
    adapter = LegacySensorTypeAdapter(metadata)
    print(f"  Adapter name: {adapter.name}")
    print(f"  Adapter value: {adapter.value}")


# =============================================================================
# Example 5: Adding support for a new dataset
# =============================================================================

def example_add_new_dataset():
    """
    Show how to add a new dataset without modifying enums.
    
    Steps:
    1. Add dataset config to configs/sensor_registry.yaml
    2. Create data reader classes
    3. Register readers with registry
    4. Use it!
    """
    
    # Example: Adding a hypothetical "MyDataset" with custom sensors
    
    # Step 1: Add to sensor_registry.yaml
    config_example = """
    datasets:
      MyDataset:
        sensors:
          - id: MY_IMU
            category: IMU
            frequency: 100
            priority: 1
            coordinate_frame: IMU
            reader_class: MyIMUDataReader
          
          - id: MY_LIDAR
            category: LIDAR
            frequency: 10
            priority: 3
            coordinate_frame: INERTIAL
            reader_class: MyLidarDataReader
    """
    
    # Step 2 & 3: Create and register reader (in code)
    """
    class MyIMUDataReader:
        def __init__(self, root_path, **kwargs):
            self.root_path = root_path
        
        def __iter__(self):
            # Yield (timestamp, data) tuples
            yield timestamp, imu_data
    
    # Register it
    from src.common import get_sensor_registry
    registry = get_sensor_registry()
    registry.register_reader("MyDataset", "MY_IMU", MyIMUDataReader)
    """
    
    # Step 4: Use it!
    """
    from src.common import create_sensor_metadata_from_config
    metadata = create_sensor_metadata_from_config("MyDataset", "MY_IMU")
    # Now use this metadata just like any other sensor
    """
    
    print("\nAdding new dataset:")
    print("  1. Add to configs/sensor_registry.yaml")
    print("  2. Create reader classes")
    print("  3. Register readers")
    print("  4. Use with create_sensor_metadata_from_config()")


# =============================================================================
# Example 6: Supporting multiple IMUs with different frequencies
# =============================================================================

def example_multiple_imus():
    """
    Show how to use multiple IMUs at different frequencies.
    
    This was the original motivation for the refactor - the UAV dataset
    has multiple IMUs that could be used together.
    """
    from src.common.sensor_config_loader import create_sensor_metadata_from_config
    
    # Create metadata for two different IMUs
    imu0_metadata = create_sensor_metadata_from_config("UAV", "VOXL_IMU0")
    imu1_metadata = create_sensor_metadata_from_config("UAV", "PX4_IMU0")
    
    print("\nMultiple IMU configuration:")
    print(f"  IMU 0: {imu0_metadata.full_name} @ {imu0_metadata.frequency}Hz")
    print(f"  IMU 1: {imu1_metadata.full_name} @ {imu1_metadata.frequency}Hz")
    print(f"  Both are IMUs: {imu0_metadata.category == imu1_metadata.category}")
    print(f"  Different frequencies: {imu0_metadata.frequency} vs {imu1_metadata.frequency}")
    
    # In processing code, you can now distinguish between them
    print(f"\n  Can distinguish by full_name:")
    print(f"    {imu0_metadata.full_name}")
    print(f"    {imu1_metadata.full_name}")


# =============================================================================
# Example 7: Creating a refactored sensor instance
# =============================================================================

def example_create_sensor():
    """Show how to create and use a RefactoredSensor"""
    import time
    from src.common.sensor_config_loader import create_sensor_metadata_from_config
    from src.internal.dataset.refactored_sensor import RefactoredSensor, register_all_readers
    from src.common.sensor_metadata import get_sensor_registry
    from queue import PriorityQueue
    
    # Setup
    registry = get_sensor_registry()
    register_all_readers(registry)
    
    # You would normally pass proper config here
    # reader = reader_class(root_path="./data/KITTI", ...)
    
    # Create sensor
    output_queue = PriorityQueue()
    
    
    print("\nCreating RefactoredSensor:")
    print("  Create reader instance with proper config")
    print("  Pass to RefactoredSensor constructor")
    print("  Call sensor.start()")


# =============================================================================
# Run all examples
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Refactored Sensor System Examples")
    print("=" * 70)
    
    print("\n" + "=" * 70)
    print("Example 1: Create metadata from config")
    print("=" * 70)
    example_create_metadata_from_config()
    
    print("\n" + "=" * 70)
    print("Example 2: Using the sensor registry")
    print("=" * 70)
    example_use_registry()
    
    print("\n" + "=" * 70)
    print("Example 3: Priority queue with SensorQueueItem")
    print("=" * 70)
    example_priority_queue()
    
    print("\n" + "=" * 70)
    print("Example 4: Backward compatibility")
    print("=" * 70)
    example_backward_compatibility()
    
    print("\n" + "=" * 70)
    print("Example 5: Adding new dataset")
    print("=" * 70)
    example_add_new_dataset()
    
    print("\n" + "=" * 70)
    print("Example 6: Multiple IMUs at different frequencies")
    print("=" * 70)
    example_multiple_imus()
    
    print("\n" + "=" * 70)
    print("Example 7: Creating RefactoredSensor")
    print("=" * 70)
    example_create_sensor()
    
    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)
