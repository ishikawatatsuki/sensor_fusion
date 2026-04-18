#!/usr/bin/env python3
"""
Simple test script for the refactored sensor system.
Tests core functionality without requiring full dependencies.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_sensor_metadata():
    """Test SensorMetadata and SensorCategory"""
    print("=" * 70)
    print("Test 1: SensorMetadata and SensorCategory")
    print("=" * 70)
    
    # Import directly to avoid loading config.py
    import importlib.util
    from common.sensor_metadata import SensorCategory, SensorMetadata, SensorQueueItem, get_sensor_registry
    from common.datatypes import CoordinateFrame
    from common.sensor_config_loader import get_sensor_registry

    # Import CoordinateFrame from datatypes
    spec = importlib.util.spec_from_file_location(
        "datatypes",
        os.path.join(os.path.dirname(__file__), '..', 'src', 'common', 'datatypes.py')
    )
    datatypes = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(datatypes)

    
    # Test SensorCategory
    assert SensorCategory.IMU.is_time_update()
    assert SensorCategory.GPS.is_measurement_update()
    assert not SensorCategory.GPS.is_time_update()
    print("✓ SensorCategory methods work correctly")
    
    # Test SensorMetadata
    metadata = SensorMetadata(
        category=SensorCategory.IMU,
        dataset="KITTI",
        sensor_id="OXTS_IMU",
        frequency=100.0,
        priority=1,
        coordinate_frame=CoordinateFrame.IMU
    )
    
    assert metadata.full_name == "KITTI_OXTS_IMU"
    assert metadata.is_time_update()
    assert not metadata.is_measurement_update()
    print(f"✓ Created SensorMetadata: {metadata}")
    
    # Test SensorQueueItem ordering
    item1 = SensorQueueItem(
        timestamp=1.0,
        priority=1,
        metadata=metadata,
        data={'test': 'data1'}
    )
    
    item2 = SensorQueueItem(
        timestamp=2.0,
        priority=1,
        metadata=metadata,
        data={'test': 'data2'}
    )
    
    item3 = SensorQueueItem(
        timestamp=1.0,
        priority=2,
        metadata=metadata,
        data={'test': 'data3'}
    )
    
    assert item1 < item2, "Earlier timestamp should be less"
    assert item1 < item3, "Lower priority value should be less when timestamps equal"
    print("✓ SensorQueueItem ordering works correctly")
    
    # Test SensorRegistry
    registry = get_sensor_registry()
    
    class DummyReader:
        pass
    
    registry.register_reader("TestDataset", "TEST_SENSOR", DummyReader)
    assert registry.get_reader_class("TestDataset", "TEST_SENSOR") == DummyReader
    print("✓ SensorRegistry works correctly")
    
    print("\n✓ All SensorMetadata tests passed!\n")


def test_priority_queue_ordering():
    """Test that PriorityQueue works correctly with SensorQueueItem"""
    print("=" * 70)
    print("Test 2: PriorityQueue Ordering")
    print("=" * 70)
    
    from queue import PriorityQueue
    import importlib.util
    from common.sensor_metadata import SensorCategory, SensorMetadata, SensorQueueItem, get_sensor_registry
    from common.datatypes import CoordinateFrame
    from common.sensor_config_loader import get_sensor_registry

    queue = PriorityQueue()
    
    # Create metadata
    imu_metadata = SensorMetadata(
        category=SensorCategory.IMU,
        dataset="TEST",
        sensor_id="IMU",
        frequency=100,
        priority=1,
        coordinate_frame=CoordinateFrame.IMU
    )
    
    gps_metadata = SensorMetadata(
        category=SensorCategory.GPS,
        dataset="TEST",
        sensor_id="GPS",
        frequency=10,
        priority=2,
        coordinate_frame=CoordinateFrame.GPS
    )
    
    # Add items in random order
    queue.put(SensorQueueItem(3.0, 1, imu_metadata, "imu3"))
    queue.put(SensorQueueItem(1.0, 2, gps_metadata, "gps1"))
    queue.put(SensorQueueItem(2.0, 1, imu_metadata, "imu2"))
    queue.put(SensorQueueItem(1.0, 1, imu_metadata, "imu1"))
    
    # Should come out ordered by timestamp, then priority
    expected_order = [
        (1.0, 1, "imu1"),  # t=1.0, prio=1
        (1.0, 2, "gps1"),  # t=1.0, prio=2
        (2.0, 1, "imu2"),  # t=2.0, prio=1
        (3.0, 1, "imu3"),  # t=3.0, prio=1
    ]
    
    for expected_t, expected_p, expected_data in expected_order:
        item = queue.get()
        assert item.timestamp == expected_t, f"Expected t={expected_t}, got {item.timestamp}"
        assert item.priority == expected_p, f"Expected p={expected_p}, got {item.priority}"
        assert item.data == expected_data, f"Expected {expected_data}, got {item.data}"
        print(f"  ✓ Correct order: t={item.timestamp}, p={item.priority}, data={item.data}")
    
    print("\n✓ PriorityQueue ordering test passed!\n")


def test_backward_compatibility():
    """Test backward compatibility functions"""
    print("=" * 70)
    print("Test 3: Backward Compatibility")
    print("=" * 70)
    
    from common.sensor_metadata import SensorCategory, SensorMetadata
    from common.datatypes import CoordinateFrame
    from common.sensor_legacy import (
        is_imu_data_legacy,
        is_time_update_legacy,
        is_measurement_update_legacy,
        is_gps_data_legacy,
        LegacySensorTypeAdapter
    )
    
    # Create test metadata
    imu_metadata = SensorMetadata(
        category=SensorCategory.IMU,
        dataset="TEST",
        sensor_id="IMU",
        frequency=100,
        priority=1,
        coordinate_frame=CoordinateFrame.IMU
    )
    
    gps_metadata = SensorMetadata(
        category=SensorCategory.GPS,
        dataset="TEST",
        sensor_id="GPS",
        frequency=10,
        priority=2,
        coordinate_frame=CoordinateFrame.GPS
    )
    
    # Test legacy functions
    assert is_imu_data_legacy(imu_metadata)
    assert not is_imu_data_legacy(gps_metadata)
    print("✓ is_imu_data_legacy works")
    
    assert is_time_update_legacy(imu_metadata)
    assert not is_time_update_legacy(gps_metadata)
    print("✓ is_time_update_legacy works")
    
    assert is_measurement_update_legacy(gps_metadata)
    assert not is_measurement_update_legacy(imu_metadata)
    print("✓ is_measurement_update_legacy works")
    
    assert is_gps_data_legacy(gps_metadata)
    assert not is_gps_data_legacy(imu_metadata)
    print("✓ is_gps_data_legacy works")
    
    # Test adapter
    adapter = LegacySensorTypeAdapter(imu_metadata)
    assert adapter.name == "IMU"
    assert adapter.value == 1
    print(f"✓ LegacySensorTypeAdapter works: {adapter}")
    
    print("\n✓ All backward compatibility tests passed!\n")


def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("REFACTORED SENSOR SYSTEM - TEST SUITE")
    print("=" * 70 + "\n")
    
    try:
        test_sensor_metadata()
        test_priority_queue_ordering()
        test_backward_compatibility()
        
        print("=" * 70)
        print("ALL TESTS PASSED! ✓")
        print("=" * 70)
        print("\nThe refactored sensor system is working correctly.")
        print("\nNext steps:")
        print("  1. Review docs/SENSOR_REFACTORING_GUIDE.md")
        print("  2. Run: python -m examples.refactored_sensor_examples")
        print("  3. Start migrating dataset classes to use RefactoredSensor")
        print()
        
        return 0
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
