"""
Refactored Sensor and Dataset Integration
==========================================

This module provides the refactored Sensor class and utilities for
integrating with the dataset readers using the new metadata system.
"""

import time
import logging
import numpy as np
from queue import PriorityQueue
from collections import namedtuple
from threading import Thread
from typing import Optional, Any

from src.common import (
    MAX_CONSECUTIVE_DROPOUT_RATIO
)
from src.common.sensor_metadata import (
    SensorMetadata,
    SensorQueueItem,
    SensorRegistry,
    get_sensor_registry,
)


class RefactoredSensor:
    """
    Refactored sensor class using SensorMetadata instead of SensorType enums.
    
    This class reads data from a dataset reader and publishes it to a priority
    queue using SensorQueueItem for proper ordering.
    """
    
    def __init__(
        self,
        metadata: SensorMetadata,
        dataset_reader: Any,
        output_queue: PriorityQueue,
        dataset_starttime: Optional[float] = None
    ):
        """
        Initialize sensor.
        
        Args:
            metadata: SensorMetadata describing this sensor
            dataset_reader: Reader instance that provides data
            output_queue: Queue to publish sensor data to
            dataset_starttime: Dataset start timestamp
        """
        self.metadata = metadata
        self.dataset = dataset_reader
        self.output_queue = output_queue
        
        self.dataset_starttime = dataset_starttime
        self.starttime = None
        self.started = False
        self.stopped = False
        
        # Legacy field for compatibility
        self.field = namedtuple('sensor', ['type', 'data'])
        
        # Select publisher based on sensor type and dropout ratio
        publisher = self._get_publisher()
        self.publish_thread = Thread(target=publisher)
    
    def _get_publisher(self):
        """Select appropriate publisher based on configuration"""
        # Use consecutive dropout for stereo cameras with dropout
        from src.common import SensorCategory
        if (self.metadata.category == SensorCategory.STEREO_CAMERA and 
            self.metadata.dropout_ratio > 0.0):
            return self.publish_consecutive_drp
        return self.publish
    
    def start(self, starttime: float):
        """Start publishing sensor data"""
        self.started = True
        self.starttime = starttime
        self.publish_thread.start()
    
    def stop(self):
        """Stop publishing sensor data"""
        self.stopped = True
        if self.started:
            self.publish_thread.join(timeout=2.0)
    
    def publish(self):
        """Standard publisher - respects dropout ratio randomly"""
        dataset = iter(self.dataset)
        while not self.stopped:
            try:
                timestamp, data = next(dataset)
            except StopIteration:
                logging.debug(f"{self.metadata.full_name} finished publishing")
                break
            
            # Apply random dropout
            if self.metadata.dropout_ratio > 0.0:
                if np.random.random() < self.metadata.dropout_ratio:
                    logging.debug(f"Dropped {self.metadata.full_name} at {timestamp}")
                    continue
            
            # Create queue item with new structure
            item = SensorQueueItem(
                timestamp=timestamp,
                priority=self.metadata.priority,
                metadata=self.metadata,
                data=data
            )
            
            self.output_queue.put(item)
            
            # Small sleep to prevent CPU spinning
            time.sleep(0.0001)
    
    def publish_consecutive_drp(self):
        """
        Publisher with consecutive dropout for stereo cameras.
        
        This creates realistic dropout patterns where consecutive frames
        are dropped together (simulating temporary occlusion, etc.)
        
        Maximum consecutive dropout: 30% of total length
        """
        dataset = list(iter(self.dataset))
        iter_length = len(dataset)
        np.random.seed(777)
        
        # Create dropout pattern list
        dp_list = [MAX_CONSECUTIVE_DROPOUT_RATIO 
                   for i in range(int(self.metadata.dropout_ratio // MAX_CONSECUTIVE_DROPOUT_RATIO))]
        
        remainder = self.metadata.dropout_ratio % MAX_CONSECUTIVE_DROPOUT_RATIO
        if remainder != 0.0 and 0.001 < remainder < MAX_CONSECUTIVE_DROPOUT_RATIO:
            dp_list.append(remainder)
        
        logging.info("-" * 30)
        logging.info(f"{self.metadata.full_name} dropout ratio list: {dp_list}")
        logging.info("-" * 30)
        
        dp_list.reverse()
        dropout_length = int(iter_length * dp_list.pop())
        start_dropping_at = np.random.randint(int(iter_length * 0.2))
        end_dropping_at = start_dropping_at + dropout_length
        
        i = 0
        while not self.stopped and i < iter_length:
            timestamp, data = dataset[i]
            
            # Check if we should drop this frame
            if start_dropping_at <= i < end_dropping_at:
                logging.debug(f"Consecutive dropout: {self.metadata.full_name} at {timestamp}")
                i += 1
                continue
            
            # Move to next dropout region if needed
            if i >= end_dropping_at and len(dp_list) > 0:
                dropout_ratio = dp_list.pop()
                dropout_length = int(iter_length * dropout_ratio)
                start_dropping_at = i + np.random.randint(int(iter_length * 0.1))
                end_dropping_at = start_dropping_at + dropout_length
            
            # Create queue item
            item = SensorQueueItem(
                timestamp=timestamp,
                priority=self.metadata.priority,
                metadata=self.metadata,
                data=data
            )
            
            self.output_queue.put(item)
            i += 1
            time.sleep(0.0001)


def register_all_readers(registry: Optional[SensorRegistry] = None):
    """
    Register all available sensor data readers with the registry.
    
    This should be called once at application startup to populate
    the registry with all known reader classes.
    
    Args:
        registry: SensorRegistry to populate (uses global if None)
    """
    if registry is None:
        registry = get_sensor_registry()
    
    # Import all reader classes
    from .kitti import (
        OXTS_GPSDataReader,
        OXTS_IMUDataReader,
        KITTI_StereoFrameReader,
        KITTI_GroundTruthDataReader,
        KITTI_UpwardLeftwardVelocityDataReader
    )
    from .uav import (
        PX4_IMUDataReader,
        PX4_GPSDataReader,
        PX4_VisualOdometryDataReader,
        PX4_MagnetometerDataReader,
        PX4_ActuatorMotorDataReader,
        PX4_ActuatorOutputDataReader,
        PX4_CustomIMUDataReader,
        VOXL_TrackingCameraDataReader,
        VOXL_IMUDataReader,
        VOXL_StereoFrameReader,
        VOXL_QVIOOverlayDataReader
    )
    from .euroc import (
        EuRoC_IMUDataReader,
        EuRoC_LeiCaDataReader,
        EuRoC_StereoFrameReader,
        EuRoC_GroundTruthDataReader
    )
    
    # Register KITTI readers
    registry.register_reader("KITTI", "OXTS_IMU", OXTS_IMUDataReader)
    # registry.register_reader("KITTI", "OXTS_IMU_UNSYNCED", OXTS_IMUDataReader)
    registry.register_reader("KITTI", "OXTS_GPS", OXTS_GPSDataReader)
    # registry.register_reader("KITTI", "OXTS_GPS_UNSYNCED", OXTS_GPSDataReader)
    registry.register_reader("KITTI", "KITTI_STEREO", KITTI_StereoFrameReader)
    registry.register_reader("KITTI", "KITTI_UPWARD_LEFTWARD_VELOCITY", 
                           KITTI_UpwardLeftwardVelocityDataReader)
    registry.register_reader("KITTI", "GROUND_TRUTH", KITTI_GroundTruthDataReader)
    
    # Register EuRoC readers
    registry.register_reader("EuRoC", "EuRoC_IMU", EuRoC_IMUDataReader)
    registry.register_reader("EuRoC", "EuRoC_LEICA", EuRoC_LeiCaDataReader)
    registry.register_reader("EuRoC", "EuRoC_STEREO", EuRoC_StereoFrameReader)
    registry.register_reader("EuRoC", "GROUND_TRUTH", EuRoC_GroundTruthDataReader)
    
    # Register UAV readers
    registry.register_reader("UAV", "VOXL_IMU0", VOXL_IMUDataReader)
    registry.register_reader("UAV", "VOXL_IMU1", VOXL_IMUDataReader)
    registry.register_reader("UAV", "VOXL_STEREO", VOXL_StereoFrameReader)
    registry.register_reader("UAV", "VOXL_QVIO_OVERLAY", VOXL_QVIOOverlayDataReader)
    registry.register_reader("UAV", "VOXL_TRACKING_CAMERA", VOXL_TrackingCameraDataReader)
    
    registry.register_reader("UAV", "PX4_IMU0", PX4_IMUDataReader)
    registry.register_reader("UAV", "PX4_IMU1", PX4_IMUDataReader)
    registry.register_reader("UAV", "PX4_GPS", PX4_GPSDataReader)
    registry.register_reader("UAV", "PX4_VO", PX4_VisualOdometryDataReader)
    registry.register_reader("UAV", "PX4_MAG", PX4_MagnetometerDataReader)
    registry.register_reader("UAV", "PX4_ACTUATOR_MOTORS", PX4_ActuatorMotorDataReader)
    registry.register_reader("UAV", "PX4_ACTUATOR_OUTPUTS", PX4_ActuatorOutputDataReader)
    registry.register_reader("UAV", "PX4_CUSTOM_IMU", PX4_CustomIMUDataReader)
    
    logging.info(f"Registered {len(registry.list_readers())} sensor readers")
    
    return registry
