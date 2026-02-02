import os
import sys
import logging
import time
import numpy as np

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.common.datatypes import (
    EuRoC_SensorType,
    SensorType
)
from src.internal.dataset.dataset import (
    BaseDataset,
    Sensor,
)
from src.internal.dataset.configurable_adapter import ConfigurableDatasetAdapter
from src.internal.dataset.euroc import (
    EuRoC_LeiCaDataReader,
    EuRoC_IMUDataReader,
    EuRoC_StereoFrameReader,
    EuRoC_GroundTruthDataReader,
)
from src.common.constants import (
    EUROC_SEQUENCE_MAPS,
    KITTI_SEQUENCE_TO_DATE,
    KITTI_SEQUENCE_TO_DRIVE,
)
from src.internal.extended_common import (
    DatasetType,
    FilterConfig,
    DatasetConfig,
    SensorConfig,
    GyroSpecification,
    AccelSpecification,
    SensorType, 
    KITTI_SensorType,
    UAV_SensorType,
    EuRoC_SensorType,
    State, Pose,
    CoordinateFrame,
    ExtendedSensorConfig,
    StereoField,
    CoordinateFrame,
    SensorDataField,
    SensorData,
    ControlInput,
    VisualOdometryData,
    InitialState,
)
from src.common.datatypes import DatasetType

class EuRoCDataset_ORIGINAL(BaseDataset):
    
    def _populate_sensor_to_thread(self):
        sensor_threads = []
        
        for sensor in self.sensor_list:
            # Manual match/case for each sensor type
            match sensor.sensor_type:
                case EuRoC_SensorType.EuRoC_IMU:
                    dataset = EuRoC_IMUDataReader(
                        root_path=os.path.join(
                            self.config.root_path,
                            EUROC_SEQUENCE_MAPS.get(self.config.variant)
                        ),
                        starttime=0,
                        window_size=sensor.window_size
                    )
                
                case EuRoC_SensorType.EuRoC_LEICA:
                    dataset = EuRoC_LeiCaDataReader(
                        root_path=os.path.join(
                            self.config.root_path,
                            EUROC_SEQUENCE_MAPS.get(self.config.variant)
                        ),
                        starttime=0
                    )
                
                case EuRoC_SensorType.EuRoC_STEREO:
                    dataset = EuRoC_StereoFrameReader(
                        root_path=os.path.join(
                            self.config.root_path,
                            EUROC_SEQUENCE_MAPS.get(self.config.variant)
                        ),
                        starttime=0
                    )
                
                case _:
                    raise ValueError(f"Unknown sensor type: {sensor.sensor_type}")
            
            s = Sensor(
                type=sensor.sensor_type,
                dataset=dataset,
                output_queue=self.output_queue,
                dropout_ratio=sensor.dropout_ratio
            )
            sensor_threads.append(s)
        
        # Ground truth
        self.ground_truth_dataset = EuRoC_GroundTruthDataReader(
            root_path=os.path.join(
                self.config.root_path,
                EUROC_SEQUENCE_MAPS.get(self.config.variant)
            ),
            starttime=0
        )
        s = Sensor(
            type=SensorType.GROUND_TRUTH,
            dataset=self.ground_truth_dataset,
            output_queue=self.output_queue
        )
        sensor_threads.append(s)
        
        self.sensor_threads = sensor_threads


class EuRoCDataset_NEW(BaseDataset):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Create adapter for EuRoC dataset
        self.adapter = ConfigurableDatasetAdapter('euroc')
        
        self._populate_sensor_to_thread()
    
    def _populate_sensor_to_thread(self):
        """
        Simplified version using configurable data loader.
        No more match/case statements!
        """
        sensor_threads = []
        
        # Map sensor types to config names
        # This mapping could also be in your config file
        sensor_name_map = {
            'EuRoC_IMU': 'euroc_imu',
            'EuRoC_LEICA': 'euroc_leica',
            'EuRoC_STEREO': 'euroc_stereo',
        }
        
        for sensor in self.sensor_list:
            # Get sensor config name
            sensor_config_name = sensor_name_map.get(
                sensor.sensor_type.name,
                sensor.name  # Fallback to sensor name
            )
            
            # Create reader using adapter - ONE LINE!
            dataset = self.adapter.create_sensor_reader(
                sensor_config_name,
                root_path=self.config.root_path,
                variant=self.config.variant,
                starttime=0,
                window_size=sensor.window_size
            )
            
            s = Sensor(
                type=sensor.sensor_type,
                dataset=dataset,
                output_queue=self.output_queue,
                dropout_ratio=sensor.dropout_ratio
            )
            sensor_threads.append(s)
        
        # Ground truth
        self.ground_truth_dataset = self.adapter.create_sensor_reader(
            'ground_truth',
            root_path=self.config.root_path,
            variant=self.config.variant,
            starttime=0
        )
        
        s = Sensor(
            type=SensorType.GROUND_TRUTH,
            dataset=self.ground_truth_dataset,
            output_queue=self.output_queue
        )
        sensor_threads.append(s)
        
        self.sensor_threads = sensor_threads


# ============================================================================
# EVEN BETTER: Fully automated with no mapping needed
# ============================================================================

class EuRoCDataset_BEST(BaseDataset):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.adapter = ConfigurableDatasetAdapter('euroc')
        self._populate_sensor_to_thread()
    
    def _populate_sensor_to_thread(self):
        """
        Fully automated - uses sensor.name directly from config.
        No manual mapping needed!
        """
        sensor_threads = []
        
        for sensor in self.sensor_list:
            # sensor.name comes from your config (e.g., 'euroc_imu')
            # It matches the sensor name in sensor_schemas.yaml
            try:
                dataset = self.adapter.create_sensor_reader(
                    sensor.name,  # Direct from config!
                    root_path=self.config.root_path,
                    variant=self.config.variant,
                    starttime=0,
                    window_size=sensor.window_size
                )
            except ValueError as e:
                logging.error(f"Failed to create reader for {sensor.name}: {e}")
                continue
            
            s = Sensor(
                type=sensor.sensor_type,
                dataset=dataset,
                output_queue=self.output_queue,
                dropout_ratio=sensor.dropout_ratio
            )
            sensor_threads.append(s)
        
        # Ground truth
        self.ground_truth_dataset = self.adapter.create_sensor_reader(
            'ground_truth',
            root_path=self.config.root_path,
            variant=self.config.variant
        )
        sensor_threads.append(
            Sensor(
                type=SensorType.GROUND_TRUTH,
                dataset=self.ground_truth_dataset,
                output_queue=self.output_queue
            )
        )
        
        self.sensor_threads = sensor_threads

    def start(self):
        now = time.time()
        
        for sensor_thread in self.sensor_threads:
            sensor_thread.start(starttime=now)

        time.sleep(0.5)
        last_timestamp, _ = self.output_queue.get()
        self.last_timestamp = last_timestamp
        
    def stop(self):
        for sensor_thread in self.sensor_threads:
            sensor_thread.stop()

    def get_sensor_data(self) -> SensorDataField:
        
        timestamp, sensor_data = self.output_queue.get()

        if SensorType.is_time_update(sensor_data.type):
            last_timestamp = self.last_timestamp
            self.last_timestamp = timestamp
            dt = timestamp - last_timestamp
        
        match(sensor_data.type):
            case EuRoC_SensorType.EuRoC_IMU:
                u = np.hstack([sensor_data.data.a, sensor_data.data.w])
                return SensorDataField(
                    type=sensor_data.type, 
                    timestamp=timestamp, 
                    data=ControlInput(u=u, dt=dt),
                    coordinate_frame=CoordinateFrame.IMU)

            case EuRoC_SensorType.EuRoC_LEICA:
                z = np.array([sensor_data.data.p_x, sensor_data.data.p_y, sensor_data.data.p_z])
                return SensorDataField(
                    type=sensor_data.type, 
                    timestamp=timestamp, 
                    data=SensorData(z=z),
                    coordinate_frame=CoordinateFrame.LEICA)
            
            case EuRoC_SensorType.EuRoC_STEREO:
                return SensorDataField(
                    type=sensor_data.type, 
                    timestamp=timestamp, 
                    data=StereoField(
                        left_frame_id=sensor_data.data.left_frame_id,
                        right_frame_id=sensor_data.data.right_frame_id),
                    coordinate_frame=CoordinateFrame.STEREO_LEFT)

            case SensorType.GROUND_TRUTH:
                z = np.hstack([sensor_data.data.p, sensor_data.data.v, sensor_data.data.q])
                return SensorDataField(
                    type=sensor_data.type, 
                    timestamp=timestamp, 
                    data=SensorData(z=z),
                    coordinate_frame=CoordinateFrame.INERTIAL)

            case _:
                return SensorDataField(
                    type=sensor_data.type, 
                    timestamp=timestamp, 
                    data=SensorData(z=np.zeros(3)),
                    coordinate_frame=CoordinateFrame.INERTIAL)


# ============================================================================
# For KITTI Dataset
# ============================================================================

class KITTIDataset_NEW(BaseDataset):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.adapter = ConfigurableDatasetAdapter('kitti')
        self._populate_sensor_to_thread()
    
    def _populate_sensor_to_thread(self):
        sensor_threads = []
        
        # Get date and drive from variant
        date = KITTI_SEQUENCE_TO_DATE.get(self.config.variant)
        drive = KITTI_SEQUENCE_TO_DRIVE.get(self.config.variant)
        
        for sensor in self.sensor_list:
            dataset = self.adapter.create_sensor_reader(
                sensor.name,  # e.g., 'oxts_imu', 'oxts_gps'
                root_path=self.config.root_path,
                variant=drive,
                starttime=0,
                window_size=sensor.window_size,
                date=date,  # KITTI-specific parameter
                drive=drive  # KITTI-specific parameter
            )
            
            s = Sensor(
                type=sensor.sensor_type,
                dataset=dataset,
                output_queue=self.output_queue,
                dropout_ratio=sensor.dropout_ratio
            )
            sensor_threads.append(s)
        
        # Ground truth
        self.ground_truth_dataset = self.adapter.create_sensor_reader(
            'ground_truth',
            root_path=self.config.root_path,
            variant=drive,
            date=date,
            drive=drive
        )
        sensor_threads.append(
            Sensor(
                type=SensorType.GROUND_TRUTH,
                dataset=self.ground_truth_dataset,
                output_queue=self.output_queue
            )
        )
        
        self.sensor_threads = sensor_threads


# ============================================================================
# Complete example with error handling and logging
# ============================================================================

class RobustDataset(BaseDataset):
    """
    Production-ready dataset class with proper error handling
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Create adapter
        try:
            self.adapter = ConfigurableDatasetAdapter(self.config.type)
            logging.info(f"Created adapter for dataset type: {self.config.type}")
            logging.info(f"Available sensors: {self.adapter.get_available_sensors()}")
        except Exception as e:
            logging.error(f"Failed to create adapter: {e}")
            raise
        
        self._populate_sensor_to_thread()
    
    def _populate_sensor_to_thread(self):
        sensor_threads = []
        
        # Get dataset-specific parameters
        extra_params = self._get_dataset_params()
        
        # Create readers for each sensor
        for sensor in self.sensor_list:
            logging.debug(f"Creating reader for sensor: {sensor.name}")
            
            try:
                dataset = self.adapter.create_sensor_reader(
                    sensor.name,
                    root_path=self.config.root_path,
                    variant=self.config.variant,
                    starttime=0,
                    window_size=sensor.window_size,
                    **extra_params
                )
                
                s = Sensor(
                    type=sensor.sensor_type,
                    dataset=dataset,
                    output_queue=self.output_queue,
                    dropout_ratio=sensor.dropout_ratio
                )
                sensor_threads.append(s)
                
                logging.info(f"✓ Created reader for {sensor.name}")
                
            except Exception as e:
                logging.error(f"✗ Failed to create reader for {sensor.name}: {e}")
                # Depending on your requirements:
                # Option 1: Skip this sensor and continue
                continue
                # Option 2: Raise error and stop
                # raise
        
        # Ground truth
        try:
            self.ground_truth_dataset = self.adapter.create_sensor_reader(
                'ground_truth',
                root_path=self.config.root_path,
                variant=self.config.variant,
                **extra_params
            )
            sensor_threads.append(
                Sensor(
                    type=SensorType.GROUND_TRUTH,
                    dataset=self.ground_truth_dataset,
                    output_queue=self.output_queue
                )
            )
            logging.info("✓ Created ground truth reader")
        except Exception as e:
            logging.warning(f"No ground truth available: {e}")
        
        if not sensor_threads:
            raise ValueError("No sensor readers created!")
        
        self.sensor_threads = sensor_threads
        logging.info(f"Total sensors initialized: {len(sensor_threads)}")
    
    def _get_dataset_params(self) -> dict:
        """Get dataset-specific parameters"""
        params = {}
        
        if self.type == DatasetType.KITTI:
            params['date'] = KITTI_SEQUENCE_TO_DATE.get(self.config.variant)
            params['drive'] = KITTI_SEQUENCE_TO_DRIVE.get(self.config.variant)
        
        return params


# ============================================================================
# Usage Example
# ============================================================================

if __name__ == "__main__":
    import yaml
    from src.internal.extended_common import DatasetConfig
    
    # Load config
    with open('./configs/urban_nav_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create dataset config
    dataset_config = DatasetConfig(**config['dataset'])
    
    # Create dataset using new implementation
    dataset = RobustDataset(config=dataset_config)
    
    # Start and use as before
    dataset.start()
    
    print("Reading sensor data...")
    for i in range(100):
        if dataset.is_queue_empty():
            break
        
        sensor_data = dataset.get_sensor_data()
        print(f"{sensor_data.type.name}: t={sensor_data.timestamp}")
    
    dataset.stop()
    print("Done!")

