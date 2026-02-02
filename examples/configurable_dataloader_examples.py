"""
Example: Using the configuration-driven data loader system

This example demonstrates how to use the new configurable data loader
in various scenarios.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.internal.dataset.configurable_adapter import (
    ConfigurableDatasetAdapter,
    create_data_reader_from_config
)


def example_1_basic_usage():
    """Example 1: Basic usage with EuRoC dataset"""
    print("\n" + "="*80)
    print("Example 1: Basic Configuration-Driven Data Loading")
    print("="*80)
    
    # Create adapter for EuRoC dataset
    adapter = ConfigurableDatasetAdapter('euroc')
    
    # List available sensors
    print("\nAvailable sensors:", adapter.get_available_sensors())
    
    # Create IMU reader
    imu_reader = adapter.create_sensor_reader(
        'euroc_imu',
        root_path='./data/EuRoC',
        variant='01',
        starttime=0.0,
        window_size=5  # Apply 5-sample rolling average
    )
    
    # Read first 10 samples
    print("\nReading first 10 IMU samples:")
    for i, data in enumerate(imu_reader):
        if i >= 10:
            break
        print(f"  t={data.timestamp:.6f}s, accel={data.a}, gyro={data.w}")


def example_2_from_config_file():
    """Example 2: Create reader from existing config file"""
    print("\n" + "="*80)
    print("Example 2: Loading from Config File")
    print("="*80)
    
    import yaml
    
    # Load existing config
    with open('./configs/euroc_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    dataset_config = config['dataset']
    
    # Create reader using config
    imu_reader = create_data_reader_from_config(
        dataset_config,
        'euroc_imu',
        window_size=10  # Override window size
    )
    
    leica_reader = create_data_reader_from_config(
        dataset_config,
        'euroc_leica'
    )
    
    print(f"\nIMU start time: {imu_reader.start_time()}")
    print(f"Leica start time: {leica_reader.start_time()}")


def example_3_compare_with_legacy():
    """Example 3: Compare new reader with legacy reader"""
    print("\n" + "="*80)
    print("Example 3: Comparing New vs Legacy Readers")
    print("="*80)
    
    # Legacy reader
    from src.internal.dataset.euroc import EuRoC_IMUDataReader as LegacyReader
    
    # New reader (drop-in replacement)
    from src.internal.dataset.configurable_adapter import (
        ConfigurableEuRoCIMUReader as NewReader
    )
    
    root_path = './data/EuRoC/MH_01_easy'
    
    # Create both readers
    legacy = LegacyReader(root_path=root_path)
    new = NewReader(root_path=root_path)
    
    # Compare first 5 samples
    print("\nComparing first 5 samples:")
    legacy_iter = iter(legacy)
    new_iter = iter(new)
    
    for i in range(5):
        legacy_data = next(legacy_iter)
        new_data = next(new_iter)
        
        print(f"\nSample {i+1}:")
        print(f"  Legacy timestamp: {legacy_data.timestamp:.6f}")
        print(f"  New timestamp:    {new_data.timestamp:.6f}")
        print(f"  Match: {np.allclose([legacy_data.timestamp], [new_data.timestamp])}")


def example_4_visualize_data():
    """Example 4: Visualize sensor data"""
    print("\n" + "="*80)
    print("Example 4: Visualizing Sensor Data")
    print("="*80)
    
    adapter = ConfigurableDatasetAdapter('euroc')
    
    # Read IMU data
    imu_reader = adapter.create_sensor_reader(
        'euroc_imu',
        root_path='./data/EuRoC',
        variant='01',
        window_size=1
    )
    
    # Collect data
    timestamps = []
    accelerations = []
    gyroscopes = []
    
    print("\nCollecting 1000 samples...")
    for i, data in enumerate(imu_reader):
        if i >= 1000:
            break
        timestamps.append(data.timestamp)
        accelerations.append(data.a)
        gyroscopes.append(data.w)
    
    timestamps = np.array(timestamps)
    accelerations = np.array(accelerations)
    gyroscopes = np.array(gyroscopes)
    
    # Normalize timestamps
    timestamps = timestamps - timestamps[0]
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Acceleration
    ax1.plot(timestamps, accelerations[:, 0], label='X', alpha=0.7)
    ax1.plot(timestamps, accelerations[:, 1], label='Y', alpha=0.7)
    ax1.plot(timestamps, accelerations[:, 2], label='Z', alpha=0.7)
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Acceleration [m/s²]')
    ax1.set_title('IMU Acceleration (Config-Driven Reader)')
    ax1.legend()
    ax1.grid(True)
    
    # Gyroscope
    ax2.plot(timestamps, gyroscopes[:, 0], label='X', alpha=0.7)
    ax2.plot(timestamps, gyroscopes[:, 1], label='Y', alpha=0.7)
    ax2.plot(timestamps, gyroscopes[:, 2], label='Z', alpha=0.7)
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Angular Velocity [rad/s]')
    ax2.set_title('IMU Gyroscope (Config-Driven Reader)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('./outputs/configurable_reader_example.png', dpi=150)
    print("Saved plot to: ./outputs/configurable_reader_example.png")
    # plt.show()


def example_5_multiple_sensors():
    """Example 5: Reading from multiple sensors with synchronized timestamps"""
    print("\n" + "="*80)
    print("Example 5: Multi-Sensor Data Fusion")
    print("="*80)
    
    from queue import PriorityQueue
    from collections import namedtuple
    
    adapter = ConfigurableDatasetAdapter('euroc')
    
    # Create multiple readers
    readers = {
        'imu': adapter.create_sensor_reader(
            'euroc_imu',
            root_path='./data/EuRoC',
            variant='01'
        ),
        'leica': adapter.create_sensor_reader(
            'euroc_leica',
            root_path='./data/EuRoC',
            variant='01'
        ),
    }
    
    # Simulate priority queue (as in your current system)
    queue = PriorityQueue()
    SensorData = namedtuple('SensorData', ['sensor_type', 'data'])
    
    # Add first 100 samples from each sensor to queue
    print("\nAdding sensor data to priority queue...")
    for sensor_type, reader in readers.items():
        for i, data in enumerate(reader):
            if i >= 100:
                break
            queue.put((data.timestamp, SensorData(sensor_type, data)))
    
    # Process in timestamp order
    print(f"\nProcessing {queue.qsize()} samples in temporal order:")
    
    sensor_counts = {'imu': 0, 'leica': 0}
    last_timestamp = -1
    
    for i in range(min(20, queue.qsize())):  # Show first 20
        timestamp, sensor_data = queue.get()
        sensor_counts[sensor_data.sensor_type] += 1
        
        # Verify ordering
        assert timestamp >= last_timestamp, "Timestamps not in order!"
        last_timestamp = timestamp
        
        print(f"  t={timestamp:.6f}s - {sensor_data.sensor_type:>5s}: {sensor_data.data}")
    
    print(f"\nSensor distribution: {sensor_counts}")
    print("✓ All data correctly ordered by timestamp")


def example_6_custom_dataset():
    """Example 6: Adding a completely new dataset via config only"""
    print("\n" + "="*80)
    print("Example 6: Custom Dataset (Config Only)")
    print("="*80)
    
    print("""
This example shows how to add a new dataset without writing Python code.

Steps:
1. Add sensor schema to configs/sensor_schemas.yaml
2. Create dataset mapping
3. Use immediately!

Example schema for custom GPS logger:

sensor_schemas:
  my_gps_logger:
    sensor_type: "CUSTOM_GPS"
    data_source:
      type: csv
      path_template: "{root_path}/log_{variant}.csv"
      delimiter: ","
      skip_header: 1
    fields:
      - name: timestamp
        columns: [0]
        type: float
      - name: lat
        columns: [1]
        type: float
      - name: lon
        columns: [2]
        type: float
      - name: alt
        columns: [3]
        type: float
    output_fields: [timestamp, lat, lon, alt]

dataset_sensor_mapping:
  my_logger:
    gps: my_gps_logger

Usage:
  adapter = ConfigurableDatasetAdapter('my_logger')
  reader = adapter.create_sensor_reader('gps', root_path='./data', variant='001')
  
No Python code needed!
    """)


def example_7_integration_with_existing_dataset():
    """Example 7: Integrate with existing Dataset class"""
    print("\n" + "="*80)
    print("Example 7: Integration with Existing Dataset Class")
    print("="*80)
    
    print("""
To integrate with your existing Dataset classes, modify the 
_populate_sensor_to_thread method:

# In dataset.py, EuRoCDataset class:

from .configurable_adapter import ConfigurableDatasetAdapter

class EuRoCDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.adapter = ConfigurableDatasetAdapter('euroc')
        self._populate_sensor_to_thread()
    
    def _populate_sensor_to_thread(self):
        sensor_threads = []
        
        for sensor in self.sensor_list:
            # Use adapter instead of manual class selection
            dataset = self.adapter.create_sensor_reader(
                sensor.name,
                root_path=self.config.root_path,
                variant=self.config.variant,
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
        gt_dataset = self.adapter.create_sensor_reader(
            'ground_truth',
            root_path=self.config.root_path,
            variant=self.config.variant
        )
        sensor_threads.append(Sensor(
            type=SensorType.GROUND_TRUTH,
            dataset=gt_dataset,
            output_queue=self.output_queue
        ))
        
        self.sensor_threads = sensor_threads

Benefits:
- Remove all the match/case statements
- No need to import specific reader classes
- Add new sensors by just updating config
- Centralized sensor definitions
    """)


def main():
    """Run all examples"""
    examples = [
        ("Basic Usage", example_1_basic_usage),
        ("From Config File", example_2_from_config_file),
        ("Compare Legacy", example_3_compare_with_legacy),
        ("Visualize Data", example_4_visualize_data),
        ("Multi-Sensor", example_5_multiple_sensors),
        ("Custom Dataset", example_6_custom_dataset),
        ("Integration", example_7_integration_with_existing_dataset),
    ]
    
    print("\n" + "="*80)
    print(" Configuration-Driven Data Loader Examples")
    print("="*80)
    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    print(f"  0. Run all")
    
    choice = input("\nSelect example (0-7): ").strip()
    
    if choice == '0':
        for name, func in examples:
            try:
                func()
            except Exception as e:
                print(f"\n⚠️  Example '{name}' failed: {e}")
    elif choice.isdigit() and 1 <= int(choice) <= len(examples):
        name, func = examples[int(choice) - 1]
        try:
            func()
        except Exception as e:
            print(f"\n⚠️  Example failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Invalid choice")
    
    print("\n" + "="*80)
    print("Done!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
