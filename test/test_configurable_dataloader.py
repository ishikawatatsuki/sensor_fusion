"""
Unit tests for the configuration-driven data loader system.
Run with: python -m pytest test/test_configurable_dataloader.py
"""

import os
import sys
import pytest
import tempfile
import numpy as np

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.internal.dataset.generic_reader import (
    GenericDataReader,
    CSVDataReader,
    SensorReaderFactory,
    SensorSchema,
    TransformRegistry,
    FieldMapping,
    DataSourceConfig
)
from src.internal.dataset.configurable_adapter import ConfigurableDatasetAdapter


class TestTransformRegistry:
    """Test transformation registry"""
    
    def test_register_and_get(self):
        """Test registering and retrieving transforms"""
        def my_transform(value, **kwargs):
            return value * 2
        
        TransformRegistry.register('test_transform', my_transform)
        retrieved = TransformRegistry.get('test_transform')
        
        assert retrieved is not None
        assert retrieved(5) == 10
    
    def test_apply_transform(self):
        """Test applying transforms"""
        result = TransformRegistry.apply('timestamp_ns_to_s', 1_000_000_000)
        assert result == 1.0


class TestCSVDataReader:
    """Test CSV data reader"""
    
    @pytest.fixture
    def temp_csv(self):
        """Create a temporary CSV file for testing"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            f.write("timestamp,x,y,z\n")
            f.write("1000000000,1.0,2.0,3.0\n")
            f.write("2000000000,1.1,2.1,3.1\n")
            f.write("3000000000,1.2,2.2,3.2\n")
            temp_path = f.name
        
        yield temp_path
        
        # Cleanup
        os.unlink(temp_path)
    
    def test_read_csv_basic(self, temp_csv):
        """Test basic CSV reading"""
        schema_config = {
            'sensor_type': 'TEST_SENSOR',
            'data_source': {
                'type': 'csv',
                'path_template': '{root_path}',
                'delimiter': ',',
                'skip_header': 1
            },
            'fields': [
                {'name': 'timestamp', 'columns': [0], 'type': 'int', 'transform': 'timestamp_ns_to_s'},
                {'name': 'x', 'columns': [1], 'type': 'float'},
                {'name': 'y', 'columns': [2], 'type': 'float'},
                {'name': 'z', 'columns': [3], 'type': 'float'}
            ],
            'output_fields': ['timestamp', 'x', 'y', 'z']
        }
        
        schema = SensorSchema(schema_config)
        reader = CSVDataReader(
            schema=schema,
            root_path=temp_csv,
            variant=None
        )
        
        data = list(reader)
        
        assert len(data) == 3
        assert data[0].timestamp == 1.0
        assert data[0].x == 1.0
        assert data[1].timestamp == 2.0
        assert data[2].z == 3.2
    
    def test_read_csv_array_columns(self, temp_csv):
        """Test reading multiple columns into array"""
        schema_config = {
            'sensor_type': 'TEST_SENSOR',
            'data_source': {
                'type': 'csv',
                'path_template': '{root_path}',
                'delimiter': ',',
                'skip_header': 1
            },
            'fields': [
                {'name': 'timestamp', 'columns': [0], 'type': 'int', 'transform': 'timestamp_ns_to_s'},
                {'name': 'position', 'columns': [1, 2, 3], 'type': 'array'}
            ],
            'output_fields': ['timestamp', 'position']
        }
        
        schema = SensorSchema(schema_config)
        reader = CSVDataReader(schema=schema, root_path=temp_csv, variant=None)
        
        data = list(reader)
        
        assert len(data) == 3
        assert isinstance(data[0].position, np.ndarray)
        assert len(data[0].position) == 3
        assert np.allclose(data[0].position, [1.0, 2.0, 3.0])
    
    def test_timestamp_filtering(self, temp_csv):
        """Test filtering by start time"""
        schema_config = {
            'sensor_type': 'TEST_SENSOR',
            'data_source': {
                'type': 'csv',
                'path_template': '{root_path}',
                'delimiter': ',',
                'skip_header': 1
            },
            'fields': [
                {'name': 'timestamp', 'columns': [0], 'type': 'int', 'transform': 'timestamp_ns_to_s'},
                {'name': 'x', 'columns': [1], 'type': 'float'}
            ],
            'output_fields': ['timestamp', 'x']
        }
        
        schema = SensorSchema(schema_config)
        reader = CSVDataReader(
            schema=schema,
            root_path=temp_csv,
            variant=None,
            starttime=1.5  # Filter out first sample
        )
        
        data = list(reader)
        
        assert len(data) == 2  # Only last 2 samples
        assert data[0].timestamp == 2.0
    
    def test_scale_and_offset(self, temp_csv):
        """Test scale and offset transformations"""
        schema_config = {
            'sensor_type': 'TEST_SENSOR',
            'data_source': {
                'type': 'csv',
                'path_template': '{root_path}',
                'delimiter': ',',
                'skip_header': 1
            },
            'fields': [
                {'name': 'timestamp', 'columns': [0], 'type': 'int', 'transform': 'timestamp_ns_to_s'},
                {
                    'name': 'x_scaled',
                    'columns': [1],
                    'type': 'float',
                    'scale': 2.0,
                    'offset': 1.0
                }
            ],
            'output_fields': ['timestamp', 'x_scaled']
        }
        
        schema = SensorSchema(schema_config)
        reader = CSVDataReader(schema=schema, root_path=temp_csv, variant=None)
        
        data = list(reader)
        
        # (1.0 + 1.0) * 2.0 = 4.0
        assert data[0].x_scaled == 4.0


class TestSensorReaderFactory:
    """Test sensor reader factory"""
    
    def test_create_csv_reader(self):
        """Test creating CSV reader from config"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            f.write("ts,val\n")
            f.write("1.0,10.0\n")
            temp_path = f.name
        
        try:
            sensor_config = {
                'sensor_type': 'TEST',
                'data_source': {
                    'type': 'csv',
                    'path_template': '{root_path}',
                    'skip_header': 1
                },
                'fields': [
                    {'name': 'ts', 'columns': [0], 'type': 'float'},
                    {'name': 'val', 'columns': [1], 'type': 'float'}
                ],
                'output_fields': ['ts', 'val']
            }
            
            reader = SensorReaderFactory.create_reader(
                sensor_config=sensor_config,
                root_path=temp_path,
                variant=None
            )
            
            assert isinstance(reader, CSVDataReader)
            data = list(reader)
            assert len(data) == 1
            assert data[0].ts == 1.0
            assert data[0].val == 10.0
        
        finally:
            os.unlink(temp_path)


class TestConfigurableDatasetAdapter:
    """Test configurable dataset adapter"""
    
    @pytest.fixture
    def temp_schema_config(self):
        """Create temporary schema config"""
        config_content = """
sensor_schemas:
  test_sensor:
    sensor_type: "TEST_SENSOR"
    data_source:
      type: csv
      path_template: "{root_path}/data_{variant}.csv"
      delimiter: ","
      skip_header: 1
    fields:
      - name: timestamp
        columns: [0]
        type: float
      - name: value
        columns: [1]
        type: float
    output_fields: [timestamp, value]

dataset_sensor_mapping:
  test_dataset:
    test_sensor: test_sensor
"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yaml') as f:
            f.write(config_content)
            temp_path = f.name
        
        yield temp_path
        
        os.unlink(temp_path)
    
    def test_adapter_creation(self, temp_schema_config):
        """Test creating adapter"""
        adapter = ConfigurableDatasetAdapter(
            'test_dataset',
            schema_config_path=temp_schema_config
        )
        
        assert adapter.dataset_type == 'test_dataset'
        assert 'test_sensor' in adapter.get_available_sensors()


class TestRollingAverage:
    """Test rolling average functionality"""
    
    def test_rolling_average(self):
        """Test rolling window averaging"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            f.write("ts,val\n")
            for i in range(10):
                f.write(f"{i}.0,{i}.0\n")
            temp_path = f.name
        
        try:
            schema_config = {
                'sensor_type': 'TEST',
                'data_source': {
                    'type': 'csv',
                    'path_template': '{root_path}',
                    'skip_header': 1
                },
                'fields': [
                    {'name': 'ts', 'columns': [0], 'type': 'float'},
                    {'name': 'val', 'columns': [1], 'type': 'float'}
                ],
                'output_fields': ['ts', 'val']
            }
            
            schema = SensorSchema(schema_config)
            reader = CSVDataReader(
                schema=schema,
                root_path=temp_path,
                variant=None,
                window_size=3
            )
            
            data = list(reader)
            
            # First 2 samples: no averaging yet
            assert data[0].val == 0.0
            assert data[1].val == 1.0
            
            # Third sample onwards: averaged
            # At index 2: mean of [0, 1, 2] = 1.0
            assert data[2].val == 1.0
            
            # At index 3: mean of [1, 2, 3] = 2.0
            assert data[3].val == 2.0
        
        finally:
            os.unlink(temp_path)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
