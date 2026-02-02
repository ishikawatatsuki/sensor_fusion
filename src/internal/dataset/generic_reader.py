"""
Generic configuration-driven data reader for sensor datasets.
This module provides a flexible data loading system that can be configured
entirely through YAML configuration files without code changes.
"""

import os
import yaml
import numpy as np
from collections import namedtuple
from typing import Dict, List, Any, Optional, Callable
from abc import ABC, abstractmethod


class DataSourceConfig:
    """Configuration for a data source (file, API, etc.)"""
    def __init__(self, config: Dict[str, Any]):
        self.type = config.get('type', 'csv')  # csv, binary, api, etc.
        self.path_template = config.get('path_template', '')
        self.format_spec = config.get('format', {})
        
        # CSV-specific settings
        self.delimiter = config.get('delimiter', ',')
        self.skip_header = config.get('skip_header', 1)
        self.encoding = config.get('encoding', 'utf-8')
        

class FieldMapping:
    """Mapping configuration for a single field"""
    def __init__(self, config: Dict[str, Any]):
        self.name = config['name']
        self.columns = config.get('columns', [])  # Column indices or names
        self.type = config.get('type', 'float')  # float, int, string, timestamp
        self.transform = config.get('transform', None)  # transformation function name
        self.scale = config.get('scale', 1.0)
        self.offset = config.get('offset', 0.0)
        self.noise = config.get('noise', None)  # Noise specification
        

class SensorSchema:
    """Complete schema for a sensor data source"""
    def __init__(self, config: Dict[str, Any]):
        self.sensor_type = config['sensor_type']
        self.data_source = DataSourceConfig(config['data_source'])
        self.fields = [FieldMapping(f) for f in config['fields']]
        self.output_fields = config.get('output_fields', [f.name for f in self.fields])
        self.timestamp_config = config.get('timestamp', {})
        

class TransformRegistry:
    """Registry of transformation functions"""
    _transforms: Dict[str, Callable] = {}
    
    @classmethod
    def register(cls, name: str, func: Callable):
        """Register a transformation function"""
        cls._transforms[name] = func
    
    @classmethod
    def get(cls, name: str) -> Optional[Callable]:
        """Get a transformation function by name"""
        return cls._transforms.get(name)
    
    @classmethod
    def apply(cls, name: str, value: Any, **kwargs) -> Any:
        """Apply a transformation"""
        func = cls.get(name)
        if func:
            return func(value, **kwargs)
        return value


# Register common transformations
def timestamp_ns_to_s(value, **kwargs):
    """Convert nanoseconds to seconds"""
    return int(value) / 1_000_000_000

def timestamp_to_unix(value, **kwargs):
    """Convert datetime to unix timestamp"""
    from datetime import datetime
    if isinstance(value, datetime):
        return datetime.timestamp(value) * kwargs.get('multiplier', 1)
    return value

def add_gaussian_noise(value, **kwargs):
    """Add Gaussian noise to value"""
    noise_std = kwargs.get('noise_std', 0.0)
    if noise_std > 0:
        return value - np.random.normal(0, noise_std, value.shape if hasattr(value, 'shape') else 1)
    return value

def array_from_columns(values, **kwargs):
    """Create numpy array from list of values"""
    return np.array(values)

TransformRegistry.register('timestamp_ns_to_s', timestamp_ns_to_s)
TransformRegistry.register('timestamp_to_unix', timestamp_to_unix)
TransformRegistry.register('add_gaussian_noise', add_gaussian_noise)
TransformRegistry.register('array_from_columns', array_from_columns)


class GenericDataReader(ABC):
    """Abstract base class for generic data readers"""
    
    def __init__(
        self,
        schema: SensorSchema,
        root_path: str,
        variant: str = None,
        starttime: float = -float('inf'),
        window_size: Optional[int] = None,
        **kwargs
    ):
        self.schema = schema
        self.root_path = root_path
        self.variant = variant
        self.starttime = starttime
        self.window_size = window_size
        self.kwargs = kwargs
        
        # Create named tuple for output
        self.field = namedtuple('data', self.schema.output_fields)
        
        # Rolling window buffer
        self.buffer = []
        
        # Build file path
        self.data_path = self._build_path()
        
    def _build_path(self) -> str:
        """Build the data file path from template"""
        template = self.schema.data_source.path_template
        return template.format(
            root_path=self.root_path,
            variant=self.variant,
            **self.kwargs
        )
    
    @abstractmethod
    def _read_raw_data(self):
        """Read raw data from source - to be implemented by subclasses"""
        pass
    
    def parse(self, raw_line: Any) -> Optional[namedtuple]:
        """Parse a raw data line according to schema"""
        try:
            parsed_values = {}
            
            for field in self.schema.fields:
                value = self._extract_field_value(raw_line, field)
                parsed_values[field.name] = value
            
            # Check timestamp filtering
            if 'timestamp' in parsed_values:
                if parsed_values['timestamp'] < self.starttime:
                    return None
            
            # Create output tuple
            output_values = [parsed_values[field] for field in self.schema.output_fields]
            return self.field(*output_values)
            
        except Exception as e:
            # Skip malformed lines
            return None
    
    def _extract_field_value(self, raw_line: Any, field: FieldMapping) -> Any:
        """Extract and transform a field value from raw data"""
        # Extract raw value(s) from columns
        if isinstance(raw_line, str):
            # CSV line
            parts = raw_line.strip().split(self.schema.data_source.delimiter)
            if len(field.columns) == 1:
                raw_value = parts[field.columns[0]]
            else:
                raw_value = [parts[i] for i in field.columns]
        else:
            # Object with attributes
            if len(field.columns) == 1:
                raw_value = getattr(raw_line, field.columns[0])
            else:
                raw_value = [getattr(raw_line, col) for col in field.columns]
        
        # Convert type
        value = self._convert_type(raw_value, field.type)
        
        # Apply transformation if specified
        if field.transform:
            transform_kwargs = {}
            if field.noise:
                transform_kwargs['noise_std'] = field.noise
            value = TransformRegistry.apply(field.transform, value, **transform_kwargs)
        
        # Apply scale and offset
        if isinstance(value, (int, float, np.ndarray)):
            value = (value + field.offset) * field.scale
        
        # Add noise if specified
        if field.noise and not field.transform:
            if isinstance(value, np.ndarray):
                value = value - np.random.normal(0, field.noise, value.shape)
            else:
                value = value - np.random.normal(0, field.noise)
        
        return value
    
    def _convert_type(self, raw_value, type_str: str) -> Any:
        """Convert raw value to specified type"""
        if type_str == 'float':
            if isinstance(raw_value, list):
                return np.array([float(v) for v in raw_value])
            return float(raw_value)
        elif type_str == 'int':
            if isinstance(raw_value, list):
                return np.array([int(v) for v in raw_value])
            return int(raw_value)
        elif type_str == 'string':
            return str(raw_value)
        elif type_str == 'array':
            return np.array([float(v) for v in raw_value])
        else:
            return raw_value
    
    def rolling_average(self, data: namedtuple) -> namedtuple:
        """Apply rolling window averaging"""
        if self.window_size is None:
            return data
        
        # Convert to array (skip timestamp)
        data_dict = data._asdict()
        timestamp = data_dict.get('timestamp')
        
        # Concatenate all numeric fields
        numeric_data = []
        for field_name in self.schema.output_fields:
            if field_name != 'timestamp':
                value = data_dict[field_name]
                if isinstance(value, np.ndarray):
                    numeric_data.extend(value)
                else:
                    numeric_data.append(value)
        
        self.buffer.append(numeric_data)
        
        if len(self.buffer) > self.window_size:
            mean = np.mean(self.buffer, axis=0)
            self.buffer = self.buffer[-self.window_size:]
            
            # Reconstruct data with averaged values
            idx = 0
            averaged_values = {'timestamp': timestamp}
            for field_name in self.schema.output_fields:
                if field_name != 'timestamp':
                    field = next(f for f in self.schema.fields if f.name == field_name)
                    num_cols = len(field.columns) if len(field.columns) > 1 else 1
                    if num_cols > 1:
                        averaged_values[field_name] = mean[idx:idx+num_cols]
                    else:
                        averaged_values[field_name] = mean[idx]
                    idx += num_cols
            
            output_values = [averaged_values[field] for field in self.schema.output_fields]
            return self.field(*output_values)
        
        return data
    
    def __iter__(self):
        """Iterate over data"""
        for raw_data in self._read_raw_data():
            parsed = self.parse(raw_data)
            if parsed is None:
                continue
            
            if self.window_size is not None:
                parsed = self.rolling_average(parsed)
            
            yield parsed
    
    def start_time(self):
        """Get the first timestamp from the dataset"""
        for raw_data in self._read_raw_data():
            parsed = self.parse(raw_data)
            if parsed and hasattr(parsed, 'timestamp'):
                return parsed.timestamp
        return None
    
    def set_starttime(self, starttime: float):
        """Set the start time filter"""
        self.starttime = starttime


class CSVDataReader(GenericDataReader):
    """CSV file data reader"""
    
    def _read_raw_data(self):
        """Read CSV file line by line"""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        with open(self.data_path, 'r', encoding=self.schema.data_source.encoding) as f:
            # Skip header lines
            for _ in range(self.schema.data_source.skip_header):
                next(f)
            
            # Read data lines
            for line in f:
                yield line


class PyKittiDataReader(GenericDataReader):
    """Data reader for PyKitti datasets"""
    
    def __init__(self, *args, date: str = None, drive: str = None, **kwargs):
        self.date = date
        self.drive = drive
        super().__init__(*args, **kwargs)
        
        # Import pykitti
        import pykitti
        self.kitti_dataset = pykitti.raw(self.root_path, self.date, self.drive)
        
    def _read_raw_data(self):
        """Read from PyKitti dataset"""
        data_type = self.schema.data_source.format_spec.get('pykitti_type', 'oxts')
        
        if data_type == 'oxts':
            for i, oxts in enumerate(self.kitti_dataset.oxts):
                # Create a wrapper object with both packet and timestamp
                class DataWrapper:
                    def __init__(self, packet, timestamp, index):
                        self.packet = packet
                        self.timestamp = timestamp
                        self.index = i
                
                wrapper = DataWrapper(
                    oxts.packet,
                    self.kitti_dataset.timestamps[i],
                    i
                )
                yield wrapper
        # Add other PyKitti data types as needed


class SensorReaderFactory:
    """Factory for creating sensor data readers from configuration"""
    
    _reader_types = {
        'csv': CSVDataReader,
        'pykitti': PyKittiDataReader,
    }
    
    @classmethod
    def register_reader(cls, name: str, reader_class: type):
        """Register a custom reader type"""
        cls._reader_types[name] = reader_class
    
    @classmethod
    def create_reader(
        cls,
        sensor_config: Dict[str, Any],
        root_path: str,
        variant: str = None,
        **kwargs
    ) -> GenericDataReader:
        """Create a data reader from configuration"""
        
        # Parse schema
        schema = SensorSchema(sensor_config)
        
        # Get reader class
        reader_type = schema.data_source.type
        reader_class = cls._reader_types.get(reader_type)
        
        if reader_class is None:
            raise ValueError(f"Unknown reader type: {reader_type}")
        
        # Create reader instance
        return reader_class(
            schema=schema,
            root_path=root_path,
            variant=variant,
            **kwargs
        )
    
    @classmethod
    def create_from_config_file(
        cls,
        config_path: str,
        sensor_name: str,
        root_path: str,
        variant: str = None,
        **kwargs
    ) -> GenericDataReader:
        """Create a reader from a YAML config file"""
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Find sensor schema in config
        sensor_schemas = config.get('sensor_schemas', {})
        sensor_config = sensor_schemas.get(sensor_name)
        
        if sensor_config is None:
            raise ValueError(f"Sensor '{sensor_name}' not found in config")
        
        return cls.create_reader(sensor_config, root_path, variant, **kwargs)
