# Sensor System Refactoring - Migration Guide

## Overview

The sensor fusion pipeline has been refactored to use a more flexible, configuration-driven sensor metadata system. This replaces the rigid IntEnum-based approach with a generalized system that supports:

- **Easy dataset addition**: Add new datasets without modifying core enums
- **Multiple sensor instances**: Support IMU0, IMU1, etc. at different frequencies
- **Configuration-driven**: Define sensors in YAML instead of Python code
- **Clear separation**: Sensor category vs. sensor instance
- **Backward compatible**: Legacy code continues to work

## Key Changes

### Before (Old System)

```python
from src.common import KITTI_SensorType, UAV_SensorType, SensorType

# Hardcoded enums per dataset
imu_type = KITTI_SensorType.OXTS_IMU
gps_type = KITTI_SensorType.OXTS_GPS

# Static methods on SensorType class
if SensorType.is_imu_data(imu_type):
    # Process IMU

# IntEnum values used for PriorityQueue ordering
queue.put((timestamp, imu_type.value, data))
```

### After (New System)

```python
from src.common import (
    SensorCategory,
    SensorMetadata,
    SensorQueueItem,
    create_sensor_metadata_from_config
)

# Configuration-driven metadata
imu_metadata = create_sensor_metadata_from_config("KITTI", "OXTS_IMU")
gps_metadata = create_sensor_metadata_from_config("KITTI", "OXTS_GPS")

# Category-based checks
if imu_metadata.category == SensorCategory.IMU:
    # Process IMU

# Explicit priority queue items
item = SensorQueueItem(
    timestamp=timestamp,
    priority=imu_metadata.priority,
    metadata=imu_metadata,
    data=data
)
queue.put(item)
```

## Architecture Components

### 1. SensorCategory (Enum)

Generic sensor categories that apply across all datasets:

```python
class SensorCategory(Enum):
    IMU = auto()
    GPS = auto()
    MAGNETOMETER = auto()
    VISUAL_ODOMETRY = auto()
    STEREO_CAMERA = auto()
    # ... etc
```

**Benefits**:
- Dataset-independent
- Clear semantic meaning
- Easy to add new categories

### 2. SensorMetadata (Dataclass)

Describes a specific sensor instance:

```python
@dataclass
class SensorMetadata:
    category: SensorCategory
    dataset: str              # "KITTI", "EuRoC", "UAV"
    sensor_id: str            # "OXTS_IMU", "IMU0", etc.
    frequency: float
    priority: int
    coordinate_frame: CoordinateFrame
    dropout_ratio: float = 0.0
    window_size: int = 1
    noise_profile: Optional[Dict] = None
```

**Benefits**:
- Rich metadata in one place
- Unique identification via `full_name` property
- Supports multiple instances of same category

### 3. SensorQueueItem (Ordered Dataclass)

Replaces raw tuples in PriorityQueue:

```python
@dataclass(order=True)
class SensorQueueItem:
    timestamp: float
    priority: int
    metadata: SensorMetadata  # Not used for comparison
    data: Any                 # Not used for comparison
```

**Benefits**:
- Explicit ordering logic
- Type-safe
- Self-documenting

### 4. SensorRegistry

Dynamic registration of readers and processors:

```python
registry = get_sensor_registry()

# Register readers
registry.register_reader("KITTI", "OXTS_IMU", OXTS_IMUDataReader)

# Get reader
reader_class = registry.get_reader_class("KITTI", "OXTS_IMU")
```

**Benefits**:
- Decouples sensor types from implementations
- Supports runtime registration
- Testable with mock readers

### 5. Configuration File

Define sensors in YAML (`configs/sensor_registry.yaml`):

```yaml
datasets:
  KITTI:
    sensors:
      - id: OXTS_IMU
        category: IMU
        frequency: 100
        priority: 1
        coordinate_frame: IMU
        reader_class: OXTS_IMUDataReader
```

**Benefits**:
- No code changes for new sensors
- Easy to maintain
- Centralized sensor definitions

## Migration Steps

### Step 1: Keep Using Legacy Code (Immediate)

The system is **backward compatible**. Your existing code continues to work:

```python
# Old code still works
from src.common import KITTI_SensorType, SensorType

sensor_type = KITTI_SensorType.OXTS_IMU
if SensorType.is_imu_data(sensor_type):
    pass
```

### Step 2: Use Legacy Compatibility Layer (Gradual)

Use new metadata with legacy-style functions:

```python
from src.common import (
    create_sensor_metadata_from_config,
    is_imu_data_legacy,
    is_time_update_legacy
)

# Create new metadata
metadata = create_sensor_metadata_from_config("KITTI", "OXTS_IMU")

# Use with legacy functions
if is_imu_data_legacy(metadata):
    pass

if is_time_update_legacy(metadata):
    pass
```

### Step 3: Migrate to New API (Recommended)

Fully embrace the new system:

```python
from src.common import (
    SensorCategory,
    create_sensor_metadata_from_config
)

# Create metadata
metadata = create_sensor_metadata_from_config("KITTI", "OXTS_IMU")

# Use category-based checks
if metadata.category == SensorCategory.IMU:
    pass

if metadata.is_time_update():
    pass
```

### Step 4: Update Dataset Classes

Replace `Sensor` with `RefactoredSensor`:

```python
from src.internal.dataset.refactored_sensor import RefactoredSensor
from src.common import create_sensor_metadata_from_config

# Old way
sensor = Sensor(
    type=KITTI_SensorType.OXTS_IMU,
    dataset=reader,
    output_queue=queue
)

# New way
metadata = create_sensor_metadata_from_config("KITTI", "OXTS_IMU")
sensor = RefactoredSensor(
    metadata=metadata,
    dataset_reader=reader,
    output_queue=queue
)
```

### Step 5: Update Queue Processing

Use `SensorQueueItem` instead of tuples:

```python
# Old way
timestamp, sensor_data = queue.get()
sensor_type = sensor_data.type

# New way
item = queue.get()  # SensorQueueItem
timestamp = item.timestamp
metadata = item.metadata
data = item.data
```

## Adding a New Dataset

### 1. Add to `configs/sensor_registry.yaml`

```yaml
datasets:
  MyNewDataset:
    sensors:
      - id: MY_IMU
        category: IMU
        frequency: 200
        priority: 1
        coordinate_frame: IMU
        reader_class: MyIMUDataReader
```

### 2. Create Reader Classes

```python
class MyIMUDataReader:
    def __init__(self, root_path, variant, **kwargs):
        self.root_path = root_path
        self.variant = variant
    
    def __iter__(self):
        # Read data from files
        for timestamp, data in self._read_data():
            yield timestamp, data
```

### 3. Register Readers

```python
from src.common import get_sensor_registry

registry = get_sensor_registry()
registry.register_reader("MyNewDataset", "MY_IMU", MyIMUDataReader)
```

### 4. Use It!

```python
from src.common import create_sensor_metadata_from_config

metadata = create_sensor_metadata_from_config("MyNewDataset", "MY_IMU")
# Now use this metadata throughout your pipeline
```

## Benefits Realized

### Before: Adding UAV_IMU1

```python
# Had to modify enum
class UAV_SensorType(IntEnum):
    VOXL_IMU0 = auto()
    VOXL_IMU1 = auto()  # Added this

# Had to update checks
@staticmethod
def is_imu_data(t):
    return t.name in [
        SensorType.VOXL_IMU0.name,
        SensorType.VOXL_IMU1.name,  # Added this
        # ... all other IMUs
    ]
```

### After: Adding UAV_IMU1

```yaml
# Just add to config file
datasets:
  UAV:
    sensors:
      - id: VOXL_IMU1
        category: IMU  # Already handled
        frequency: 200
        priority: 1
```

**That's it!** No Python code changes needed.

## Common Patterns

### Pattern 1: Filter Sensors by Category

```python
imu_sensors = [
    metadata for metadata in all_sensors
    if metadata.category == SensorCategory.IMU
]
```

### Pattern 2: Different Processing for Same Category

```python
if metadata.category == SensorCategory.IMU:
    # Distinguish by dataset or sensor_id
    if metadata.dataset == "UAV" and metadata.sensor_id == "VOXL_IMU0":
        # Special processing for VOXL IMU0
        pass
    else:
        # Standard IMU processing
        pass
```

### Pattern 3: Multiple IMU Fusion

```python
# Get all configured IMUs
imu_metadatas = [
    create_sensor_metadata_from_config("UAV", "VOXL_IMU0"),
    create_sensor_metadata_from_config("UAV", "PX4_IMU0")
]

# Each has unique full_name
for metadata in imu_metadatas:
    print(metadata.full_name)  # "UAV_VOXL_IMU0", "UAV_PX4_IMU0"
    
# Process data from both
for imu_metadata in imu_metadatas:
    if item.metadata.full_name == imu_metadata.full_name:
        # Process this specific IMU's data
        pass
```

## Testing

Run the examples to verify the system:

```bash
python -m examples.refactored_sensor_examples
```

This will demonstrate:
- Creating metadata from config
- Using the registry
- Priority queue ordering
- Backward compatibility
- Adding new datasets
- Multiple IMU configurations

## FAQ

**Q: Do I have to migrate all code at once?**
A: No! The system is backward compatible. Migrate gradually.

**Q: What if I need dataset-specific behavior?**
A: Check `metadata.dataset` or `metadata.sensor_id` for specific handling.

**Q: Can I still use IntEnum values for ordering?**
A: Use `metadata.priority` instead. It's explicit and configurable.

**Q: How do I add a new sensor category?**
A: Add to `SensorCategory` enum in `src/common/sensor_metadata.py`.

**Q: Where are reader classes registered?**
A: In `src/internal/dataset/refactored_sensor.py` via `register_all_readers()`.

## Next Steps

1. Run examples: `python -m examples.refactored_sensor_examples`
2. Review `configs/sensor_registry.yaml`
3. Start using `create_sensor_metadata_from_config()` in new code
4. Gradually migrate dataset classes to use `RefactoredSensor`
5. Update processing code to use `SensorQueueItem`

## Files Created

```
src/common/
├── sensor_metadata.py          # Core classes (SensorCategory, SensorMetadata, etc.)
├── sensor_config_loader.py     # Config loading utilities
└── sensor_legacy.py            # Backward compatibility layer

src/internal/dataset/
└── refactored_sensor.py        # RefactoredSensor class and registration

configs/
└── sensor_registry.yaml        # Sensor definitions

examples/
└── refactored_sensor_examples.py  # Usage examples

docs/
└── SENSOR_REFACTORING_GUIDE.md    # This file
```

## Support

For questions or issues, refer to:
- Examples: `examples/refactored_sensor_examples.py`
- This guide: `docs/SENSOR_REFACTORING_GUIDE.md`
- Code comments in `src/common/sensor_metadata.py`
