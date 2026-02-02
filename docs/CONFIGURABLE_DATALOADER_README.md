# Configuration-Driven Data Loader

## 🎯 Quick Start

Add a new dataset **without writing Python code**:

```yaml
# configs/sensor_schemas.yaml

sensor_schemas:
  my_imu:
    sensor_type: "MY_IMU"
    data_source:
      type: csv
      path_template: "{root_path}/sensor_{variant}.csv"
      delimiter: ","
      skip_header: 1
    fields:
      - name: timestamp
        columns: [0]
        type: float
      - name: accel
        columns: [1, 2, 3]
        type: array
      - name: gyro
        columns: [4, 5, 6]
        type: array
    output_fields: [timestamp, accel, gyro]

dataset_sensor_mapping:
  my_dataset:
    imu: my_imu
```

Use it immediately:

```python
from src.internal.dataset.configurable_adapter import ConfigurableDatasetAdapter

adapter = ConfigurableDatasetAdapter('my_dataset')
reader = adapter.create_sensor_reader('imu', root_path='./data', variant='001')

for data in reader:
    print(f"t={data.timestamp}: accel={data.accel}, gyro={data.gyro}")
```

## 📁 Files Created

```
src/internal/dataset/
├── generic_reader.py          # Core implementation (400+ lines)
└── configurable_adapter.py    # Integration adapter (200+ lines)

configs/
└── sensor_schemas.yaml        # Sensor definitions ← Edit this!

docs/
├── CONFIGURABLE_DATALOADER_SOLUTION.md  # Solution overview
└── CONFIGURABLE_DATALOADER_GUIDE.md     # Complete guide

examples/
└── configurable_dataloader_examples.py  # 7 working examples

test/
└── test_configurable_dataloader.py      # Unit tests
```

## 🚀 Quick Examples

### Run Interactive Examples

```bash
python examples/configurable_dataloader_examples.py
```

Options:
1. Basic Usage
2. From Config File
3. Compare with Legacy
4. Visualize Data
5. Multi-Sensor Fusion
6. Custom Dataset
7. Integration Guide

### Run Tests

```bash
python -m pytest test/test_configurable_dataloader.py -v
```

## 📖 Documentation

- **[Solution Summary](./CONFIGURABLE_DATALOADER_SOLUTION.md)** - Overview and benefits
- **[Complete Guide](./CONFIGURABLE_DATALOADER_GUIDE.md)** - API docs, examples, migration

## ✨ Key Features

✅ **Config-Only Dataset Addition** - No Python code needed  
✅ **Flexible Field Mapping** - Map any CSV structure  
✅ **Custom Transformations** - Define in YAML  
✅ **Backward Compatible** - Drop-in replacements  
✅ **Built-in Features** - Rolling average, noise, filtering  

## 📝 Example Configs Already Defined

- ✅ EuRoC IMU, Leica, Stereo, Ground Truth
- ✅ KITTI OXTS IMU, GPS, Ground Truth
- ✅ UAV PX4 IMU, GPS

Add yours in [`configs/sensor_schemas.yaml`](../configs/sensor_schemas.yaml)!

## 🔄 Migration Path

1. **Add schemas** for existing sensors (no code changes)
2. **Test** new readers alongside old ones
3. **Update** Dataset classes to use adapter
4. **Remove** old reader classes

## 💡 Need Help?

See [Complete Guide](./CONFIGURABLE_DATALOADER_GUIDE.md) for:
- Detailed examples
- Advanced features
- Troubleshooting
- API reference

---

**Before:** Write 50-100 line Python class for each sensor  
**After:** Add 10-20 line YAML config 🎉
