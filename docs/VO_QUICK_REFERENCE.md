# Visual Odometry Configuration - Quick Reference

## 🎯 What's New

The sensor fusion system now supports **flexible visual odometry configuration** with automatic fallback:

- ✅ Load pre-computed VO estimates (fast)
- ✅ Compute VO at runtime (flexible)
- ✅ Automatic fallback when pre-computed data is missing
- ✅ Camera dependency validation

## 🚀 Quick Start

### Option 1: Use Pre-computed VO (Recommended)

**Configuration** (`configs/euroc_config.yaml`):
```yaml
dataset:
  sensors:
    euroc_stereo:
      selected: True  # Required for fallback
    
    euroc_vo_precomputed:
      selected: True
      args:
        vo_suffix: "_2d3d"
        estimation_type: "2d3d"
        use_precomputed: True
```

**What happens**:
1. System checks for pre-computed VO at: `data/EuRoC/vo_pose_estimates_2d3d/MH_01_easy_poses.txt`
2. If found → Loads it (very fast ⚡)
3. If not found → Computes VO at runtime using 2D-3D method

### Option 2: Always Compute at Runtime

```yaml
euroc_vo_precomputed:
  selected: True
  args:
    estimation_type: "2d3d"
    use_precomputed: False  # Force runtime computation
```

## 📁 Directory Structure

Pre-computed VO files should be placed in:

```
data/
├── EuRoC/
│   ├── vo_pose_estimates/           # 2D-2D method
│   ├── vo_pose_estimates_2d3d/      # 2D-3D method ⭐ recommended
│   │   ├── MH_01_easy_poses.txt
│   │   ├── MH_02_easy_poses.txt
│   │   └── MH_03_medium_poses.txt
│   └── vo_pose_estimates_hybrid/    # Hybrid method
└── KITTI/
    ├── vo_pose_estimates/
    ├── vo_pose_estimates_2d3d/      # 2D-3D method ⭐ recommended
    │   ├── 00.txt
    │   ├── 09.txt
    │   └── 10.txt
    └── vo_pose_estimates_hybrid/
```

## 🔧 Estimation Methods

| Method | Speed | Accuracy | Use Case |
|--------|-------|----------|----------|
| **2d2d** | ⚡⚡⚡ Fast | Good | High frame rates |
| **2d3d** | ⚡⚡ Moderate | Very Good | **Recommended** ⭐ |
| **hybrid** | ⚡ Slow | Excellent | Max accuracy needed |

## 📊 Generating Pre-computed VO

### For EuRoC:
```bash
bash scripts/vo_2d3d_pose_estimate.sh
```

### For KITTI:
```bash
bash scripts/vo_2d3d_pose_estimate_local.sh
```

Or enable export in your config:
```yaml
visual_odometry:
  export_vo_data: True
  export_vo_data_path: ./data/EuRoC/vo_pose_estimates_2d3d/
```

## 💡 Examples

### Check if VO data exists:
```python
from src.utils.vo_config_helper import VOConfigHelper

exists, path = VOConfigHelper.check_vo_data_exists(
    root_path='./data/EuRoC',
    variant='01',
    dataset_type='euroc',
    estimation_type='2d3d'
)

print(f"VO data exists: {exists}")
print(f"Path: {path}")
```

### Load configuration and check VO mode:
```python
from src.internal.extended_common.extended_config import ExtendedConfig

config = ExtendedConfig('./configs/euroc_config.yaml')

print(f"VO Mode: {config.dataset.vo_mode}")
print(f"VO Path: {config.dataset.vo_path}")
print(f"Using pre-computed: {config.dataset.use_precomputed_vo}")
print(f"Using runtime: {config.dataset.use_runtime_vo}")
```

### Run examples:
```bash
python examples/vo_configuration_examples.py
```

## 📚 Documentation

- **Complete Guide**: [`docs/VISUAL_ODOMETRY_CONFIGURATION_GUIDE.md`](docs/VISUAL_ODOMETRY_CONFIGURATION_GUIDE.md)
- **Summary**: [`docs/VO_CONFIGURATION_SUMMARY.md`](docs/VO_CONFIGURATION_SUMMARY.md)
- **Sensor Schemas**: [`configs/sensor_schemas.yaml`](configs/sensor_schemas.yaml)

## 🔍 Troubleshooting

### "Pre-computed VO data not found"
- Check file path: `ls data/EuRoC/vo_pose_estimates_2d3d/`
- Generate data: `bash scripts/vo_2d3d_pose_estimate.sh`
- Or set `use_precomputed: False` to always compute at runtime

### "Camera sensor required for runtime VO"
- Enable camera sensor: `euroc_stereo` or `kitti_stereo`
```yaml
euroc_stereo:
  selected: True
```

### Slow performance
- Generate and use pre-computed VO (10x faster)
- Or switch to `2d2d` estimation method (faster but less accurate)

## ⚙️ Configuration Parameters

| Parameter | Values | Default | Description |
|-----------|--------|---------|-------------|
| `use_precomputed` | `True`/`False` | `True` | Try to load pre-computed data |
| `vo_suffix` | `""`, `"_2d3d"`, `"_hybrid"` | `"_2d3d"` | Directory suffix for VO data |
| `estimation_type` | `"2d2d"`, `"2d3d"`, `"hybrid"` | `"2d3d"` | Algorithm for runtime computation |

## 🎓 Best Practices

1. **Development**: Use runtime computation for flexibility
   ```yaml
   use_precomputed: False
   ```

2. **Production/Experiments**: Generate and use pre-computed VO
   ```yaml
   use_precomputed: True
   vo_suffix: "_2d3d"
   ```

3. **Always enable camera sensor** for automatic fallback:
   ```yaml
   euroc_stereo:
     selected: True
   ```

4. **Use 2D-3D estimation** for best balance of speed and accuracy:
   ```yaml
   estimation_type: "2d3d"
   ```

## 📝 Migration from Old Configuration

### Before:
```yaml
euroc_vo:
  selected: True
```

### After:
```yaml
euroc_vo_precomputed:
  selected: True
  args:
    vo_suffix: "_2d3d"
    estimation_type: "2d3d"
    use_precomputed: True
```

**Benefits**:
- ✅ Automatic fallback
- ✅ Better performance
- ✅ More flexibility
- ✅ Camera validation

## 🚦 Status Indicators

When running, you'll see log messages like:

**Using pre-computed VO**:
```
[INFO] VO configured: mode=precomputed, estimation_type=2d3d
[INFO] Pre-computed VO data found: ./data/EuRoC/vo_pose_estimates_2d3d/MH_01_easy_poses.txt
```

**Falling back to runtime**:
```
[INFO] Pre-computed VO data not found at ./data/EuRoC/vo_pose_estimates_2d3d/MH_01_easy_poses.txt
[INFO] Will compute VO at runtime using 2d3d method.
[INFO] Camera sensor 'euroc_stereo' is available for runtime VO computation.
```

---

**Questions?** See the [complete guide](docs/VISUAL_ODOMETRY_CONFIGURATION_GUIDE.md) for detailed information.
