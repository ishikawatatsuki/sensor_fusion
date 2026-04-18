# Visual Odometry Configuration Guide

## Overview

The sensor fusion system now supports flexible visual odometry (VO) configuration with automatic detection and fallback capabilities:

- **Pre-computed VO**: Load previously computed VO estimates from disk for faster processing
- **Runtime VO**: Compute VO on-the-fly from camera images
- **Automatic Fallback**: Automatically fall back to runtime computation if pre-computed data is not available
- **Camera Validation**: Ensures camera sensors are available when needed for runtime VO

## Quick Start

### Basic Configuration

To use visual odometry with automatic pre-computed/runtime selection:

```yaml
# In your config file (e.g., euroc_config.yaml)
dataset:
  type: euroc
  root_path: ./data/EuRoC
  variant: "01"
  sensors:
    euroc_stereo:
      selected: True  # Required for runtime VO
      dropout_ratio: 0
      window_size: 1
      args:
        frequency: 30
    
    euroc_vo_precomputed:
      selected: True
      dropout_ratio: 0
      window_size: 1
      args:
        frequency: 30
        vo_suffix: "_2d3d"  # Corresponds to estimation method
        estimation_type: "2d3d"  # For runtime computation
        use_precomputed: True  # Try to load pre-computed first
```

## Configuration Options

### VO Sensor Types

#### 1. `euroc_vo` (Original VO Data)
Uses VO data from the dataset itself (if available).

```yaml
euroc_vo:
  selected: True
  dropout_ratio: 0
  window_size: 1
  args:
    frequency: 30
```

#### 2. `euroc_vo_precomputed` (Flexible Pre-computed/Runtime)
**RECOMMENDED**: Attempts to load pre-computed VO, falls back to runtime if unavailable.

```yaml
euroc_vo_precomputed:
  selected: True
  dropout_ratio: 0
  window_size: 1
  args:
    frequency: 30
    vo_suffix: "_2d3d"  # Options: "", "_2d3d", "_hybrid"
    estimation_type: "2d3d"  # Options: "2d2d", "2d3d", "hybrid"
    use_precomputed: True  # Set to False to force runtime computation
```

### VO Estimation Types

The `estimation_type` parameter determines which VO algorithm to use:

- **`2d2d` (Epipolar Geometry)**: Uses Essential matrix decomposition
  - Faster but less accurate
  - Good for high frame rates
  
- **`2d3d` (PnP)**: Uses Perspective-n-Point with depth estimation
  - More accurate
  - Requires depth estimation (slower)
  - **RECOMMENDED for most use cases**
  
- **`hybrid`**: Combines 2D-2D and 2D-3D methods
  - Best accuracy
  - Slowest performance

### VO Directory Suffixes

Pre-computed VO data is stored in directories with specific suffixes:

| Estimation Type | Directory Suffix | Example Path |
|----------------|------------------|--------------|
| 2d2d (Epipolar) | `` (empty) | `data/EuRoC/vo_pose_estimates/` |
| 2d3d (PnP) | `_2d3d` | `data/EuRoC/vo_pose_estimates_2d3d/` |
| hybrid | `_hybrid` | `data/EuRoC/vo_pose_estimates_hybrid/` |
| stereo | `_stereo` | `data/EuRoC/vo_pose_estimates_stereo/` |

The `vo_suffix` parameter should match the estimation type used to generate the pre-computed data.

## Pre-computed VO Data

### Directory Structure

Expected directory structure for pre-computed VO data:

```
EuRoC/
├── vo_pose_estimates/           # 2d2d method
│   ├── MH_01_easy_poses.txt
│   ├── MH_02_easy_poses.txt
│   └── MH_03_medium_poses.txt
├── vo_pose_estimates_2d3d/      # 2d3d method
│   ├── MH_01_easy_poses.txt
│   └── ...
└── vo_pose_estimates_hybrid/    # Hybrid method
    └── ...

KITTI/
├── vo_pose_estimates/           # 2d2d method
│   ├── 00.txt
│   ├── 01.txt
│   └── ...
└── vo_pose_estimates_2d3d/      # 2d3d method
    └── ...
```

### File Format

Pre-computed VO files should contain transformation matrices in the format:

```
timestamp r11 r12 r13 tx r21 r22 r23 ty r31 r32 r33 tz
```

Where:
- `timestamp`: Frame timestamp (nanoseconds for EuRoC, frame index for KITTI)
- `r11...r33`: 3x3 rotation matrix elements (row-major)
- `tx, ty, tz`: Translation vector

### Generating Pre-computed VO Data

To generate pre-computed VO data, run your VO pipeline with export enabled:

```yaml
visual_odometry:
  type: monocular
  estimator: 2d3d  # or 2d2d, hybrid
  export_vo_data: True
  export_vo_data_path: ./data/EuRoC/vo_pose_estimates_2d3d/
```

Or use the provided scripts:

```bash
# EuRoC dataset
bash scripts/vo_2d3d_pose_estimate.sh

# KITTI dataset  
bash scripts/vo_2d3d_pose_estimate_local.sh
```

## Runtime VO Configuration

When computing VO at runtime, ensure:

1. **Camera sensor is selected**: The system needs image data
2. **Visual odometry parameters are configured**: Set in `visual_odometry` section

```yaml
dataset:
  sensors:
    euroc_stereo:
      selected: True  # REQUIRED for runtime VO
    
    euroc_vo_precomputed:
      selected: True
      args:
        use_precomputed: False  # Force runtime computation

visual_odometry:
  type: monocular
  estimator: 2d3d  # Must match estimation_type in sensor config
  camera_id: left
  depth_estimator: depth_anything
  feature_detector: SIFT
  feature_matcher: BF
  params:
    confidence: 0.90
    ransac_reproj_threshold: 1.5
    matching_threshold: 0.45
    max_features: 1500
```

## Usage Examples

### Example 1: Use Pre-computed VO (with fallback)

```yaml
dataset:
  type: euroc
  root_path: ./data/EuRoC
  variant: "01"
  sensors:
    euroc_stereo:
      selected: True  # Needed for fallback
    euroc_vo_precomputed:
      selected: True
      args:
        vo_suffix: "_2d3d"
        estimation_type: "2d3d"
        use_precomputed: True  # Try pre-computed first
```

**Behavior**:
- If `./data/EuRoC/vo_pose_estimates_2d3d/MH_01_easy_poses.txt` exists → Use it
- Otherwise → Compute VO at runtime using 2d3d method

### Example 2: Force Runtime Computation

```yaml
dataset:
  sensors:
    euroc_stereo:
      selected: True
    euroc_vo_precomputed:
      selected: True
      args:
        estimation_type: "2d3d"
        use_precomputed: False  # Always compute at runtime
```

**Behavior**:
- Always computes VO at runtime, even if pre-computed data exists
- Useful for testing or when you want fresh computations

### Example 3: Use Original Dataset VO

```yaml
dataset:
  sensors:
    euroc_vo:  # Use built-in VO from dataset
      selected: True
```

**Behavior**:
- Loads VO data directly from dataset
- Only works if dataset includes VO data (some EuRoC sequences don't)

### Example 4: KITTI with Pre-computed VO

```yaml
dataset:
  type: kitti
  root_path: ./data/KITTI
  variant: "09"
  sensors:
    kitti_stereo:
      selected: True
    kitti_vo_precomputed:
      selected: True
      args:
        vo_suffix: "_2d3d"
        estimation_type: "2d3d"
        use_precomputed: True
```

**Expected path**: `./data/KITTI/vo_pose_estimates_2d3d/09.txt`

## Validation and Diagnostics

### Checking VO Configuration

Use the `VOConfigHelper` to validate your configuration:

```python
from src.utils.vo_config_helper import check_and_configure_vo

result = check_and_configure_vo(
    sensors=config.dataset.sensors,
    root_path=config.dataset.root_path,
    variant=config.dataset.variant,
    dataset_type=config.dataset.type
)

print(f"VO Mode: {result['vo_mode']}")
print(f"VO Path: {result['vo_path']}")
print(f"Camera Available: {result['camera_available']}")
```

### Runtime Logs

The system provides informative logs:

```
[INFO] VO configured: mode=precomputed, estimation_type=2d3d
[INFO] Pre-computed VO data found: ./data/EuRoC/vo_pose_estimates_2d3d/MH_01_easy_poses.txt
```

Or if falling back:

```
[INFO] Pre-computed VO data not found at ./data/EuRoC/vo_pose_estimates_2d3d/MH_01_easy_poses.txt
[INFO] Will compute VO at runtime using 2d3d method.
[WARNING] Camera sensor 'euroc_stereo' is required for runtime VO computation.
```

### VO Status Properties

Check VO mode programmatically:

```python
# In DatasetConfig
print(config.dataset.vo_mode)  # 'precomputed' or 'runtime'
print(config.dataset.use_precomputed_vo)  # True/False
print(config.dataset.use_runtime_vo)  # True/False

# In VisualOdometry instance
print(vo.is_using_precomputed)  # True if using pre-computed
print(vo.is_using_runtime)  # True if computing at runtime
```

## Performance Considerations

| Mode | Speed | Accuracy | Disk Space | Notes |
|------|-------|----------|------------|-------|
| Pre-computed | ⚡⚡⚡ Fast | Depends on generation | High | Best for repeated experiments |
| Runtime 2d2d | ⚡⚡ Moderate | Good | Low | Good balance |
| Runtime 2d3d | ⚡ Slow | Very Good | Low | Recommended for accuracy |
| Runtime Hybrid | 🐌 Very Slow | Best | Low | Use only when accuracy critical |

**Recommendations**:
- **Development/Testing**: Use runtime computation for flexibility
- **Production/Experiments**: Generate and use pre-computed VO
- **First-time users**: Start with `2d3d` pre-computed with fallback

## Troubleshooting

### Issue: "Pre-computed VO data not found"

**Solution**: Check that:
1. Path exists: `ls data/EuRoC/vo_pose_estimates_2d3d/`
2. File exists for your variant: `ls data/EuRoC/vo_pose_estimates_2d3d/MH_01_easy_poses.txt`
3. `vo_suffix` matches directory name

### Issue: "Camera sensor required for runtime VO"

**Solution**: Enable a camera sensor:
```yaml
euroc_stereo:  # or kitti_stereo, voxl_stereo
  selected: True
```

### Issue: VO estimates seem incorrect

**Solution**: 
1. Check estimation type matches pre-computed data generation method
2. Verify VO parameters in `visual_odometry` section
3. Try regenerating pre-computed data
4. Compare with ground truth visualization

### Issue: Slow performance

**Solutions**:
- Generate pre-computed VO data for faster processing
- Use 2d2d instead of 2d3d estimation
- Reduce `max_features` parameter
- Enable `use_precomputed: True`

## Advanced Usage

### Custom VO Paths

Modify the path template in `sensor_schemas.yaml`:

```yaml
euroc_vo_precomputed:
  data_source:
    path_template: "{root_path}/custom_vo_dir{vo_suffix}/MH_{variant}_easy_poses.txt"
```

### Multiple VO Sources

You can configure multiple VO sensors and select them programmatically:

```yaml
sensors:
  euroc_vo_precomputed_2d3d:
    selected: True
    args:
      vo_suffix: "_2d3d"
      estimation_type: "2d3d"
  
  euroc_vo_precomputed_hybrid:
    selected: False  # Disabled by default
    args:
      vo_suffix: "_hybrid"
      estimation_type: "hybrid"
```

### Programmatic Configuration

```python
from src.utils.vo_config_helper import VOConfigHelper

# Check availability
exists, path = VOConfigHelper.check_vo_data_exists(
    root_path="./data/EuRoC",
    variant="01",
    dataset_type="euroc",
    estimation_type="2d3d"
)

if exists:
    print(f"Using pre-computed VO: {path}")
else:
    print("Will compute at runtime")
```

## Summary

The flexible VO configuration system provides:

✅ **Automatic detection** of pre-computed VO data  
✅ **Fallback to runtime** computation when needed  
✅ **Camera validation** ensures dependencies are met  
✅ **Multiple estimation methods** (2d2d, 2d3d, hybrid)  
✅ **Performance optimization** through pre-computation  
✅ **Easy configuration** via YAML files  

For most use cases, use `euroc_vo_precomputed` or `kitti_vo_precomputed` with `use_precomputed: True` for the best balance of performance and flexibility.
