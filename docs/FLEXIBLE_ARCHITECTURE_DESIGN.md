# Configuration-Driven Sensor Fusion Architecture: From Scratch Design

**Date:** February 2026  
**Problem:** Hard-coded sensor type dependencies limit flexibility  
**Solution:** Role-based, configuration-driven architecture

---

## Table of Contents

1. [Problem Analysis](#problem-analysis)
2. [Proposed Architecture](#proposed-architecture)
3. [Sensor Role System](#sensor-role-system)
4. [Configuration Schema](#configuration-schema)
5. [Implementation Design](#implementation-design)
6. [Example Use Cases](#example-use-cases)
7. [Migration Path](#migration-path)

---

## Problem Analysis

### Current Architecture Issues

**Current Code Pattern:**
```python
# datatypes.py - Hard-coded sensor types
class KITTI_SensorType(IntEnum):
    OXTS_IMU = auto()
    OXTS_GPS = auto()
    # ... 8 more types

class UAV_SensorType(IntEnum):
    VOXL_IMU0 = auto()
    VOXL_IMU1 = auto()
    PX4_IMU0 = auto()
    # ... 17 more types

class SensorType(KITTI_SensorType, UAV_SensorType, EuRoC_SensorType):
    @staticmethod
    def is_imu_data(t):
        return t.name in [
            SensorType.OXTS_IMU.name,
            SensorType.VOXL_IMU0.name,
            SensorType.PX4_IMU0.name,
            # ... hard-coded list
        ]
```

**Problems:**

1. ❌ **New Dataset = Code Changes**: Adding a new dataset requires editing `datatypes.py`
2. ❌ **Hard-coded Logic**: Sensor classification logic scattered across codebase
3. ❌ **Type Explosion**: Every sensor variant needs its own enum value
4. ❌ **Inflexible**: Can't change sensor roles without code changes
5. ❌ **Multiple Instances**: Awkward handling (VOXL_IMU0, VOXL_IMU1, etc.)
6. ❌ **Testing Complexity**: Hard to test with custom/synthetic sensors

### What We Need

✅ **Configuration-Driven**: Define sensors and their roles in YAML  
✅ **Extensible**: Add new datasets without code changes  
✅ **Flexible Roles**: Same sensor can have different roles in different contexts  
✅ **Multi-Instance**: Support N sensors of same type naturally  
✅ **Testable**: Easy to create test configurations  
✅ **Clear Separation**: Sensor identity ≠ Sensor role ≠ Processing strategy

---

## Proposed Architecture

### Core Concepts

```
┌─────────────────────────────────────────────────────────────────┐
│                    CONFIGURATION LAYER                          │
│  Defines: What sensors exist, what they measure, how to read   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                     SEMANTIC LAYER                              │
│  Sensor → Role Mapping (What does this sensor do?)             │
│  - Prediction source (IMU, odometry, motion model)             │
│  - Measurement source (GPS, VO, magnetometer)                  │
│  - Reference (ground truth)                                     │
│  - Auxiliary (visualization, logging)                          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    PROCESSING LAYER                             │
│  Role → Strategy Mapping (How to process this data?)           │
│  - PredictionStrategy                                           │
│  - MeasurementStrategy                                          │
│  - FusionStrategy                                               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      FILTER LAYER                               │
│  Executes: predict(), update(), fuse()                         │
└─────────────────────────────────────────────────────────────────┘
```

### Key Design Principles

1. **Separation of Concerns**
   - **Identity**: What is this sensor? (IMU, GPS, Camera)
   - **Role**: What does it contribute? (Prediction, Measurement, Reference)
   - **Strategy**: How to process it? (Specific algorithm/model)

2. **Composition over Inheritance**
   - Don't use enums for types
   - Use configuration to compose behavior

3. **Data-Driven Dispatch**
   - Configuration determines which processing strategy to use
   - No hard-coded conditionals based on sensor type

---

## Sensor Role System

### Role Taxonomy

Instead of hard-coding sensor types, define **semantic roles**:

```yaml
# Sensor Role Definitions
sensor_roles:
  
  # Prediction Sources (Time Update)
  prediction:
    description: "Sensors that drive state prediction"
    update_type: "time_update"
    examples:
      - IMU (acceleration, angular velocity)
      - Wheel odometry
      - Motor commands
      - Motion model
    
  # Measurement Sources (Measurement Update)
  measurement:
    description: "Sensors that provide observations"
    update_type: "measurement_update"
    subtypes:
      - position        # GPS, UWB, motion capture
      - velocity        # Doppler, optical flow
      - orientation     # Magnetometer, star tracker
      - pose            # Visual odometry, SLAM
      - constraint      # Zero-velocity, planar motion
    
  # Reference Sources
  reference:
    description: "Ground truth for evaluation"
    update_type: "none"
    examples:
      - Motion capture ground truth
      - RTK GPS
      - Simulator truth
  
  # Auxiliary Sources
  auxiliary:
    description: "Supporting data (not fused)"
    update_type: "none"
    examples:
      - Camera images for visualization
      - Diagnostic data
      - Metadata
```

### Sensor Instance Configuration

Each sensor instance is configured with:

```yaml
sensors:
  imu_main:                           # Unique instance ID
    identity:
      type: "IMU"                     # Sensor type (for data parsing)
      model: "ADIS16448"              # Hardware model
      location: "body_center"         # Physical location
    
    role:
      primary: "prediction"           # Primary role
      provides:                       # What measurements it provides
        - linear_acceleration
        - angular_velocity
    
    processing:
      strategy: "imu_prediction"      # Which processing strategy
      frame: "body"                   # Reference frame
      noise_model: "gaussian_white"   # Noise characteristics
    
    data_source:
      reader: "csv"                   # How to read data
      path: "imu0/data.csv"
      fields: [...]                   # Field mapping
  
  # Multiple IMUs - each with unique ID
  imu_backup:
    identity:
      type: "IMU"
      model: "MPU9250"
      location: "left_wing"
    
    role:
      primary: "measurement"          # Different role!
      provides:
        - linear_acceleration
        - angular_velocity
      fusion_mode: "complementary"    # How to combine with imu_main
    
    processing:
      strategy: "imu_measurement"     # Different strategy
      weight: 0.3                     # Lower trust
```

### Key Insight: Same Sensor Type, Different Roles

```yaml
# Example: Two IMUs with different roles

# Primary IMU - drives prediction
imu_primary:
  identity: { type: "IMU" }
  role: { primary: "prediction" }
  processing: { strategy: "imu_ekf_prediction" }

# Secondary IMU - used for measurement updates
imu_secondary:
  identity: { type: "IMU" }
  role: { primary: "measurement" }
  processing: { strategy: "imu_innovation_check" }

# The filter doesn't care about "type", only "role"!
```

---

## Configuration Schema

### Complete Sensor Configuration Example

```yaml
# config/sensors/euroc_mh01_sensors.yaml

pipeline:
  name: "EuRoC MH_01 Multi-IMU Fusion"
  
  # Define all sensor instances
  sensors:
    
    # Primary IMU - prediction source
    imu_main:
      identity:
        type: "IMU"
        model: "ADIS16448"
        frame_id: "imu0"
      
      role:
        primary: "prediction"
        provides:
          - field: "linear_acceleration"
            axes: [x, y, z]
            unit: "m/s^2"
          - field: "angular_velocity"
            axes: [x, y, z]
            unit: "rad/s"
      
      processing:
        strategy: "imu_motion_prediction"
        motion_model: "velocity_model"
        compensate_gravity: true
        bias_estimation: true
        
        noise_model:
          type: "gaussian"
          parameters:
            gyro_noise_density: 2.36e-4
            gyro_random_walk: 3.33e-3
            accel_noise_density: 2.26e-3
            accel_random_walk: 3.19e-2
      
      data_source:
        reader_type: "csv"
        path_template: "{dataset_root}/imu0/data.csv"
        timestamp_column: 0
        timestamp_scale: 1e-9  # ns to s
        field_mapping:
          angular_velocity: [1, 2, 3]
          linear_acceleration: [4, 5, 6]
      
      quality:
        dropout_handling: "forward_propagate"
        outlier_detection: "mahalanobis"
        threshold: 5.0
    
    # GPS/Leica - position measurement
    gnss:
      identity:
        type: "GNSS"
        model: "Leica_MS50"
        frame_id: "leica0"
      
      role:
        primary: "measurement"
        provides:
          - field: "position"
            axes: [x, y, z]
            unit: "m"
            reference_frame: "world"
      
      processing:
        strategy: "position_measurement"
        measurement_model: "gnss_position"
        innovation_gating: true
        
        noise_model:
          type: "gaussian"
          parameters:
            position_stddev: [0.01, 0.01, 0.02]  # x, y, z
      
      data_source:
        reader_type: "csv"
        path_template: "{dataset_root}/leica0/data.csv"
        timestamp_column: 0
        timestamp_scale: 1e-9
        field_mapping:
          position: [1, 2, 3]
      
      quality:
        min_satellites: 4
        max_hdop: 2.0
    
    # Visual Odometry - pose measurement
    visual_odometry:
      identity:
        type: "VISUAL_ODOMETRY"
        source: "stereo_camera"
        frame_id: "cam0"
      
      role:
        primary: "measurement"
        provides:
          - field: "relative_pose"
            components: [position, orientation]
          - field: "velocity"  # optional
            axes: [x, y, z]
      
      processing:
        strategy: "vo_pose_measurement"
        measurement_model: "relative_pose"
        
        # VO-specific parameters
        vo_config:
          estimator_type: "2d3d"
          feature_detector: "SIFT"
          min_features: 500
          tracking_quality_threshold: 0.7
        
        # When VO fails, what to do?
        failure_handling:
          mode: "skip"  # skip, predict_only, use_last
          max_consecutive_failures: 10
      
      data_source:
        # VO is computed online, not from file
        reader_type: "online_processor"
        input_topics:
          - "/camera/left/image"
          - "/camera/right/image"
      
      quality:
        min_inliers: 50
        max_reprojection_error: 2.0
    
    # Ground Truth - reference
    ground_truth:
      identity:
        type: "GROUND_TRUTH"
        source: "motion_capture"
      
      role:
        primary: "reference"
        provides:
          - field: "pose"
          - field: "velocity"
          - field: "biases"
      
      processing:
        strategy: "none"  # Not processed, just logged
      
      data_source:
        reader_type: "csv"
        path_template: "{dataset_root}/state_groundtruth_estimate0/data.csv"
        field_mapping:
          position: [1, 2, 3]
          quaternion: [4, 5, 6, 7]
          velocity: [8, 9, 10]
  
  # Define fusion strategy
  fusion:
    filter_type: "EKF"
    
    # Map sensor roles to filter operations
    role_mapping:
      prediction:
        sources: ["imu_main"]        # Which sensors drive prediction
        frequency: 200               # Hz
        
      measurement:
        sources: ["gnss", "visual_odometry"]
        asynchronous: true           # Process as they arrive
        
      reference:
        sources: ["ground_truth"]
        use_for: ["evaluation", "initialization"]
    
    # State vector definition
    state:
      components:
        - name: "position"
          size: 3
          initial_value: [0, 0, 0]
          initial_covariance: 0.01
        
        - name: "velocity"
          size: 3
          initial_value: [0, 0, 0]
          initial_covariance: 0.01
        
        - name: "orientation"
          size: 4  # quaternion
          initial_value: [1, 0, 0, 0]
          initial_covariance: 0.01
        
        - name: "gyro_bias"
          size: 3
          initial_value: [0, 0, 0]
          initial_covariance: 1e-6
        
        - name: "accel_bias"
          size: 3
          initial_value: [0, 0, 0]
          initial_covariance: 1e-4
```

### Multi-IMU Configuration Example

```yaml
# config/sensors/uav_multi_imu.yaml

pipeline:
  name: "UAV with 4 IMUs - Redundant Fusion"
  
  sensors:
    # Primary IMU - prediction
    imu_0:
      identity: { type: "IMU", model: "ICM20948", location: "center" }
      role:
        primary: "prediction"
        weight: 1.0
      processing:
        strategy: "imu_prediction"
    
    # Secondary IMUs - measurement + fault detection
    imu_1:
      identity: { type: "IMU", model: "ICM20948", location: "left" }
      role:
        primary: "measurement"
        weight: 0.3
        fault_detection: true
      processing:
        strategy: "imu_consistency_check"
        consistency_threshold: 0.5  # rad/s or m/s^2
    
    imu_2:
      identity: { type: "IMU", model: "MPU9250", location: "right" }
      role:
        primary: "measurement"
        weight: 0.3
        fault_detection: true
      processing:
        strategy: "imu_consistency_check"
    
    imu_3:
      identity: { type: "IMU", model: "BMI088", location: "rear" }
      role:
        primary: "auxiliary"  # Standby/logging only
      processing:
        strategy: "log_only"
  
  fusion:
    # Multi-sensor fusion strategy
    multi_imu_mode: "federated"  # federated, centralized, voting
    
    # How to handle IMU disagreement
    fault_detection:
      method: "chi_squared"
      threshold: 7.815  # 95% confidence, 3 DOF
      
      # If imu_0 fails, promote imu_1
      failover_sequence: ["imu_0", "imu_1", "imu_2"]
```

---

## Implementation Design

### 1. Sensor Registry (Replaces Enum)

**Old Way (Enum-based):**
```python
# Hard-coded enums
class KITTI_SensorType(IntEnum):
    OXTS_IMU = 1
    OXTS_GPS = 2
    # ...

# Hard-coded checks
if sensor_type == KITTI_SensorType.OXTS_IMU:
    do_prediction()
```

**New Way (Registry-based):**
```python
class SensorInstance:
    """Represents a single sensor instance"""
    
    def __init__(self, instance_id: str, config: dict):
        self.id = instance_id
        
        # Identity
        self.type = config['identity']['type']
        self.model = config['identity'].get('model')
        self.frame_id = config['identity'].get('frame_id')
        
        # Role
        self.role = SensorRole(config['role'])
        
        # Processing
        self.strategy = ProcessingStrategyFactory.create(
            config['processing']['strategy'],
            config['processing']
        )
        
        # Data source
        self.reader = DataReaderFactory.create(
            config['data_source']
        )
    
    def get_role(self) -> str:
        """Returns: 'prediction', 'measurement', 'reference', 'auxiliary'"""
        return self.role.primary
    
    def provides(self, field: str) -> bool:
        """Check if sensor provides a specific measurement"""
        return field in self.role.provides
    
    def process(self, raw_data):
        """Process raw data using assigned strategy"""
        return self.strategy.process(raw_data)


class SensorRegistry:
    """Global registry of all sensor instances"""
    
    def __init__(self):
        self._sensors: Dict[str, SensorInstance] = {}
    
    def register(self, sensor: SensorInstance):
        """Register a sensor instance"""
        self._sensors[sensor.id] = sensor
    
    def get(self, instance_id: str) -> SensorInstance:
        """Get sensor by ID"""
        return self._sensors.get(instance_id)
    
    def get_by_role(self, role: str) -> List[SensorInstance]:
        """Get all sensors with specific role"""
        return [s for s in self._sensors.values() if s.get_role() == role]
    
    def get_prediction_sources(self) -> List[SensorInstance]:
        """Get all sensors used for prediction"""
        return self.get_by_role('prediction')
    
    def get_measurement_sources(self) -> List[SensorInstance]:
        """Get all sensors used for measurement updates"""
        return self.get_by_role('measurement')
    
    @classmethod
    def from_config(cls, config_path: str):
        """Create registry from configuration file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        registry = cls()
        
        for sensor_id, sensor_config in config['pipeline']['sensors'].items():
            sensor = SensorInstance(sensor_id, sensor_config)
            registry.register(sensor)
        
        return registry


# Usage
registry = SensorRegistry.from_config('euroc_sensors.yaml')

# No enums! Just query by role
for sensor in registry.get_prediction_sources():
    print(f"Prediction sensor: {sensor.id}")

# Check capabilities
imu = registry.get('imu_main')
if imu.provides('angular_velocity'):
    print("IMU provides angular velocity")
```

### 2. Processing Strategy Pattern

**Old Way (Type-based dispatch):**
```python
# In filter class
def process_sensor_data(self, sensor_type, data):
    if SensorType.is_imu_data(sensor_type):
        self.predict(data)
    elif SensorType.is_gps_data(sensor_type):
        self.update_position(data)
    elif SensorType.is_vo_data(sensor_type):
        self.update_pose(data)
    # ... many more conditions
```

**New Way (Strategy pattern):**
```python
class ProcessingStrategy(ABC):
    """Base class for all processing strategies"""
    
    @abstractmethod
    def process(self, data, filter_state):
        """Process sensor data and update filter"""
        pass


class IMUPredictionStrategy(ProcessingStrategy):
    """Strategy for IMU-based prediction"""
    
    def __init__(self, config):
        self.motion_model = MotionModelFactory.create(
            config.get('motion_model', 'velocity_model')
        )
        self.compensate_gravity = config.get('compensate_gravity', True)
        self.noise_model = NoiseModel(config['noise_model'])
    
    def process(self, imu_data, filter_state):
        """Perform prediction step"""
        
        # Extract measurements
        accel = imu_data.linear_acceleration
        gyro = imu_data.angular_velocity
        dt = imu_data.dt
        
        # Compensate gravity if needed
        if self.compensate_gravity:
            accel = accel - filter_state.gravity_in_body_frame()
        
        # Create control input
        u = np.hstack([accel, gyro])
        
        # Compute process noise
        Q = self.noise_model.get_process_noise(dt)
        
        # Call filter's predict
        filter_state.predict(
            control_input=u,
            dt=dt,
            Q=Q,
            motion_model=self.motion_model
        )
        
        return filter_state


class GNSSMeasurementStrategy(ProcessingStrategy):
    """Strategy for GNSS position measurements"""
    
    def __init__(self, config):
        self.measurement_model = MeasurementModelFactory.create('position')
        self.noise_model = NoiseModel(config['noise_model'])
        self.innovation_gating = config.get('innovation_gating', True)
        self.threshold = config.get('threshold', 5.0)
    
    def process(self, gnss_data, filter_state):
        """Perform measurement update"""
        
        # Extract measurement
        z = gnss_data.position
        
        # Measurement noise
        R = self.noise_model.get_measurement_noise()
        
        # Innovation gating
        if self.innovation_gating:
            innovation = self.measurement_model.compute_innovation(z, filter_state)
            if not self.check_innovation(innovation, R):
                logger.warning("GNSS measurement rejected (innovation gate)")
                return filter_state
        
        # Call filter's update
        filter_state.update(
            measurement=z,
            R=R,
            measurement_model=self.measurement_model
        )
        
        return filter_state
    
    def check_innovation(self, innovation, R):
        """Chi-squared test for innovation"""
        mahalanobis = innovation.T @ np.linalg.inv(R) @ innovation
        return mahalanobis < self.threshold


class VOPoseMeasurementStrategy(ProcessingStrategy):
    """Strategy for visual odometry pose measurements"""
    
    def __init__(self, config):
        self.measurement_model = MeasurementModelFactory.create('relative_pose')
        self.failure_handling = config.get('failure_handling', {})
        self.quality_threshold = config['vo_config']['tracking_quality_threshold']
    
    def process(self, vo_data, filter_state):
        """Perform VO measurement update"""
        
        # Check tracking quality
        if vo_data.quality < self.quality_threshold:
            return self.handle_failure(vo_data, filter_state)
        
        # Extract relative pose
        relative_pose = vo_data.relative_pose
        
        # Measurement noise (scale with uncertainty)
        R = self.compute_vo_covariance(vo_data)
        
        # Update filter
        filter_state.update(
            measurement=relative_pose,
            R=R,
            measurement_model=self.measurement_model
        )
        
        return filter_state
    
    def handle_failure(self, vo_data, filter_state):
        """Handle VO tracking failure"""
        mode = self.failure_handling.get('mode', 'skip')
        
        if mode == 'skip':
            return filter_state
        elif mode == 'predict_only':
            # Continue with prediction only
            return filter_state
        elif mode == 'use_last':
            # Use last good estimate with high uncertainty
            return self.use_last_estimate(filter_state)


class ProcessingStrategyFactory:
    """Factory for creating processing strategies"""
    
    _strategies = {
        'imu_prediction': IMUPredictionStrategy,
        'imu_measurement': IMUMeasurementStrategy,
        'gnss_measurement': GNSSMeasurementStrategy,
        'vo_pose_measurement': VOPoseMeasurementStrategy,
        'magnetometer_measurement': MagnetometerMeasurementStrategy,
        'zero_velocity_update': ZeroVelocityStrategy,
        'none': NoOpStrategy,
    }
    
    @classmethod
    def create(cls, strategy_name: str, config: dict) -> ProcessingStrategy:
        """Create processing strategy from config"""
        strategy_class = cls._strategies.get(strategy_name)
        
        if strategy_class is None:
            raise ValueError(f"Unknown processing strategy: {strategy_name}")
        
        return strategy_class(config)
    
    @classmethod
    def register(cls, name: str, strategy_class: type):
        """Register custom processing strategy"""
        cls._strategies[name] = strategy_class
```

### 3. Unified Filter Interface

**New Filter Implementation:**
```python
class SensorFusionFilter:
    """
    Generic sensor fusion filter.
    NO hard-coded sensor type logic!
    """
    
    def __init__(self, config: dict, sensor_registry: SensorRegistry):
        self.config = config
        self.registry = sensor_registry
        
        # Create actual filter (EKF, UKF, PF, etc.)
        self.filter = FilterFactory.create(
            config['fusion']['filter_type'],
            config['fusion']
        )
        
        # Get prediction sources
        self.prediction_sources = sensor_registry.get_prediction_sources()
        
        # Get measurement sources
        self.measurement_sources = sensor_registry.get_measurement_sources()
        
        # State
        self.state = self.initialize_state()
    
    def process_sensor_data(self, sensor_id: str, data):
        """
        Process incoming sensor data.
        Dispatch is based on ROLE, not TYPE!
        """
        
        # Get sensor instance
        sensor = self.registry.get(sensor_id)
        
        if sensor is None:
            logger.warning(f"Unknown sensor: {sensor_id}")
            return
        
        # Process using sensor's strategy
        self.state = sensor.strategy.process(data, self.state)
        
        return self.state
    
    def initialize_state(self):
        """Initialize filter state from config"""
        state_config = self.config['fusion']['state']
        
        # Build state vector from components
        state_vector = []
        covariance = []
        
        for component in state_config['components']:
            state_vector.extend(component['initial_value'])
            
            # Covariance (simplified)
            cov_val = component['initial_covariance']
            covariance.extend([cov_val] * component['size'])
        
        x = np.array(state_vector)
        P = np.diag(covariance)
        
        return FilterState(x=x, P=P)


# Usage - completely configuration-driven!
config = load_config('euroc_mh01_sensors.yaml')
registry = SensorRegistry.from_config(config)
fusion_filter = SensorFusionFilter(config, registry)

# Process data
for timestamp, sensor_id, data in data_stream:
    state = fusion_filter.process_sensor_data(sensor_id, data)
    
    # NO conditionals based on sensor type!
    # Everything driven by configuration
```

### 4. Data Pipeline

**Complete Pipeline Example:**
```python
class ConfigurableFusionPipeline:
    """
    Fully configurable sensor fusion pipeline.
    No hard-coded sensor types anywhere!
    """
    
    def __init__(self, config_path: str):
        # Load configuration
        self.config = self.load_config(config_path)
        
        # Create sensor registry
        self.registry = SensorRegistry.from_config(config_path)
        
        # Create data readers for each sensor
        self.readers = self.create_readers()
        
        # Create fusion filter
        self.filter = SensorFusionFilter(self.config, self.registry)
        
        # Priority queue for time-ordered processing
        self.data_queue = PriorityQueue()
        
        # Evaluation (if ground truth available)
        self.evaluator = self.create_evaluator()
    
    def create_readers(self) -> Dict[str, DataReader]:
        """Create data reader for each sensor"""
        readers = {}
        
        for sensor_id, sensor in self.registry._sensors.items():
            # Skip sensors without data source (e.g., online processors)
            if sensor.reader is None:
                continue
            
            readers[sensor_id] = sensor.reader
        
        return readers
    
    def populate_queue(self):
        """Fill queue with all sensor data"""
        for sensor_id, reader in self.readers.items():
            for data in reader:
                self.data_queue.put((data.timestamp, sensor_id, data))
    
    def run(self):
        """Run the fusion pipeline"""
        
        # Load all data into queue
        self.populate_queue()
        
        # Process in temporal order
        while not self.data_queue.empty():
            timestamp, sensor_id, data = self.data_queue.get()
            
            # Get sensor instance
            sensor = self.registry.get(sensor_id)
            
            # Skip auxiliary sensors
            if sensor.get_role() == 'auxiliary':
                continue
            
            # Process data
            state = self.filter.process_sensor_data(sensor_id, data)
            
            # Evaluate if ground truth available
            if self.evaluator and sensor.get_role() == 'reference':
                self.evaluator.update(timestamp, state, data)
            
            # Log/visualize
            self.log_state(timestamp, sensor_id, state)
        
        # Generate report
        if self.evaluator:
            self.evaluator.generate_report()
    
    def log_state(self, timestamp, sensor_id, state):
        """Log filter state"""
        sensor = self.registry.get(sensor_id)
        
        logger.info(
            f"t={timestamp:.3f} | "
            f"sensor={sensor_id}({sensor.type}) | "
            f"role={sensor.get_role()} | "
            f"pos={state.position}"
        )


# Run pipeline - completely from configuration!
pipeline = ConfigurableFusionPipeline('config/euroc_mh01_sensors.yaml')
pipeline.run()

# Want to try different configuration? Just change YAML!
pipeline2 = ConfigurableFusionPipeline('config/kitti_07_sensors.yaml')
pipeline2.run()

# Want to add new sensor? Just edit YAML!
# NO CODE CHANGES NEEDED!
```

---

## Example Use Cases

### Use Case 1: Add New Dataset (Zero Code Changes)

**Task:** Add support for a new "CustomRobot" dataset

**Solution:** Create configuration file only

```yaml
# config/sensors/custom_robot_sensors.yaml

pipeline:
  name: "Custom Robot Dataset"
  
  sensors:
    # Novel sensor: Radar odometry
    radar_odom:
      identity:
        type: "RADAR_ODOMETRY"  # New type, but no code changes!
        model: "AWR1843"
      
      role:
        primary: "measurement"  # Standard role
        provides:
          - field: "velocity"   # Standard field
      
      processing:
        strategy: "velocity_measurement"  # Existing strategy!
      
      data_source:
        reader_type: "csv"      # Existing reader
        path_template: "{dataset_root}/radar/velocity.csv"
        field_mapping:
          velocity: [1, 2, 3]
    
    # Standard IMU
    imu:
      identity: { type: "IMU" }
      role: { primary: "prediction" }
      processing: { strategy: "imu_prediction" }
      data_source: { ... }
```

**Result:** Works immediately! No code changes needed.

### Use Case 2: Multiple Sensors Same Type

**Task:** Use 3 IMUs with different roles

```yaml
sensors:
  # Primary - drives prediction
  imu_primary:
    identity: { type: "IMU", location: "center" }
    role:
      primary: "prediction"
      weight: 1.0
    processing:
      strategy: "imu_prediction"
  
  # Secondary - consistency check
  imu_check_1:
    identity: { type: "IMU", location: "left" }
    role:
      primary: "measurement"
      weight: 0.2
    processing:
      strategy: "imu_consistency_check"
      reference_sensor: "imu_primary"
  
  # Tertiary - fault detection
  imu_check_2:
    identity: { type: "IMU", location: "right" }
    role:
      primary: "measurement"
      weight: 0.2
    processing:
      strategy: "imu_fault_detection"
      nominal_sensors: ["imu_primary", "imu_check_1"]
```

### Use Case 3: Experimental Configuration

**Task:** Test "IMU-only" vs "IMU+GPS" vs "IMU+GPS+VO"

```bash
# Three configurations, same code

# IMU only
python run_fusion.py --config configs/experiments/imu_only.yaml

# IMU + GPS
python run_fusion.py --config configs/experiments/imu_gps.yaml

# IMU + GPS + VO
python run_fusion.py --config configs/experiments/imu_gps_vo.yaml
```

**imu_only.yaml:**
```yaml
sensors:
  imu: { role: { primary: "prediction" }, ... }
  # That's it! No GPS, no VO
```

**imu_gps.yaml:**
```yaml
sensors:
  imu: { role: { primary: "prediction" }, ... }
  gps: { role: { primary: "measurement" }, ... }
```

**imu_gps_vo.yaml:**
```yaml
sensors:
  imu: { role: { primary: "prediction" }, ... }
  gps: { role: { primary: "measurement" }, ... }
  vo: { role: { primary: "measurement" }, ... }
```

### Use Case 4: Custom Processing Strategy

**Task:** Add custom "IMU temperature compensation" strategy

```python
# Define strategy (one time)
class IMUTemperatureCompensationStrategy(ProcessingStrategy):
    def __init__(self, config):
        self.temp_model = config['temperature_model']
    
    def process(self, data, filter_state):
        # Apply temperature compensation
        compensated_accel = self.compensate(data.accel, data.temperature)
        compensated_gyro = self.compensate(data.gyro, data.temperature)
        
        # Continue with standard prediction
        ...

# Register it
ProcessingStrategyFactory.register(
    'imu_temp_compensated',
    IMUTemperatureCompensationStrategy
)
```

**Use in configuration:**
```yaml
sensors:
  imu_precise:
    identity: { type: "IMU" }
    role: { primary: "prediction" }
    processing:
      strategy: "imu_temp_compensated"  # Use custom strategy!
      temperature_model: "polynomial_3rd_order"
```

### Use Case 5: Sensor Switching/Failover

```yaml
sensors:
  gps_primary:
    identity: { type: "GPS", receiver: "primary" }
    role:
      primary: "measurement"
      priority: 1  # Highest priority
    
    quality:
      min_satellites: 6
      failover_to: "gps_backup"  # If quality drops, switch
  
  gps_backup:
    identity: { type: "GPS", receiver: "backup" }
    role:
      primary: "auxiliary"  # Standby by default
      priority: 2
    
    activation:
      condition: "gps_primary.quality < threshold"
      auto_promote: true  # Automatically promote to measurement source
```

---

## Migration Path

### Phase 1: Parallel Implementation

1. Keep existing enum-based system
2. Implement new configuration system alongside
3. Create adapter between old and new
4. Test with one dataset

### Phase 2: Gradual Migration

```python
class HybridSensorType:
    """Adapter between old enum-based and new config-based systems"""
    
    def __init__(self, config_path: str):
        self.registry = SensorRegistry.from_config(config_path)
        self.legacy_mapping = self.create_legacy_mapping()
    
    def create_legacy_mapping(self):
        """Map new sensor IDs to old enums for compatibility"""
        mapping = {}
        
        for sensor_id, sensor in self.registry._sensors.items():
            # Determine old enum based on sensor type
            if sensor.type == "IMU":
                if "KITTI" in sensor_id.upper():
                    mapping[sensor_id] = KITTI_SensorType.OXTS_IMU
                elif "EUROC" in sensor_id.upper():
                    mapping[sensor_id] = EuRoC_SensorType.EuRoC_IMU
            # ... more mappings
        
        return mapping
    
    def is_time_update(self, sensor_id: str) -> bool:
        """Compatibility method"""
        sensor = self.registry.get(sensor_id)
        return sensor.get_role() == 'prediction'
```

### Phase 3: Complete Migration

1. Remove all enum definitions
2. Remove all `is_*_data()` methods
3. Use only configuration-based dispatch
4. Update tests

---

## Summary: Key Benefits

### Before (Enum-based)
```python
# Hard-coded types
class KITTI_SensorType(IntEnum):
    OXTS_IMU = 1
    OXTS_GPS = 2

# Hard-coded logic
if SensorType.is_imu_data(t):
    predict()
elif SensorType.is_gps_data(t):
    update_position()

# New dataset = code changes
```

### After (Configuration-based)
```yaml
# Configuration defines everything
sensors:
  imu: { role: { primary: "prediction" }, ... }
  gps: { role: { primary: "measurement" }, ... }
```

```python
# Generic processing
sensor = registry.get(sensor_id)
state = sensor.strategy.process(data, state)

# New dataset = new config file (NO code changes!)
```

### Benefits

✅ **Zero Code Changes** for new datasets  
✅ **Flexible Roles** - same sensor, different roles  
✅ **Multi-Instance** - natural support for N sensors  
✅ **Testable** - easy to create test configurations  
✅ **Maintainable** - no sprawling conditional logic  
✅ **Extensible** - plugin architecture for strategies  
✅ **Clear** - separation of identity, role, and strategy  
✅ **Type-Safe** - still use strong typing, just not enums  

---

**End of Document**

This architecture achieves complete configuration-driven flexibility while maintaining all the power of your current system!
