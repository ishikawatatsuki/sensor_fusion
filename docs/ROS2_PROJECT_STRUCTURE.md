# ROS2 Sensor Fusion Pipeline: Project Structure from Scratch

**Date:** February 2026  
**ROS2 Distribution:** Jazzy Jalisco (latest LTS) or Humble Hawksbill

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Package Organization](#package-organization)
4. [Configuration Architecture](#configuration-architecture)
5. [Node Architecture](#node-architecture)
6. [Message Definitions](#message-definitions)
7. [Launch System](#launch-system)
8. [Integration Patterns](#integration-patterns)
9. [Development Workflow](#development-workflow)

---

## Overview

This document presents a complete ROS2-based sensor fusion system architecture that maintains all features of the configuration-driven data loader while leveraging ROS2's distributed computing, message passing, and lifecycle management capabilities.

### Design Principles

1. **Configuration-Driven**: All sensors and algorithms configurable via YAML
2. **Modular**: Each component is a separate ROS2 node
3. **Reusable**: Generic nodes work across datasets
4. **Scalable**: Distribute processing across multiple machines
5. **Real-time Capable**: Support for live sensor streams
6. **Replay-Friendly**: Easy dataset replay for development

---

## Project Structure

```
sensor_fusion_ros2/
├── src/
│   ├── sensor_fusion_interfaces/          # Message/Service definitions
│   │   ├── msg/
│   │   │   ├── IMUData.msg
│   │   │   ├── GNSSData.msg
│   │   │   ├── StereoFrames.msg
│   │   │   ├── VisualOdometry.msg
│   │   │   ├── FilterState.msg
│   │   │   └── GroundTruth.msg
│   │   ├── srv/
│   │   │   ├── SetFilterParameters.srv
│   │   │   └── ResetFilter.srv
│   │   ├── action/
│   │   │   └── ProcessDataset.action
│   │   ├── CMakeLists.txt
│   │   └── package.xml
│   │
│   ├── sensor_fusion_core/                # Core fusion algorithms
│   │   ├── sensor_fusion_core/
│   │   │   ├── __init__.py
│   │   │   ├── filters/
│   │   │   │   ├── ekf_filter.py
│   │   │   │   ├── ukf_filter.py
│   │   │   │   ├── pf_filter.py
│   │   │   │   ├── enkf_filter.py
│   │   │   │   └── ckf_filter.py
│   │   │   ├── motion_models/
│   │   │   │   ├── kinematics_model.py
│   │   │   │   └── velocity_model.py
│   │   │   ├── measurement_models/
│   │   │   │   ├── imu_model.py
│   │   │   │   ├── gnss_model.py
│   │   │   │   └── vo_model.py
│   │   │   └── utils/
│   │   │       ├── quaternion.py
│   │   │       ├── coordinate_transforms.py
│   │   │       └── noise_models.py
│   │   ├── setup.py
│   │   ├── package.xml
│   │   └── CMakeLists.txt
│   │
│   ├── sensor_fusion_nodes/               # ROS2 node implementations
│   │   ├── sensor_fusion_nodes/
│   │   │   ├── __init__.py
│   │   │   ├── fusion_node.py             # Main fusion filter node
│   │   │   ├── sensor_reader_node.py      # Generic sensor reader
│   │   │   ├── vo_estimator_node.py       # Visual odometry
│   │   │   ├── visualizer_node.py         # Visualization
│   │   │   └── evaluation_node.py         # Error metrics
│   │   ├── launch/
│   │   │   ├── fusion_system.launch.py
│   │   │   ├── euroc_replay.launch.py
│   │   │   ├── kitti_replay.launch.py
│   │   │   ├── live_sensors.launch.py
│   │   │   └── multi_robot.launch.py
│   │   ├── config/
│   │   │   ├── sensor_schemas/
│   │   │   │   ├── euroc_sensors.yaml
│   │   │   │   ├── kitti_sensors.yaml
│   │   │   │   ├── uav_sensors.yaml
│   │   │   │   └── live_sensors.yaml
│   │   │   ├── filters/
│   │   │   │   ├── ekf_config.yaml
│   │   │   │   ├── ukf_config.yaml
│   │   │   │   └── pf_config.yaml
│   │   │   └── pipelines/
│   │   │       ├── euroc_pipeline.yaml
│   │   │       ├── kitti_pipeline.yaml
│   │   │       └── custom_pipeline.yaml
│   │   ├── setup.py
│   │   ├── package.xml
│   │   └── CMakeLists.txt
│   │
│   ├── sensor_fusion_dataset_loaders/     # Dataset-specific loaders
│   │   ├── sensor_fusion_dataset_loaders/
│   │   │   ├── __init__.py
│   │   │   ├── base_loader.py
│   │   │   ├── euroc_loader.py
│   │   │   ├── kitti_loader.py
│   │   │   ├── uav_loader.py
│   │   │   └── rosbag_loader.py
│   │   ├── setup.py
│   │   ├── package.xml
│   │   └── CMakeLists.txt
│   │
│   ├── sensor_fusion_visual_odometry/     # VO processing
│   │   ├── sensor_fusion_visual_odometry/
│   │   │   ├── __init__.py
│   │   │   ├── feature_detectors/
│   │   │   │   ├── sift.py
│   │   │   │   ├── orb.py
│   │   │   │   └── superpoint.py
│   │   │   ├── depth_estimators/
│   │   │   │   ├── depth_anything.py
│   │   │   │   └── zoe_depth.py
│   │   │   ├── estimators/
│   │   │   │   ├── estimator_2d2d.py
│   │   │   │   ├── estimator_2d3d.py
│   │   │   │   └── estimator_hybrid.py
│   │   │   └── object_detection/
│   │   │       ├── yolo.py
│   │   │       └── segformer.py
│   │   ├── setup.py
│   │   ├── package.xml
│   │   └── CMakeLists.txt
│   │
│   ├── sensor_fusion_visualization/       # Visualization tools
│   │   ├── sensor_fusion_visualization/
│   │   │   ├── __init__.py
│   │   │   ├── rviz_publisher.py
│   │   │   ├── plotjuggler_exporter.py
│   │   │   ├── trajectory_visualizer.py
│   │   │   └── live_plotter.py
│   │   ├── rviz/
│   │   │   ├── fusion_display.rviz
│   │   │   └── multi_robot.rviz
│   │   ├── setup.py
│   │   ├── package.xml
│   │   └── CMakeLists.txt
│   │
│   └── sensor_fusion_evaluation/          # Metrics and analysis
│       ├── sensor_fusion_evaluation/
│       │   ├── __init__.py
│       │   ├── metrics_calculator.py
│       │   ├── error_analysis.py
│       │   └── report_generator.py
│       ├── setup.py
│       ├── package.xml
│       └── CMakeLists.txt
│
├── datasets/                               # Dataset storage
│   ├── EuRoC/
│   ├── KITTI/
│   └── UAV/
│
├── results/                                # Output storage
│   ├── trajectories/
│   ├── metrics/
│   └── visualizations/
│
├── docker/                                 # Containerization
│   ├── Dockerfile.dev
│   ├── Dockerfile.prod
│   └── docker-compose.yaml
│
├── docs/                                   # Documentation
│   ├── architecture.md
│   ├── configuration_guide.md
│   ├── node_reference.md
│   └── tutorials/
│
├── scripts/                                # Utility scripts
│   ├── download_datasets.sh
│   ├── run_experiments.sh
│   └── benchmark.py
│
├── .github/
│   └── workflows/
│       ├── ci.yaml
│       └── docker-build.yaml
│
├── colcon.meta                             # Colcon configuration
├── workspace.repos                         # VCS import file
└── README.md
```

---

## Package Organization

### 1. sensor_fusion_interfaces

**Purpose**: Define all custom messages, services, and actions

**Key Files**:
```
msg/
  IMUData.msg              - IMU measurements (accel, gyro, timestamp)
  GNSSData.msg             - GPS/GNSS position
  StereoFrames.msg         - Stereo camera image pairs
  VisualOdometry.msg       - VO pose estimates
  FilterState.msg          - Complete filter state (pose, velocity, biases)
  GroundTruth.msg          - Ground truth data
  SensorHealth.msg         - Sensor status/diagnostics

srv/
  SetFilterParameters.srv  - Reconfigure filter at runtime
  ResetFilter.srv          - Reset filter state
  GetState.srv             - Query current state

action/
  ProcessDataset.action    - Long-running dataset processing
  Calibration.action       - Sensor calibration procedure
```

**Message Design Example**:
```yaml
# IMUData.msg
std_msgs/Header header
geometry_msgs/Vector3 linear_acceleration
geometry_msgs/Vector3 angular_velocity
float64[9] acceleration_covariance
float64[9] angular_velocity_covariance
uint8 sensor_id
string sensor_name

# FilterState.msg
std_msgs/Header header
geometry_msgs/PoseWithCovariance pose
geometry_msgs/TwistWithCovariance twist
geometry_msgs/Vector3 gyro_bias
geometry_msgs/Vector3 accel_bias
float64[] state_vector              # Full state for advanced filters
float64[] covariance_matrix         # Flattened covariance
uint8 filter_mode                   # EKF=0, UKF=1, PF=2, etc.
float64 innovation_norm             # For health monitoring
```

---

### 2. sensor_fusion_core

**Purpose**: Algorithm implementations (filters, models)

**Architecture**:
```python
# filters/base_filter.py
class SensorFusionFilter(ABC):
    """Abstract base class for all filters"""
    
    @abstractmethod
    def predict(self, control_input, dt):
        """Prediction step"""
        pass
    
    @abstractmethod
    def update(self, measurement, measurement_type):
        """Update step"""
        pass
    
    @abstractmethod
    def get_state(self):
        """Return current state estimate"""
        pass

# filters/ekf_filter.py
class ExtendedKalmanFilter(SensorFusionFilter):
    """EKF implementation"""
    
    def __init__(self, config: FilterConfig):
        self.config = config
        self.state = None
        self.covariance = None
        self.motion_model = self._create_motion_model()
        self.measurement_models = self._create_measurement_models()
    
    def predict(self, control_input, dt):
        # Standard EKF prediction
        pass
    
    def update(self, measurement, measurement_type):
        # Standard EKF update with sensor-specific model
        pass
```

**Plugin Architecture**:
```python
# Plugin registry for filters
FILTER_REGISTRY = {
    'ekf': ExtendedKalmanFilter,
    'ukf': UnscentedKalmanFilter,
    'pf': ParticleFilter,
    'enkf': EnsembleKalmanFilter,
    'ckf': CubatureKalmanFilter,
}

def create_filter(filter_type: str, config: dict):
    """Factory function"""
    filter_class = FILTER_REGISTRY.get(filter_type.lower())
    if filter_class is None:
        raise ValueError(f"Unknown filter type: {filter_type}")
    return filter_class(config)
```

---

### 3. sensor_fusion_nodes

**Purpose**: ROS2 node implementations

#### Main Nodes:

**A. Fusion Node** (`fusion_node.py`)
```python
class SensorFusionNode(Node):
    """
    Main fusion filter node.
    Subscribes to sensor topics, runs filter, publishes state.
    """
    
    def __init__(self):
        super().__init__('sensor_fusion_node')
        
        # Parameters
        self.declare_parameters()
        
        # Create filter based on config
        self.filter = self.create_filter_from_config()
        
        # Subscribers (configured from YAML)
        self.create_sensor_subscribers()
        
        # Publishers
        self.state_pub = self.create_publisher(
            FilterState, '/fusion/state', 10
        )
        self.pose_pub = self.create_publisher(
            PoseStamped, '/fusion/pose', 10
        )
        self.path_pub = self.create_publisher(
            Path, '/fusion/path', 10
        )
        
        # Services
        self.reset_srv = self.create_service(
            ResetFilter, '/fusion/reset', self.reset_callback
        )
        
        # Diagnostics
        self.diagnostics = DiagnosticUpdater(self)
        
        # TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)
    
    def create_sensor_subscribers(self):
        """Dynamically create subscribers based on config"""
        sensor_config = self.get_parameter('sensors').value
        
        for sensor_name, sensor_info in sensor_config.items():
            topic = sensor_info['topic']
            msg_type = self.get_message_type(sensor_info['type'])
            
            self.create_subscription(
                msg_type,
                topic,
                lambda msg, sn=sensor_name: self.sensor_callback(msg, sn),
                10
            )
    
    def sensor_callback(self, msg, sensor_name):
        """Handle incoming sensor data"""
        
        # Check if this is IMU (prediction) or measurement (update)
        if self.is_prediction_sensor(sensor_name):
            self.predict_step(msg)
        else:
            self.update_step(msg, sensor_name)
        
        # Publish state
        self.publish_state()
        
        # Publish TF
        self.publish_transform()
```

**B. Sensor Reader Node** (`sensor_reader_node.py`)
```python
class SensorReaderNode(Node):
    """
    Generic sensor data reader node.
    Reads from dataset files or live sensors, publishes to topics.
    Configuration-driven - no code changes for new sensors!
    """
    
    def __init__(self):
        super().__init__('sensor_reader_node')
        
        # Load sensor schema from config
        self.sensor_schemas = self.load_sensor_schemas()
        
        # Create publishers for each sensor
        self.publishers = {}
        self.readers = {}
        
        for sensor_name, schema in self.sensor_schemas.items():
            # Create publisher
            topic = schema['topic']
            msg_type = self.get_message_type(schema['message_type'])
            self.publishers[sensor_name] = self.create_publisher(
                msg_type, topic, 10
            )
            
            # Create data reader (using configurable loader)
            self.readers[sensor_name] = self.create_reader(
                sensor_name, schema
            )
        
        # Start publishing
        self.timer = self.create_timer(0.001, self.publish_data)
        
        # Priority queue for time-synchronized publishing
        self.data_queue = PriorityQueue()
        self.populate_queue()
    
    def create_reader(self, sensor_name, schema):
        """Create data reader using configuration"""
        from sensor_fusion_dataset_loaders import create_loader
        
        return create_loader(
            dataset_type=schema['dataset_type'],
            sensor_type=sensor_name,
            config=schema
        )
    
    def populate_queue(self):
        """Fill queue with all sensor data, sorted by timestamp"""
        for sensor_name, reader in self.readers.items():
            for data in reader:
                self.data_queue.put((data.timestamp, sensor_name, data))
    
    def publish_data(self):
        """Publish next data point from queue"""
        if self.data_queue.empty():
            return
        
        timestamp, sensor_name, data = self.data_queue.get()
        
        # Convert to ROS message
        msg = self.convert_to_ros_message(data, sensor_name)
        
        # Publish
        self.publishers[sensor_name].publish(msg)
```

**C. Visual Odometry Node** (`vo_estimator_node.py`)
```python
class VisualOdometryNode(Node):
    """
    Monocular/stereo visual odometry estimation.
    Subscribes to camera images, publishes pose estimates.
    """
    
    def __init__(self):
        super().__init__('visual_odometry_node')
        
        # Parameters
        self.vo_config = self.get_vo_config()
        
        # Create VO estimator
        self.estimator = self.create_estimator()
        
        # Subscribers
        self.create_image_subscribers()
        
        # Publishers
        self.vo_pub = self.create_publisher(
            VisualOdometry, '/vo/estimate', 10
        )
        self.feature_pub = self.create_publisher(
            Image, '/vo/features', 10
        )
        
        # Image buffer for stereo
        self.image_buffer = {}
    
    def create_estimator(self):
        """Create VO estimator from config"""
        from sensor_fusion_visual_odometry import create_vo_estimator
        
        return create_vo_estimator(
            estimator_type=self.vo_config['estimator'],
            feature_detector=self.vo_config['feature_detector'],
            depth_estimator=self.vo_config.get('depth_estimator'),
            config=self.vo_config
        )
    
    def stereo_callback(self, left_msg, right_msg):
        """Process stereo pair"""
        
        # Convert to OpenCV
        left_img = self.bridge.imgmsg_to_cv2(left_msg)
        right_img = self.bridge.imgmsg_to_cv2(right_msg)
        
        # Estimate pose
        pose_estimate = self.estimator.estimate(left_img, right_img)
        
        # Publish
        vo_msg = self.create_vo_message(pose_estimate, left_msg.header)
        self.vo_pub.publish(vo_msg)
```

**D. Visualizer Node** (`visualizer_node.py`)
```python
class VisualizationNode(Node):
    """
    Real-time visualization using RViz markers and PlotJuggler.
    """
    
    def __init__(self):
        super().__init__('visualization_node')
        
        # Publishers for RViz
        self.marker_pub = self.create_publisher(
            MarkerArray, '/visualization/markers', 10
        )
        self.trajectory_pub = self.create_publisher(
            Path, '/visualization/trajectory', 10
        )
        
        # Subscribers
        self.state_sub = self.create_subscription(
            FilterState, '/fusion/state',
            self.state_callback, 10
        )
        self.gt_sub = self.create_subscription(
            GroundTruth, '/ground_truth',
            self.gt_callback, 10
        )
        
        # Trajectory history
        self.estimated_path = Path()
        self.ground_truth_path = Path()
    
    def state_callback(self, msg):
        """Visualize filter state"""
        
        # Add to trajectory
        pose = PoseStamped()
        pose.header = msg.header
        pose.pose = msg.pose.pose
        self.estimated_path.poses.append(pose)
        
        # Publish trajectory
        self.trajectory_pub.publish(self.estimated_path)
        
        # Publish covariance ellipsoid
        self.publish_covariance_marker(msg)
        
        # Publish particle cloud (for PF)
        if msg.filter_mode == FilterMode.PARTICLE_FILTER:
            self.publish_particles(msg)
```

---

## Configuration Architecture

### Hierarchical Configuration System

```
config/
├── pipelines/                    # Top-level pipeline configs
│   └── euroc_pipeline.yaml
├── filters/                      # Filter-specific configs
│   └── ekf_config.yaml
└── sensor_schemas/               # Sensor definitions
    └── euroc_sensors.yaml
```

### Example: Pipeline Configuration

**`config/pipelines/euroc_pipeline.yaml`**:
```yaml
# Top-level pipeline configuration
pipeline:
  name: "EuRoC MH_01 Fusion Pipeline"
  dataset:
    type: euroc
    variant: "01"
    root_path: "/datasets/EuRoC"
    playback_rate: 1.0          # 1.0 = real-time, 2.0 = 2x speed
  
  # Filter configuration
  filter:
    type: ekf                     # ekf, ukf, pf, enkf, ckf
    config_file: "$(find sensor_fusion_nodes)/config/filters/ekf_config.yaml"
    initial_state:
      position: [0.0, 0.0, 0.0]
      velocity: [0.0, 0.0, 0.0]
      orientation: [1.0, 0.0, 0.0, 0.0]  # quaternion
  
  # Sensor configuration (references sensor schemas)
  sensors:
    imu:
      enabled: true
      schema_file: "$(find sensor_fusion_nodes)/config/sensor_schemas/euroc_sensors.yaml"
      schema_name: "euroc_imu"
      topic: "/sensors/imu"
      frame_id: "imu_link"
      dropout_ratio: 0.0
      
    leica:
      enabled: true
      schema_file: "$(find sensor_fusion_nodes)/config/sensor_schemas/euroc_sensors.yaml"
      schema_name: "euroc_leica"
      topic: "/sensors/gnss"
      frame_id: "gnss_link"
      dropout_ratio: 0.0
    
    stereo:
      enabled: true
      schema_file: "$(find sensor_fusion_nodes)/config/sensor_schemas/euroc_sensors.yaml"
      schema_name: "euroc_stereo"
      left_topic: "/camera/left/image_raw"
      right_topic: "/camera/right/image_raw"
      frame_id: "camera_link"
    
    visual_odometry:
      enabled: true
      input_source: "stereo"      # stereo, monocular
      topic: "/vo/estimate"
      config:
        estimator: "2d3d"
        feature_detector: "SIFT"
        max_features: 1500
  
  # Visualization
  visualization:
    enabled: true
    rviz_config: "$(find sensor_fusion_visualization)/rviz/fusion_display.rviz"
    publish_tf: true
    show_covariance: true
    trajectory_length: 1000
  
  # Evaluation
  evaluation:
    enabled: true
    ground_truth_topic: "/ground_truth"
    metrics:
      - rmse
      - ate
      - rpe
    output_dir: "/results/euroc_mh01"
```

### Example: Sensor Schema Configuration

**`config/sensor_schemas/euroc_sensors.yaml`**:
```yaml
# EuRoC dataset sensor schemas
# These define how to read sensor data and convert to ROS messages

sensor_schemas:
  euroc_imu:
    sensor_type: "IMU"
    message_type: "sensor_fusion_interfaces/msg/IMUData"
    
    data_source:
      type: csv
      path_template: "{root_path}/MH_{variant}_easy/imu0/data.csv"
      delimiter: ","
      skip_header: 1
    
    timestamp:
      field: timestamp
      type: int
      scale: 1e-9              # nanoseconds to seconds
    
    fields:
      - name: angular_velocity
        columns: [1, 2, 3]     # w_x, w_y, w_z
        type: array
        units: "rad/s"
      
      - name: linear_acceleration
        columns: [4, 5, 6]     # a_x, a_y, a_z
        type: array
        units: "m/s^2"
    
    noise_model:
      angular_velocity:
        density: 0.000236      # rad/s/√Hz
        random_walk: 0.00333   # rad/s²/√Hz
      linear_acceleration:
        density: 0.00226       # m/s²/√Hz
        random_walk: 0.0319    # m/s³/√Hz
    
    frame_id: "imu_link"
    frequency: 200             # Hz
  
  euroc_leica:
    sensor_type: "GNSS"
    message_type: "sensor_fusion_interfaces/msg/GNSSData"
    
    data_source:
      type: csv
      path_template: "{root_path}/MH_{variant}_easy/leica0/data.csv"
      delimiter: ","
      skip_header: 1
    
    timestamp:
      field: timestamp
      type: int
      scale: 1e-9
    
    fields:
      - name: position
        columns: [1, 2, 3]     # p_x, p_y, p_z
        type: array
        units: "m"
    
    noise_model:
      position:
        covariance: [0.01, 0.01, 0.02]  # x, y, z variance
    
    frame_id: "gnss_link"
    frequency: 10
  
  euroc_stereo:
    sensor_type: "STEREO_CAMERA"
    message_type: "sensor_msgs/msg/Image"
    
    data_source:
      type: image_folder
      left_path_template: "{root_path}/MH_{variant}_easy/cam0/data"
      right_path_template: "{root_path}/MH_{variant}_easy/cam1/data"
      timestamp_file: "{root_path}/MH_{variant}_easy/cam0/data.csv"
    
    calibration:
      fx: 458.654
      fy: 457.296
      cx: 367.215
      cy: 248.375
      baseline: 0.11
      distortion: [-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05]
    
    frame_id: "camera_link"
    frequency: 20
```

### Example: Filter Configuration

**`config/filters/ekf_config.yaml`**:
```yaml
# Extended Kalman Filter Configuration

filter:
  type: ekf
  dimension: 3              # 2D or 3D
  
  motion_model:
    type: velocity          # kinematics or velocity
    compensate_gravity: true
    
  state_vector:
    # 15-state: [p_x, p_y, p_z, v_x, v_y, v_z, q_w, q_x, q_y, q_z, b_w_x, b_w_y, b_w_z, b_a_x, b_a_y, b_a_z]
    position: [0, 1, 2]
    velocity: [3, 4, 5]
    orientation: [6, 7, 8, 9]
    gyro_bias: [10, 11, 12]
    accel_bias: [13, 14, 15]
  
  initial_covariance:
    position: 0.01           # m^2
    velocity: 0.01           # (m/s)^2
    orientation: 0.01        # rad^2
    gyro_bias: 1e-6          # (rad/s)^2
    accel_bias: 1e-4         # (m/s^2)^2
  
  process_noise:
    type: default            # default, adaptive
    position: 0.001
    velocity: 0.01
    orientation: 0.001
    gyro_bias: 1e-8
    accel_bias: 1e-6
  
  measurement_models:
    imu:
      enabled: true
      innovation_masking: false
      mahalanobis_threshold: 5.0
      
    gnss:
      enabled: true
      innovation_masking: true
      mahalanobis_threshold: 3.0
      outlier_rejection: true
      
    visual_odometry:
      enabled: true
      measurement_type: pose  # pose, velocity, position
      innovation_masking: true
      mahalanobis_threshold: 4.0
  
  advanced:
    numerical_stability:
      min_eigenvalue: 1e-9
      use_joseph_form: true
    
    adaptive_tuning:
      enabled: false
      window_size: 50
      alpha: 0.3
```

---

## Node Architecture

### Node Graph (Example: EuRoC Dataset Replay)

```
                    ┌─────────────────────────────────────┐
                    │     Dataset Reader Nodes            │
                    └─────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
┌───────▼────────┐        ┌─────────▼────────┐      ┌──────────▼─────────┐
│  IMU Reader    │        │  GNSS Reader     │      │  Stereo Reader     │
│  Node          │        │  Node            │      │  Node              │
└───────┬────────┘        └─────────┬────────┘      └──────────┬─────────┘
        │                           │                           │
        │ /sensors/imu              │ /sensors/gnss             │ /camera/left
        │                           │                           │ /camera/right
        │                           │                           │
        │                           │                ┌──────────▼─────────┐
        │                           │                │  Visual Odometry   │
        │                           │                │  Estimator Node    │
        │                           │                └──────────┬─────────┘
        │                           │                           │
        │                           │                           │ /vo/estimate
        │                           │                           │
        └───────────────────────────┼───────────────────────────┘
                                    │
                           ┌────────▼────────┐
                           │  Sensor Fusion  │
                           │  Filter Node    │
                           └────────┬────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
                    │ /fusion/state │ /tf           │ /fusion/path
                    │               │               │
        ┌───────────▼──────┐  ┌─────▼─────┐  ┌─────▼──────────┐
        │  Visualization   │  │  TF Tree  │  │  Evaluation    │
        │  Node            │  │           │  │  Node          │
        └──────────────────┘  └───────────┘  └────────────────┘
                │                                      │
                │ /visualization_markers               │ /metrics
                ▼                                      ▼
            RViz                                  Results Files
```

### Topic Naming Conventions

```
/sensors/                    # Raw sensor data
  ├── imu                    # IMU measurements
  ├── gnss                   # GNSS position
  ├── mag                    # Magnetometer
  └── ...

/camera/                     # Camera topics
  ├── left/
  │   ├── image_raw
  │   └── camera_info
  └── right/
      ├── image_raw
      └── camera_info

/vo/                         # Visual odometry
  ├── estimate               # VO pose estimate
  ├── features               # Feature visualization
  └── debug

/fusion/                     # Fusion outputs
  ├── state                  # Complete filter state
  ├── pose                   # Pose only
  ├── twist                  # Velocity
  ├── path                   # Trajectory
  └── diagnostics            # Health monitoring

/ground_truth/               # Ground truth (if available)
  ├── pose
  └── path

/visualization/              # Visualization markers
  ├── markers
  ├── trajectory
  └── covariance

/metrics/                    # Evaluation metrics
  ├── rmse
  ├── ate
  └── rpe
```

### Frame Tree (TF2)

```
map
 └── odom
      └── base_link
           ├── imu_link
           ├── gnss_link
           ├── camera_link
           │    ├── camera_left
           │    └── camera_right
           └── body_link
```

---

## Message Definitions

### Custom Messages

**IMUData.msg**:
```
std_msgs/Header header

geometry_msgs/Vector3 linear_acceleration
geometry_msgs/Vector3 angular_velocity

# Covariances (row-major about x, y, z axes)
float64[9] linear_acceleration_covariance
float64[9] angular_velocity_covariance

# Sensor metadata
uint8 sensor_id
string sensor_name
```

**FilterState.msg**:
```
std_msgs/Header header

# Pose (position + orientation)
geometry_msgs/PoseWithCovariance pose

# Velocity
geometry_msgs/TwistWithCovariance twist

# Biases
geometry_msgs/Vector3 gyro_bias
geometry_msgs/Vector3 accel_bias
float64[9] gyro_bias_covariance
float64[9] accel_bias_covariance

# Full state for advanced filters
float64[] state_vector
float64[] covariance_matrix

# Filter metadata
uint8 filter_type  # EKF=0, UKF=1, PF=2, ENKF=3, CKF=4
float64 innovation_norm
float64 process_time_ms
uint32 update_count
```

**VisualOdometry.msg**:
```
std_msgs/Header header

# Relative transformation
geometry_msgs/TransformStamped transform

# Alternative: absolute pose
geometry_msgs/PoseWithCovariance pose

# Velocity estimate (optional)
geometry_msgs/TwistWithCovariance twist

# Quality metrics
float32 num_features
float32 num_inliers
float32 reprojection_error
bool tracking_lost
```

---

## Launch System

### Hierarchical Launch Files

**`launch/fusion_system.launch.py`** (Main launcher):
```python
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    
    # Arguments
    pipeline_config = LaunchConfiguration('pipeline_config')
    use_rviz = LaunchConfiguration('use_rviz')
    
    declare_pipeline_config = DeclareLaunchArgument(
        'pipeline_config',
        default_value='euroc_pipeline.yaml',
        description='Pipeline configuration file'
    )
    
    declare_use_rviz = DeclareLaunchArgument(
        'use_rviz',
        default_value='true',
        description='Launch RViz for visualization'
    )
    
    # Config path
    config_path = PathJoinSubstitution([
        FindPackageShare('sensor_fusion_nodes'),
        'config', 'pipelines',
        pipeline_config
    ])
    
    # Nodes
    sensor_reader_node = Node(
        package='sensor_fusion_nodes',
        executable='sensor_reader_node',
        name='sensor_reader',
        parameters=[config_path],
        output='screen'
    )
    
    fusion_node = Node(
        package='sensor_fusion_nodes',
        executable='fusion_node',
        name='sensor_fusion',
        parameters=[config_path],
        output='screen'
    )
    
    vo_node = Node(
        package='sensor_fusion_visual_odometry',
        executable='vo_estimator_node',
        name='visual_odometry',
        parameters=[config_path],
        output='screen'
    )
    
    visualization_node = Node(
        package='sensor_fusion_visualization',
        executable='visualizer_node',
        name='visualization',
        parameters=[config_path],
        condition=IfCondition(use_rviz)
    )
    
    evaluation_node = Node(
        package='sensor_fusion_evaluation',
        executable='evaluation_node',
        name='evaluation',
        parameters=[config_path]
    )
    
    # RViz
    rviz_config = PathJoinSubstitution([
        FindPackageShare('sensor_fusion_visualization'),
        'rviz', 'fusion_display.rviz'
    ])
    
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        condition=IfCondition(use_rviz)
    )
    
    return LaunchDescription([
        declare_pipeline_config,
        declare_use_rviz,
        sensor_reader_node,
        vo_node,
        fusion_node,
        visualization_node,
        evaluation_node,
        rviz_node
    ])
```

**Usage**:
```bash
# Launch EuRoC pipeline
ros2 launch sensor_fusion_nodes fusion_system.launch.py \
    pipeline_config:=euroc_pipeline.yaml

# Launch KITTI pipeline without visualization
ros2 launch sensor_fusion_nodes fusion_system.launch.py \
    pipeline_config:=kitti_pipeline.yaml \
    use_rviz:=false

# Launch with custom config
ros2 launch sensor_fusion_nodes fusion_system.launch.py \
    pipeline_config:=/path/to/custom_pipeline.yaml
```

---

## Integration Patterns

### Pattern 1: Offline Dataset Processing

```
Dataset Files → Reader Nodes → Sensor Topics → Fusion Node → Results
```

- Reader nodes publish at controlled rate (playback_rate parameter)
- All data timestamped for proper synchronization
- Can pause/resume/step through data
- Results saved to files

### Pattern 2: Live Sensor Integration

```
Live Sensors → ROS2 Drivers → Sensor Topics → Fusion Node → Real-time Output
```

- Hardware drivers publish sensor data
- Fusion node processes in real-time
- TF broadcasts for robot localization
- Integration with Nav2, MoveIt, etc.

### Pattern 3: Hybrid (Replay + Live)

```
Recorded Bag + Live Sensors → Fusion Node → Output
```

- Some sensors from rosbag (e.g., cameras)
- Some sensors live (e.g., IMU, GPS)
- Useful for testing/development

### Pattern 4: Distributed Processing

```
Machine 1: Cameras → VO Node
Machine 2: IMU/GPS → Reader Nodes
Machine 3: Fusion Node → Results
```

- Nodes can run on different machines
- ROS2 discovery handles networking
- Good for computationally intensive tasks (VO, deep learning)

---

## Development Workflow

### 1. Build System

**Using colcon**:
```bash
# Build all packages
cd sensor_fusion_ros2
colcon build

# Build specific package
colcon build --packages-select sensor_fusion_core

# Build with debugging
colcon build --cmake-args -DCMAKE_BUILD_TYPE=Debug

# Build in parallel
colcon build --parallel-workers 8
```

### 2. Testing

**Unit Tests** (pytest for Python):
```bash
# Run all tests
colcon test

# Run specific package tests
colcon test --packages-select sensor_fusion_core

# View test results
colcon test-result --verbose
```

**Integration Tests** (launch_testing):
```python
# test/test_euroc_pipeline.py
import launch_testing
import pytest

@pytest.fixture
def launch_euroc_pipeline():
    return IncludeLaunchDescription(...)

def test_fusion_publishes_state(launch_euroc_pipeline):
    # Test that fusion node publishes state
    pass
```

### 3. Debugging

**Using RQt**:
```bash
# View node graph
rqt_graph

# Monitor topics
rqt_topic

# Plot data
rqt_plot /fusion/pose/pose/position/x /ground_truth/pose/position/x

# View TF tree
rqt_tf_tree
```

**Using PlotJuggler**:
```bash
# Real-time plotting
ros2 run plotjuggler plotjuggler
```

### 4. Running Experiments

**Batch Processing Script** (`scripts/run_experiments.sh`):
```bash
#!/bin/bash

# Run multiple configurations
CONFIGS=(
    "euroc_ekf.yaml"
    "euroc_ukf.yaml"
    "euroc_pf.yaml"
)

for config in "${CONFIGS[@]}"; do
    echo "Running $config..."
    
    ros2 launch sensor_fusion_nodes fusion_system.launch.py \
        pipeline_config:=$config \
        use_rviz:=false
    
    # Wait for completion
    sleep 5
done

# Generate comparison report
python3 scripts/compare_results.py --configs ${CONFIGS[@]}
```

### 5. Docker Deployment

**Dockerfile.prod**:
```dockerfile
FROM ros:jazzy

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    ros-jazzy-vision-opencv \
    ros-jazzy-image-transport \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt /tmp/
RUN pip3 install -r /tmp/requirements.txt

# Copy workspace
COPY . /workspace/sensor_fusion_ros2
WORKDIR /workspace/sensor_fusion_ros2

# Build
RUN . /opt/ros/jazzy/setup.sh && \
    colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release

# Entrypoint
COPY docker/entrypoint.sh /
ENTRYPOINT ["/entrypoint.sh"]
CMD ["ros2", "launch", "sensor_fusion_nodes", "fusion_system.launch.py"]
```

**Usage**:
```bash
# Build image
docker build -f docker/Dockerfile.prod -t sensor_fusion:latest .

# Run container
docker run -it --rm \
    -v /path/to/datasets:/datasets:ro \
    -v /path/to/results:/results \
    sensor_fusion:latest \
    ros2 launch sensor_fusion_nodes fusion_system.launch.py \
        pipeline_config:=euroc_pipeline.yaml
```

---

## Summary: Key Advantages of ROS2 Architecture

1. **Modularity**: Each component is a separate node
2. **Scalability**: Distribute processing across machines
3. **Reusability**: Generic nodes work across datasets
4. **Real-time**: Native support for real-time systems
5. **Visualization**: Rich ecosystem (RViz, PlotJuggler, rqt)
6. **Debugging**: Extensive tooling for inspection
7. **Community**: Large ecosystem of packages and tools
8. **Standards**: Uses standard message types where possible
9. **Configuration-Driven**: No code changes for new sensors
10. **Integration**: Easy integration with robotics stacks

---

## Next Steps

1. **Set up workspace**: Create ROS2 workspace
2. **Create packages**: Use `ros2 pkg create` for each package
3. **Implement core algorithms**: Port existing filters
4. **Create reader nodes**: Implement configuration-driven readers
5. **Design messages**: Define custom message types
6. **Test with datasets**: Validate against EuRoC, KITTI
7. **Add visualization**: Create RViz displays
8. **Document**: API docs, tutorials, examples
9. **Deploy**: Docker containers, CI/CD
10. **Extend**: Add new sensors, algorithms, features

---

**End of Document**
