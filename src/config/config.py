import yaml
from collections import namedtuple

ModeConfig = namedtuple('ModeConfig', [
  'log_level', 
  'log_sensor_data', 
  'save_estimation', 
  'save_output_debug_frames',
  'sensor_data_output_filepath'
  ])

DatasetConfig = namedtuple('DatasetConfig', [
  'type', 
  'mode', 
  'root_path', 
  'variant', 
  'sensors'
  ])

VisualizationConfig = namedtuple('VisualizationConfig', [
  'realtime', 
  'output_filepath', 
  'save_frames',
  'save_trajectory',
  'show_vo_trajectory',
  'show_angle_estimation',
  'show_end_result',
  'show_vio_frame',
  'show_particles',
  'set_lim_in_plot',
  'show_innovation_history',
  'limits',
  ])
GeometricLimit = namedtuple('GeometricLimit', ['min', 'max'])

ReportConfig = namedtuple('ReportConfig', ['export_error', 'error_output_root_path', 'kitti_pose_result_folder', 'location_only'])

SensorConfig = namedtuple('SensorConfig', ('name', 'dropout_ratio', 'window_size'))
FilterConfig = namedtuple('FilterConfig', ['type', 'dimension', 'motion_model', 'noise_type',  'innovation_masking', 'vo_velocity_only_update_when_failure', 'params'])

VO_Mode = namedtuple('Mode', ['type', 'save_traj', 'save_poses_kitti_format', 'save_output_debug_frames', 'output_directory', 'log_level'])
VO_Stereo = namedtuple('Stereo', ['algorithm', 'speckle_window_size', 'median_filter', 'wsl_filter', 'dynamic_depth'])
VO_Detector = namedtuple('Detector', ['algorithm', 'no_of_keypoints', 'homography', 'error_threshold', 'circular_matching'])
VO_Matcher = namedtuple('Matcher', ['algorithm', 'ratio_test', 'dynamic_ratio'])
VO_MotionEstimation = namedtuple('MotionEstimation', ['algorithm', 'depth_threshold', 'depth_limit', 'x_correspondence_threshold', 'invalidate_cars'])

VisualOdometryConfig = namedtuple('VisualOdometryConfig', [
  'mode',
  'stereo',
  'detector',
  'matcher',
  'motion_estimation'
])

GyroSpecification = namedtuple("GyroSpecification", ['noise', 'offset'])
AccelSpecification = namedtuple("AccelSpecification", ['noise', 'offset', 'scale'])

def get_geometric_limitations(dataset: str, variant: str) -> GeometricLimit:
  if dataset == "kitti" or dataset == "experiment":
    match (variant):
      case "0016":
        return GeometricLimit(min=[], max=[])
      case "0033":
        return GeometricLimit(min=[-100, -400, -30], max=[600, 200, 15])
      case _:
        return GeometricLimit(min=[0, 0, 0], max=[100, 100, 20])
  else:
    match (variant):
      case "log0001":
        return GeometricLimit(min=[-5, -20, -8], max=[15, 15, 10])
      case "log0002":
        return GeometricLimit(min=[], max=[])
      case _:
        return GeometricLimit(min=[0, 0, 0], max=[100, 100, 20])

class Config:
  
  def __init__(
    self, 
    config_filepath,
    ):
    """
      - config_filepath: yaml file path that stores system's configurations
    """
    self.config_filepath = config_filepath
    
    config = None
    with open(config_filepath, "r") as f:
        config = yaml.safe_load(f)
        f.close()
    
    self.mode = ModeConfig(**config['mode'])
    
    self.dataset = DatasetConfig(**config["dataset"])
    sensors = [SensorConfig(name=sensor, dropout_ratio=value['dropout_ratio'], window_size=value['window_size']) \
                for sensor, value in self.dataset.sensors.items() if value['selected']]
    
    self.dataset = self.dataset._replace(sensors=sensors)
    self.filter = FilterConfig(**config["filter"])

    self.report = ReportConfig(**config['report'])

    self.vo_config = VisualOdometryConfig(
      mode=VO_Mode(**config["visual_odometry"]["mode"]),
      stereo=VO_Stereo(**config["visual_odometry"]["stereo"]),
      detector=VO_Detector(**config["visual_odometry"]["detector"]),
      matcher=VO_Matcher(**config["visual_odometry"]["matcher"]),
      motion_estimation=VO_MotionEstimation(**config["visual_odometry"]["motion_estimation"])
    )
    
    self.visualization = VisualizationConfig(**config['visualization'])
    limits = get_geometric_limitations(self.dataset.type, self.dataset.variant)
    self.visualization = self.visualization._replace(limits=limits)
