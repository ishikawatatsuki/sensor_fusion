import os
import sys
import abc
import logging
from time import sleep
import numpy as np
from typing import List

from .common import (
  Config,
  Pose, State,
  SensorType,
  FilterType,
  NoiseType,
  DatasetType,
  FusionResponse,
  VisualizationData
)
from .visual_odometry import VisualOdometry
from .internal.dataset import Dataset
from .internal.extended_common import (
  ExtendedConfig
)
from .internal.visualizers import (
  BaseVisualizationField, PoseVisualizationField, VisualizationDataType, RealtimeVisualizer
)
from .internal.error_reporter import ErrorReporter, ErrorReportType, ReportMessage
from .utils.time_reporter import time_reporter
from .misc import setup_logging, parse_args
from .sensor_fusion import SensorFusion

class SingleThreadedPipeline(abc.ABC):
    def __init__(
        self,
        config: ExtendedConfig,
        early_stop: bool=False,
    ):
        self.config = config
        self.log_level = config.general.log_level
        self.early_stop = early_stop
        
        self.dataset = Dataset(config=config.dataset)
        self.sensor_fusion = SensorFusion(
            filter_config=config.filter,
            hardware_config=config.hardware
        )
        self.visual_odometry = VisualOdometry(
            config=config.visual_odometry,
            dataset_config=config.dataset,
            debug=self.log_level.lower() == "debug"
        )

        # self.error_reporter = ErrorReporter(
        #     report_config=config.report, 
        #     dataset_config=config.dataset
        # )
        
        self.visualizer = RealtimeVisualizer(config=config.visualization)

        initial_state = self.dataset.get_initial_state(filter_config=config.filter)
        self.sensor_fusion.set_initial_state(initial_state=initial_state)

        self.timestamp_divider = 1e9
        self.visualization_initial_timestamp = None
        self.visualization_queue = {
            viz.type: viz.queue for viz in self.visualizer.interface.queues
        }


        logging.info("Initialization completed.")

    def _visualize_data(self, visualization_type: VisualizationDataType, data: VisualizationData):
        """Visualize data including sensors, estimation, and etc."""

        queue = self.visualization_queue.get(visualization_type)
        if queue is None:
            logging.warning(f"Visualization queue not found for type: {visualization_type.name}")
            return
        
        if self.visualization_initial_timestamp is None:
            self.visualization_initial_timestamp = data.timestamp

        # NOTE: Convert timestamp into elapsed time
        data.timestamp = (data.timestamp - self.visualization_initial_timestamp) / self.timestamp_divider
        queue.put(data)
        
    def _prepare_visualization(self, response: FusionResponse):
        if response is None or not self.config.visualization.realtime:
            return
        
        if response.imu_acceleration is not None:
            self._visualize_data(
                visualization_type=VisualizationDataType.ACCELEROMETER,
                data=VisualizationData(
                    timestamp=response.timestamp,
                    data=response.imu_acceleration
                )
            )
        if response.imu_angular_velocity is not None:
            self._visualize_data(
                visualization_type=VisualizationDataType.GYROSCOPE,
                data=VisualizationData(
                    timestamp=response.timestamp,
                    data=response.imu_angular_velocity
                )
            )
        
        if response.estimated_angle is not None:
            self._visualize_data(
                visualization_type=VisualizationDataType.ANGLE,
                data=VisualizationData(
                    timestamp=response.timestamp,
                    data=response.estimated_angle
                )
            )
        
        if response.estimated_linear_velocity is not None:
            self._visualize_data(
                visualization_type=VisualizationDataType.VELOCITY,
                data=VisualizationData(
                    timestamp=response.timestamp,
                    data=response.estimated_linear_velocity
                )
            )
        
        if response.pose is not None:
            data = response.pose[:3, 3].flatten()
            # data = np.array([data[1], data[2], data[0]])
            self._visualize_data(
                visualization_type=VisualizationDataType.ESTIMATION,
                data=VisualizationData(
                    timestamp=response.timestamp,
                    data=data
                )
            )
        
        if response.gps_data is not None:
            self._visualize_data(
                visualization_type=VisualizationDataType.GPS,
                data=VisualizationData(
                    timestamp=response.timestamp,
                    data=response.gps_data
                )
            )

        if response.leica_data is not None:
            data = response.leica_data.flatten()
            # data = np.array([data[1], data[2], data[0]])
            self._visualize_data(
                visualization_type=VisualizationDataType.LEICA,
                data=VisualizationData(
                    timestamp=response.timestamp,
                    data=data
                )
            )
        
        if response.geo_fencing_data is not None:
            self._visualize_data(
                visualization_type=VisualizationDataType.BEACON,
                data=VisualizationData(
                    timestamp=response.timestamp,
                    data=response.geo_fencing_data
                )
            )

        if response.vo_data is not None:
            self._visualize_data(
                visualization_type=VisualizationDataType.VO,
                data=VisualizationData(
                    timestamp=response.timestamp,
                    data=response.vo_data
                )
            )


    def run(self):
        
        self.dataset.start()
        self.visualizer.start()
        
        if config.general.log_sensor_data:
            f = open(config.general.sensor_data_output_filepath, "w")
        
        sleep(0.5)
        logging.info("Starting the process.")
        is_debugging = logging.root.level is logging.DEBUG
        time_update_step_durations = []
        measurement_update_step_durations = []
        count = 0
        try:
            while True:
                
                if self.dataset.is_queue_empty():
                    logging.debug("Dataset queue is empty")
                    break

                count += 1
                if count > 100 and self.early_stop:
                    break

                sensor_data = self.dataset.get_sensor_data()
                if sensor_data is None:
                    continue

                logging.debug(
                    f"Sensor: {sensor_data.type.name} at {sensor_data.timestamp}")
                
                if SensorType.is_time_update(sensor_data.type):
                    response, duration = self.sensor_fusion.run_time_update(sensor_data)
                    self._prepare_visualization(response)

                    if is_debugging:
                        time_update_step_durations.append(duration)

                elif SensorType.is_measurement_update(sensor_data.type):
                    response, duration = self.sensor_fusion.run_measurement_update(sensor_data)
                    self._prepare_visualization(response)

                    if is_debugging:
                        measurement_update_step_durations.append(duration)

                elif SensorType.is_stereo_image_data(sensor_data.type):
                    self._visualize_data(
                        visualization_type=VisualizationDataType.IMAGE,
                        data=VisualizationData(
                            data=np.zeros(1),
                            timestamp=sensor_data.timestamp,
                            extra=sensor_data.data.left_frame_id)
                    )
                
                elif SensorType.is_reference_data(sensor_data.type):
                    # For visualization
                    if config.dataset.type == "kitti":
                        data = sensor_data.data.z[:3, 3].flatten()
                    elif config.dataset.type == "euroc":
                        data = sensor_data.data.z.flatten()[:3]
                    else:
                        continue
                    
                    self._visualize_data(
                        visualization_type=VisualizationDataType.GROUND_TRUTH,
                        data=VisualizationData(data=data,
                                                timestamp=sensor_data.timestamp,
                                                extra=None)
                    )
                    
                    # NOTE: this works only for KITTI dataset
                    current_estimate = self.sensor_fusion.kalman_filter.get_current_estimate().t.flatten()
                    estimated = np.array([current_estimate[0], current_estimate[1], current_estimate[2]])
                    report1 = ReportMessage(t=VisualizationDataType.ESTIMATION, timestamp=sensor_data.timestamp, value=estimated)
                    report2 = ReportMessage(t=VisualizationDataType.GPS, timestamp=sensor_data.timestamp, value=sensor_data.data.z)
                    # self.error_reporter.enqueue_data(
                    #     y=report1,
                    #     y_hat=report2
                    # )

                if config.general.log_sensor_data:
                    f.write(f"[{self.dataset.get_queue_size():05}] Sensor: {sensor_data.type.name} at {sensor_data.timestamp}\n")

        # except Exception as e:
        #     logging.error(e)
        #     logging.error(f"Data remaining in queue: {self.dataset.get_queue_size()}")
        finally:
            f.close()
            self.dataset.stop()
            logging.info("Process finished.")

        if is_debugging and len(time_update_step_durations) > 0:
            logging.info(
                f"Average time update step: {np.mean(time_update_step_durations) * 1e3:.3f}ms"
            )  #CKF: 0.65ms
            logging.info(
                f"Average measurement update step: {np.mean(measurement_update_step_durations) * 1e3:.3f}ms"
            )  #CKF: 0.30ms

        logging.critical("Press control +c to stop the process.")
        try:
            while True:
                continue
        except KeyboardInterrupt:
            logging.info("Process finished completely.")
        finally:
            if config.visualization.show_innovation_history:
                RealtimeVisualizer.show_innovation_history(self.sensor_fusion.kalman_filter.innovations)
            self.visualizer.stop()

if __name__ == "__main__":
  
    args = parse_args()
    setup_logging(log_level=args.log_level, log_output=args.log_output)
    logging.info("Starting pipeline")
    logging.debug("Debugging information")
    logging.warning("Warning message")
    logging.error("Error message")
    logging.critical("Critical error")


    filename = args.config_file
    config = ExtendedConfig(config_filepath=filename)

    pipeline = SingleThreadedPipeline(
        config=config,
        early_stop=args.early_stop
    )
    pipeline.run()