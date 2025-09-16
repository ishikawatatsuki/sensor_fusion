import os
import sys
import abc
import cv2
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
    ImageData,
    SensorDataField,
    VisualizationData
)
from .visual_odometry import VisualOdometry
from .internal.dataset import Dataset
from .internal.extended_common import (
    ExtendedConfig,
    CoordinateFrame
)
from .internal.visualizers import (
    VisualizationMessage, VisualizationDataType, RealtimeVisualizer
)
from .internal.error_reporter import ErrorReporter, ErrorReportType, ReportMessage
from .utils.time_reporter import time_reporter
from .utils.geometric_transformer import TransformationField
from .utils.geometric_transformer.base_geometric_transformer import BaseGeometryTransformer
from .utils.data_logger import DataLogger, LoggingData, LoggingMessage
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
        
        self.data_logger = DataLogger(
            config=config.general,
            dataset_config=config.dataset
        )
        self.data_logger.prepare()

        self.dataset = Dataset(config=config.dataset)
        
        self.sensor_fusion = SensorFusion(
            filter_config=config.filter,
            hardware_config=config.hardware,
            data_logger=self.data_logger
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

        self.visualizer = RealtimeVisualizer(config=config.visualization, dataset_config=config.dataset)

        initial_state = self.dataset.get_initial_state(filter_config=config.filter)
        self.sensor_fusion.set_initial_state(initial_state=initial_state)

        self.visualization_queue = self.visualizer.get_queue()


        logging.info("Initialization completed.")

    def _visualize_data(self, message: VisualizationMessage):
        """Visualize data including sensors, estimation, and etc."""
        self.visualization_queue.put(message)
        
    def _prepare_visualization(self, response: FusionResponse):
        if response is None or not self.config.visualization.realtime:
            return
        
        if response.imu_acceleration is not None:
            self._visualize_data(
                message=VisualizationMessage(
                    type=VisualizationDataType.ACCELEROMETER,
                    timestamp=response.timestamp,
                    data=VisualizationData(
                        data=response.imu_acceleration,
                    ),
                )
            )
        if response.imu_angular_velocity is not None:
            self._visualize_data(
                message=VisualizationMessage(
                    type=VisualizationDataType.GYROSCOPE,
                    timestamp=response.timestamp,
                    data=VisualizationData(
                        data=response.imu_angular_velocity,
                    ),
                )
            )

        if response.estimated_angle is not None:
            self._visualize_data(
                message=VisualizationMessage(
                    type=VisualizationDataType.ANGLE,
                    timestamp=response.timestamp,
                    data=VisualizationData(
                        data=response.estimated_angle
                    )
                )
            )
        
        if response.estimated_linear_velocity is not None:
            self._visualize_data(
                message=VisualizationMessage(
                    type=VisualizationDataType.VELOCITY,
                    timestamp=response.timestamp,
                    data=VisualizationData(
                        data=response.estimated_linear_velocity
                    )
                )
            )
        
        if response.pose is not None:
            data = response.pose[:3, 3].flatten()
            # data = np.array([data[1], data[2], data[0]])
            self._visualize_data(
                message=VisualizationMessage(
                    type=VisualizationDataType.ESTIMATION,
                    timestamp=response.timestamp,
                    data=VisualizationData(
                        data=data
                    )
                )
            )
        
        if response.gps_data is not None:
            self._visualize_data(
                message=VisualizationMessage(
                    type=VisualizationDataType.GPS,
                    timestamp=response.timestamp,
                    data=VisualizationData(
                        data=response.gps_data
                    )
                )
            )

        if response.vo_data is not None:
            self._visualize_data(
                message=VisualizationMessage(
                    type=VisualizationDataType.VO,
                    timestamp=response.timestamp,
                    data=VisualizationData(
                        data=response.vo_data
                    )
                )
            )

        if response.leica_data is not None:
            self._visualize_data(
                message=VisualizationMessage(
                    type=VisualizationDataType.LEICA,
                    timestamp=response.timestamp,
                    data=VisualizationData(
                        data=response.leica_data
                    )
                )
            )

    def run(self):
        
        self.dataset.start()
        self.visualizer.start()
        
        if self.config.general.log_sensor_data:
            f = open(self.config.general.sensor_data_output_filepath, "w")

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
                    # Visualize the left image
                    self._visualize_data(
                        message=VisualizationMessage(
                            type=VisualizationDataType.IMAGE,
                            timestamp=sensor_data.timestamp,
                            data=VisualizationData(
                                data=np.empty(0),
                                extra=sensor_data.data.left_frame_id
                            )
                        )
                    )

                    if self.config.dataset.should_run_visual_odometry:
                        # Run visual odometry
                        frame = cv2.imread(sensor_data.data.left_frame_id)
                        image_data = ImageData(image=frame, timestamp=sensor_data.timestamp)
                        vo_response = self.visual_odometry.compute_pose(data=image_data)
                        if vo_response.success:
                            _sensor_data = SensorDataField(
                                type=self.visual_odometry.get_datatype, 
                                timestamp=vo_response.estimate_timestamp, 
                                data=vo_response,
                                coordinate_frame=CoordinateFrame.STEREO_LEFT)
                            response, duration = self.sensor_fusion.run_measurement_update(_sensor_data)
                            self._prepare_visualization(response)

                            if is_debugging:
                                measurement_update_step_durations.append(duration)

                elif SensorType.is_reference_data(sensor_data.type):
                    # For visualization
                    if config.dataset.type == "kitti":
                        value = sensor_data.data.z
                        gt_inertial = self.sensor_fusion.geo_transformer.transform(fields=TransformationField(
                            state=self.sensor_fusion.kalman_filter.x,
                            value=value,
                            coord_from=CoordinateFrame.STEREO_LEFT,
                            coord_to=CoordinateFrame.INERTIAL))
                        data = gt_inertial[:3, 3].flatten()

                    elif config.dataset.type == "euroc":
                        gt = sensor_data.data.z.flatten()
                        data = gt[:3]
                        
                    else:
                        continue
                    
                    response = self.sensor_fusion.get_current_estimate(sensor_data.timestamp)
                    self._prepare_visualization(response)

                    self._visualize_data(
                        message=VisualizationMessage(
                            type=VisualizationDataType.GROUND_TRUTH,
                            timestamp=sensor_data.timestamp,
                            data=VisualizationData(
                                data=data
                            )
                        )
                    )

                    # NOTE: this works only for KITTI dataset
                    # current_estimate = self.sensor_fusion.kalman_filter.get_current_estimate().t.flatten()
                    # estimated = np.array([current_estimate[0], current_estimate[1], current_estimate[2]])
                    # report1 = ReportMessage(t=VisualizationDataType.ESTIMATION, timestamp=sensor_data.timestamp, value=estimated)
                    # report2 = ReportMessage(t=VisualizationDataType.GPS, timestamp=sensor_data.timestamp, value=sensor_data.data.z)
                    # self.error_reporter.enqueue_data(
                    #     y=report1,
                    #     y_hat=report2
                    # )

                self.data_logger.log(
                    message=LoggingMessage(
                        sensor_type=sensor_data.type,
                        timestamp=sensor_data.timestamp,
                        data=sensor_data.data
                    ),
                    is_raw=True
                )

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

    filename = args.config_file
    config = ExtendedConfig(config_filepath=filename)

    pipeline = SingleThreadedPipeline(
        config=config,
        early_stop=args.early_stop
    )
    pipeline.run()