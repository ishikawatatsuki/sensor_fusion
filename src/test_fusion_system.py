import os
import sys
import yaml
import json
import logging
import argparse
import numpy as np
import pandas as pd
from queue import Queue
from time import sleep
from typing import Dict
import multiprocessing as mp

from .misc import setup_logging, parse_args
from .fusion_system import FusionSystem
from .common import SensorType, VisualizationDataType, VisualizationData
from .common.datatypes import FusionResponse

from .internal.error_reporter import ErrorReporter, ReportInterface, ReportMessage
from .internal.visualizers import RealtimeVisualizer
from .internal.extended_common import ExtendedConfig, State
from .internal.dataset import Dataset


initial_timestamp = None

def visualize_data(visualization_queue: Dict[SensorType, mp.Queue], 
                   visualization_type: VisualizationDataType, 
                   data: VisualizationData):
        """Visualize data including sensors, estimation, and etc."""
        global initial_timestamp

        queue = visualization_queue.get(visualization_type)
        if queue is None:
            logging.warning(f"Visualization queue not found for type: {visualization_type.name}")
            return
        if initial_timestamp is None:
            initial_timestamp = data.timestamp

        # NOTE: Convert timestamp into elapsed time
        data.timestamp = (data.timestamp - initial_timestamp) / 1e9
        queue.put(data)


def run_kalman_filter(args: argparse.Namespace):
    """Test Kalman Filter with noisy sensor data.
    This function is used for experimenting with the Kalman Filter module.
    """ 
    
    setup_logging(args.log_level, args.log_output)
    logging.info("Starting pipeline")
    logging.debug("Debugging information")
    logging.warning("Warning message")
    logging.error("Error message")
    logging.critical("Critical error")
    
    filename = args.config_file
    config = ExtendedConfig(config_filepath=filename)

    logging.debug(config.hardware)
    logging.debug(config.filter)

    visualizer = RealtimeVisualizer(config=config.visualization)

    # Create key value pair of visualization queue
    visualization_queue = {
        viz.type: viz.queue
        for viz in visualizer.interface.queues
    }

    def _visualize_data(response: FusionResponse):
        if response is None or not config.visualization.realtime:
            return
        
        if response.imu_acceleration is not None:
            visualize_data(
                visualization_queue=visualization_queue,
                visualization_type=VisualizationDataType.ACCELEROMETER,
                data=VisualizationData(
                    timestamp=response.timestamp,
                    data=response.imu_acceleration
                )
            )
        if response.imu_angular_velocity is not None:
            visualize_data(
                visualization_queue=visualization_queue,
                visualization_type=VisualizationDataType.GYROSCOPE,
                data=VisualizationData(
                    timestamp=response.timestamp,
                    data=response.imu_angular_velocity
                )
            )
        
        if response.estimated_angle is not None:
            visualize_data(
                visualization_queue=visualization_queue,
                visualization_type=VisualizationDataType.ANGLE,
                data=VisualizationData(
                    timestamp=response.timestamp,
                    data=response.estimated_angle
                )
            )
        
        if response.estimated_linear_velocity is not None:
            visualize_data(
                visualization_queue=visualization_queue,
                visualization_type=VisualizationDataType.VELOCITY,
                data=VisualizationData(
                    timestamp=response.timestamp,
                    data=response.estimated_linear_velocity
                )
            )
        
        if response.pose is not None:
            data = response.pose[:3, 3].flatten()
            # data = np.array([data[1], data[2], data[0]])
            visualize_data(
                visualization_queue=visualization_queue,
                visualization_type=VisualizationDataType.ESTIMATION,
                data=VisualizationData(
                    timestamp=response.timestamp,
                    data=data
                )
            )
        
        if response.gps_data is not None:
            visualize_data(
                visualization_queue=visualization_queue,
                visualization_type=VisualizationDataType.GPS,
                data=VisualizationData(
                    timestamp=response.timestamp,
                    data=response.gps_data
                )
            )

        if response.leica_data is not None:
            data = response.leica_data.flatten()
            # data = np.array([data[1], data[2], data[0]])
            visualize_data(
                visualization_queue=visualization_queue,
                visualization_type=VisualizationDataType.LEICA,
                data=VisualizationData(
                    timestamp=response.timestamp,
                    data=data
                )
            )
        
        if response.geo_fencing_data is not None:
            visualize_data(
                visualization_queue=visualization_queue,
                visualization_type=VisualizationDataType.BEACON,
                data=VisualizationData(
                    timestamp=response.timestamp,
                    data=response.geo_fencing_data
                )
            )

        if response.vo_data is not None:
            visualize_data(
                visualization_queue=visualization_queue,
                visualization_type=VisualizationDataType.VO,
                data=VisualizationData(
                    timestamp=response.timestamp,
                    data=response.vo_data
                )
            )

    error_report_queue = Queue()
    reporter = ErrorReporter(
        report_config=config.report,
        filter_config=config.filter,
        hardware_config=config.hardware,
        interface=ReportInterface(queue=error_report_queue))

    dataset = Dataset(config=config.dataset)

    # Initialize Fusion System
    fusion_system = FusionSystem(
        filter_config=config.filter,
        hardware_config=config.hardware,
    )
    initial_state = dataset.get_initial_state(filter_config=config.filter)
    fusion_system.set_initial_state(initial_state=initial_state)

    logging.info("Initialization completed.")

    # Start both thread
    dataset.start()
    visualizer.start()
    reporter.start()
    
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
            
            if dataset.is_queue_empty():
                logging.debug("Dataset queue is empty")
                break

            # count += 1
            # if count > 500:
            #     break

            sensor_data = dataset.get_sensor_data()
            if sensor_data is None:
                continue

            logging.debug(
                f"Sensor: {sensor_data.type.name} at {sensor_data.timestamp}")
            
            if SensorType.is_time_update(sensor_data.type):
                response, duration = fusion_system.run_time_update(sensor_data)
                _visualize_data(response)

                if is_debugging:
                    time_update_step_durations.append(duration)

            elif SensorType.is_measurement_update(sensor_data.type):
                response, duration = fusion_system.run_measurement_update(sensor_data)
                _visualize_data(response)

                if is_debugging:
                    measurement_update_step_durations.append(duration)

            elif SensorType.is_stereo_image_data(sensor_data.type):
                visualize_data(
                    visualization_queue=visualization_queue,
                    visualization_type=VisualizationDataType.IMAGE,
                    data=VisualizationData(data=np.zeros(1),
                                            timestamp=sensor_data.timestamp,
                                            extra=sensor_data.data.left_frame_id)
                )
            
            elif SensorType.is_reference_data(sensor_data.type):
                if config.dataset.type == "uav":
                    continue
                elif config.dataset.type == "kitti":
                    data = sensor_data.data.z[:3, 3].flatten()
                else:
                    data = sensor_data.data.z.flatten()[:3]
                    # data = np.array([data[1], data[2], data[0]])
                    # fusion_system.kalman_filter.x.q =
                    # print(sensor_data.data.z.flatten()[6:].reshape(-1, 1))
                    # visualize_data(
                    #     visualization_queue=visualization_queue,
                    #     visualization_type=VisualizationDataType.VELOCITY,
                    #     data=VisualizationData(data=sensor_data.data.z.flatten()[3:6],
                    #                             timestamp=sensor_data.timestamp,
                    #                             extra=None)
                    # )
                    # euler = State.get_euler_angle_from_quaternion_vector(sensor_data.data.z[6:10])
                    # visualize_data(
                    #     visualization_queue=visualization_queue,
                    #     visualization_type=VisualizationDataType.ANGLE,
                    #     data=VisualizationData(data=euler,
                    #                             timestamp=sensor_data.timestamp,
                    #                             extra=None)
                    # )

                visualize_data(
                    visualization_queue=visualization_queue,
                    visualization_type=VisualizationDataType.GROUND_TRUTH,
                    data=VisualizationData(data=data,
                                            timestamp=sensor_data.timestamp,
                                            extra=None)
                )

            if config.dataset.type == "kitti" and\
                SensorType.is_gps_data(sensor_data.type):
                # NOTE: this works only for KITTI dataset
                current_estimate = fusion_system.kalman_filter.get_current_estimate().t.flatten()
                estimated = np.array([current_estimate[0], current_estimate[1], current_estimate[2]])
                report1 = ReportMessage(t=VisualizationDataType.ESTIMATION, timestamp=sensor_data.timestamp, value=estimated)
                report2 = ReportMessage(t=VisualizationDataType.GPS, timestamp=sensor_data.timestamp, value=sensor_data.data.z)
                reporter.interface.queue.put(report1)
                reporter.interface.queue.put(report2)

            if config.general.log_sensor_data:
                f.write(f"[{dataset.dataset.output_queue.qsize():05}] Sensor: {sensor_data.type.name} at {sensor_data.timestamp}\n")

    # except Exception as e:
    #     logging.error(e)
    #     logging.error(dataset.output_queue.qsize())
    finally:
        f.close()
        dataset.stop()
        reporter.stop()
        logging.info("Process finished.")

    if is_debugging and len(time_update_step_durations) > 0:
        logging.info(
            f"Average time update step: {np.mean(time_update_step_durations) * 1e3:.3f}ms"
        )  #CKF: 0.65ms
        logging.info(
            f"Average measurement update step: {np.mean(measurement_update_step_durations) * 1e3:.3f}ms"
        )  #CKF: 0.30ms

    print("Press control +c to stop the process.")
    try:
        while True:
            continue
    except KeyboardInterrupt:
        logging.info("Process finished completely.")
    finally:
        if config.visualization.show_innovation_history:
            RealtimeVisualizer.show_innovation_history(fusion_system.kalman_filter.innovations)
        visualizer.stop()



if __name__ == "__main__":

    args = parse_args()
    run_kalman_filter(args)