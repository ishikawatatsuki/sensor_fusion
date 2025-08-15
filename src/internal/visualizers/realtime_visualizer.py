import os
import time
import logging
import numpy as np
from PIL import Image
from typing import Tuple, Dict
from queue import Empty
import matplotlib.pyplot as plt
from matplotlib import axes, figure
import matplotlib.patches as mpatches
from collections import namedtuple
from dataclasses import dataclass
import multiprocessing as mp
from multiprocessing import Event

from ..extended_common import (
    VisualizationConfig, FilterConfig, Pose, VisualizationDataType, VisualizationMessage, VisualizationData
)

WindowParam = namedtuple(
    'WindowParam',
    ['label', 'xlabel', 'ylabel', 'tick_left', 'tick_bottom', 'patches'])

colors = ["black", "red", "blue", "green", "pink", "yellow"]

LINE_GRAPH_TYPES = [
    VisualizationDataType.ACCELEROMETER, 
    VisualizationDataType.GYROSCOPE, 
    VisualizationDataType.ANGLE, 
    VisualizationDataType.VELOCITY
    ]
XY_GRAPH_TYPES = [
    VisualizationDataType.GPS, 
    VisualizationDataType.VO, 
    VisualizationDataType.ESTIMATION, 
    VisualizationDataType.GROUND_TRUTH
    ]

COLOR_MAP = {
    VisualizationDataType.GROUND_TRUTH: "black",
    VisualizationDataType.GPS: "black",
    VisualizationDataType.VO: "red",
    VisualizationDataType.ESTIMATION: "blue"
}

x_color = "red"
y_color = "blue"
z_color = "green"

WINDOW_PARAMS = {
    VisualizationDataType.ACCELEROMETER:
    WindowParam(label="Acceleration [m/s2]",
                xlabel=None,
                ylabel="m/s2",
                tick_left=True,
                tick_bottom=False,
                patches=[
                    mpatches.Patch(color=x_color, label="Acc X [m/s^2]"),
                    mpatches.Patch(color=y_color, label="Acc Y [m/s^2]"),
                    mpatches.Patch(color=z_color, label="Acc Z [m/s^2]")
                ]),
    VisualizationDataType.GYROSCOPE:
    WindowParam(label="Gyroscope [rad/s]",
                xlabel=None,
                ylabel="rad/s",
                tick_left=True,
                tick_bottom=False,
                patches=[
                    mpatches.Patch(color=x_color, label="Gyro X [rad/s]"),
                    mpatches.Patch(color=y_color, label="Gyro Y [rad/s]"),
                    mpatches.Patch(color=z_color, label="Gyro Z [rad/s]")
                ]),
    VisualizationDataType.ANGLE:
    WindowParam(label="Estimated Angle [rad]",
                xlabel=None,
                ylabel="rad",
                tick_left=True,
                tick_bottom=False,
                patches=[
                    mpatches.Patch(color=x_color,
                                    label="Angle in X [rad]"),
                    mpatches.Patch(color=y_color,
                                    label="Angle in Y [rad]"),
                    mpatches.Patch(color=z_color, label="Angle in Z [rad]")
                ]),
    VisualizationDataType.VELOCITY:
    WindowParam(label="Estimated Velocity [m/s]",
                xlabel="Time (s)",
                ylabel="m/s",
                tick_left=True,
                tick_bottom=True,
                patches=[
                    mpatches.Patch(color=x_color,
                                    label="Velocity in X [m/s]"),
                    mpatches.Patch(color=y_color,
                                    label="Velocity in Y [m/s]"),
                    mpatches.Patch(color=z_color,
                                    label="Velocity in Z [m/s]")
                ]),
    VisualizationDataType.GPS:
    WindowParam(label="GPS",
                xlabel="X (m)",
                ylabel=" Y (m)",
                tick_left=False,
                tick_bottom=False,
                patches=[]),
    VisualizationDataType.VO:
    WindowParam(label="Visual Odometry",
                xlabel="X (m)",
                ylabel=" Y (m)",
                tick_left=False,
                tick_bottom=False,
                patches=[]),
    VisualizationDataType.ESTIMATION:
    WindowParam(label="Trajectory comparison",
                xlabel="X (m)",
                ylabel="Y (m)",
                tick_left=False,
                tick_bottom=False,
                patches=[
                    mpatches.Patch(color='black', label='GPS trajectory'),
                    mpatches.Patch(color='red',
                                    label='Visual odometry estimation'),
                    mpatches.Patch(color='blue',
                                    label='Estimated trajectory'),
                ]),
}

REFRESH_RATE = 0.5  # seconds


class RealtimeVisualizer:

    def __init__(self, config: VisualizationConfig):
        self.config = config
        self.data_queue = mp.Queue()  # Single shared queue
        
        if self.config.realtime:
            self.figure, self.windows = self._create_window()

        self.inactivity_timeout = 5.0
        self.debugging = self.config.realtime

        self.frame = 0
        self.output_frame_path = os.path.join(self.config.output_filepath, "frames")
        os.makedirs(self.output_frame_path, exist_ok=True)

        self.data_points = {t: [] for t in VisualizationDataType if t in LINE_GRAPH_TYPES + XY_GRAPH_TYPES}
        self.state = None
        self.initial_pose = Pose(R=np.eye(3), t=np.zeros(3))

        self.stop_event = Event()
        self.process = mp.Process(target=self.run)

    def _create_window(
            self
    ) -> Tuple[figure.Figure, Dict[VisualizationDataType, axes.Axes]]:

        def _setup_window(window: axes.Axes,
                          params: WindowParam,
                          legend_loc='upper left') -> axes.Axes:
            window.set(xlabel=params.xlabel, ylabel=params.ylabel)
            window.set_title(params.label)
            window.legend(handles=params.patches, loc=legend_loc)
            if params.tick_left or params.tick_bottom:
                window.tick_params(left=params.tick_left,
                                   bottom=params.tick_bottom)

            window.grid()
            return window

        plt.ion()  # Enable interactive mode

        figure = plt.figure(figsize=(14, 6))
        gs = figure.add_gridspec(ncols=14, nrows=9)

        window = dict()
        window[VisualizationDataType.IMAGE] = figure.add_subplot(gs[:4, :5])

        window[VisualizationDataType.ESTIMATION] = _setup_window(
            figure.add_subplot(gs[4:, :5]),
            WINDOW_PARAMS[VisualizationDataType.ESTIMATION],
            legend_loc='upper right')

        window[VisualizationDataType.ACCELEROMETER] = _setup_window(
            figure.add_subplot(gs[:2, 6:]),
            WINDOW_PARAMS[VisualizationDataType.ACCELEROMETER])

        window[VisualizationDataType.GYROSCOPE] = _setup_window(
            figure.add_subplot(gs[2:4, 6:]),
            WINDOW_PARAMS[VisualizationDataType.GYROSCOPE])

        window[VisualizationDataType.ANGLE] = _setup_window(
            figure.add_subplot(gs[4:6, 6:]),
            WINDOW_PARAMS[VisualizationDataType.ANGLE])

        window[VisualizationDataType.VELOCITY] = _setup_window(
            figure.add_subplot(gs[6:8, 6:]),
            WINDOW_PARAMS[VisualizationDataType.VELOCITY])

        figure.tight_layout()

        return figure, window

    def _visualize(self):
        """Visualize the data in the window"""
        # Visualize line graphs
        for t in LINE_GRAPH_TYPES:
            data = np.array(self.data_points[t])
            if data.shape[0] == 0:
                continue
            self.windows[t].plot(data[:, 0], data[:, 1], color=x_color)
            self.windows[t].plot(data[:, 0], data[:, 2], color=y_color)
            self.windows[t].plot(data[:, 0], data[:, 3], color=z_color)

            self.data_points[t] = self.data_points[t][-1:]

        # Visualize xy graphs
        for t in XY_GRAPH_TYPES:
            data = np.array(self.data_points[t])

            if data.shape[0] < 2:
                continue
            self.windows[VisualizationDataType.ESTIMATION].plot(data[:, 0], data[:, 1], color=COLOR_MAP.get(t), lw=1)
            self.data_points[t] = self.data_points[t][-1:]


        plt.draw()
        plt.pause(interval=0.01)

    def _save_current_frame(self):
        if self.config.save_frames and self.figure is not None:
            self.figure.savefig(
                os.path.join(self.output_frame_path, f"{str(self.frame)}.png"))
            self.frame += 1

    def _set_frame(self, data: VisualizationData):
        if not self.debugging:
            return
        
        image_path = os.path.abspath(data.extra)
        if os.path.exists(image_path):
            image = np.asarray(Image.open(image_path))
            self.windows[VisualizationDataType.IMAGE].clear()
            self.windows[VisualizationDataType.IMAGE].imshow(image)
            self.windows[VisualizationDataType.IMAGE].axis(
                "off")  # Hide axes for better viewing

        self._save_current_frame()

    def _set_general_data(self, message: VisualizationMessage):
        """Visualize IMU data & Angle, Velocity estimation."""

        if message.type not in LINE_GRAPH_TYPES:
            return
        
        self.data_points[message.type].append([message.timestamp, *message.data.data])

    def _set_estimation(self, message: VisualizationMessage):
        """Visualize GPS, VO and Estimated position"""
        if message.type not in XY_GRAPH_TYPES:
            return
        
        self.data_points[message.type].append([message.data.data[0], message.data.data[1]])

    def _handle_message(self, message: VisualizationMessage):
        match message.type:
            case VisualizationDataType.IMAGE:
                self._set_frame(data=message.data)
            case VisualizationDataType.ACCELEROMETER | VisualizationDataType.GYROSCOPE | \
                VisualizationDataType.VELOCITY | VisualizationDataType.ANGLE:
                self._set_general_data(message)
            case VisualizationDataType.GPS | VisualizationDataType.VO | \
                VisualizationDataType.ESTIMATION | VisualizationDataType.GROUND_TRUTH:
                self._set_estimation(message)
            case _:
                logging.warning(f"Unknown data type received: {message.type}")

    def run(self):
        POLL_TIMEOUT = 0.05  # seconds
        MAX_EMPTY_COUNT = int(self.inactivity_timeout / POLL_TIMEOUT)
        last_refresh_timestamp = -np.inf  # track timestamp of last refresh
        empty_count = 0

        while True:
            try:
                try:
                    message: VisualizationMessage = self.data_queue.get(timeout=POLL_TIMEOUT)
                    if message.type not in self.config.fields:
                        continue
                    
                    self._handle_message(message)

                    # Refresh plot every REFRESH_RATE seconds
                    if (message.timestamp - last_refresh_timestamp) >= REFRESH_RATE:
                        self._visualize()
                        last_refresh_timestamp = message.timestamp

                    empty_count = 0  # Reset on successful data

                except Empty:
                    empty_count += 1
                    if empty_count > MAX_EMPTY_COUNT:
                        print(f"No data received for {self.inactivity_timeout:.1f} seconds, stopping.")
                        break


            except Exception as e:
                logging.error(f"Visualization error: {e}")
                break

        if not self.debugging:
            self._visualize()
            plt.draw()
            plt.pause(interval=1)

        logging.info("Stopping visualization process...")
        time.sleep(1)

    def _drain_queues(self):
        while not self.data_queue.empty():
            try:
                data = self.data_queue.get_nowait()
                logging.debug(f"Draining: {data}")  # Process remaining data
            except Empty:
                break  # Queue is empty
            
    def get_queue(self) -> mp.Queue:
        return self.data_queue

    def start(self):
        self.process.start()

    def stop(self):
        self._drain_queues()  # Drain remaining data
        self.process.join()
        self.process.terminate()


if __name__ == "__main__":

    dataset_path = "/Volumes/Data_EXT/data/workspaces/sensor_fusion/data/KITTI/2011_10_03/2011_10_03_drive_0034_sync/image_02/data"
    filter_config = FilterConfig(
        type="ekf",
        dimension=3,
        motion_model="kinematics",
        noise_type=False,
        params=None,
        innovation_masking=False,
    )

    config = VisualizationConfig(
        realtime=True,
        output_filepath="./",
        save_trajectory=False,
        show_end_result=False,
        save_frames=False,
        show_vo_trajectory=False,
        show_vio_frame=False,
        show_particles=False,
        set_lim_in_plot=False,
        show_innovation_history=False,
        show_angle_estimation=False,
        limits=[],
        fields=[
            VisualizationDataType.VO,
            VisualizationDataType.ESTIMATION,
            VisualizationDataType.IMAGE,
            VisualizationDataType.GPS,
            VisualizationDataType.ACCELEROMETER,
            VisualizationDataType.GYROSCOPE,
            VisualizationDataType.VELOCITY,
            VisualizationDataType.ANGLE,
            VisualizationDataType.GROUND_TRUTH
        ])

    def _main():
        visualizer = RealtimeVisualizer(config=config)
        zline = np.linspace(0, 15, 100)
        xline = np.sin(zline)
        yline = np.cos(zline)

        xline2 = np.cos(zline)
        yline2 = np.sin(zline)

        queue = visualizer.data_queue

        visualizer.start()

        zline = zline[1:]
        xline = xline[1:]
        yline = yline[1:]
        xline2 = xline2[1:]
        yline2 = yline2[1:]
        
        if queue is None:
            logging.error("Queue is None, cannot put data.")
            return
        
        i = 0
        while True:
            
            for j in range(10):
                imu_t = i + j / 10
                acc = np.random.normal(0, 0.5, 3) + np.array([0, 0, -9.81])
                acc_message = VisualizationMessage(
                    timestamp=imu_t,
                    type=VisualizationDataType.ACCELEROMETER,
                    data=VisualizationData(data=acc)
                )
                gyr_message = VisualizationMessage(
                    timestamp=imu_t,
                    type=VisualizationDataType.GYROSCOPE,
                    data=VisualizationData(data=np.random.normal(0, 0.5, 3))
                )
                queue.put(acc_message)
                queue.put(gyr_message)

            for j in range(10):
                est_t = i + j / 10
                acc = np.random.normal(0, 1, 3)
                acc_message = VisualizationMessage(
                    timestamp=est_t,
                    type=VisualizationDataType.ANGLE,
                    data=VisualizationData(data=acc)
                )
                gyr_message = VisualizationMessage(
                    timestamp=est_t,
                    type=VisualizationDataType.VELOCITY,
                    data=VisualizationData(data=np.random.normal(0, 1, 3))
                )
                queue.put(acc_message)
                queue.put(gyr_message)

            vo = np.array([xline[i], yline[i], zline[i]])
            estimate = np.array([xline2[i], yline2[i], zline[i]])

            estimate_message = VisualizationMessage(
                timestamp=i,
                type=VisualizationDataType.ESTIMATION,
                data=VisualizationData(data=estimate)
            )
            queue.put(estimate_message)

            vo_message = VisualizationMessage(
                timestamp=i,
                type=VisualizationDataType.VO,
                data=VisualizationData(data=vo)
            )
            queue.put(vo_message)

            image_path = os.path.join(
                dataset_path,
                f"{i+1:010}.png")
            
            logging.debug(f"Putting image {image_path} to queue")
            image_message = VisualizationMessage(
                timestamp=i,
                type=VisualizationDataType.IMAGE,
                data=VisualizationData(data=np.empty(0), extra=image_path)
            )
            queue.put(image_message)

            time.sleep(0.01)
            if i < len(zline) - 1:
                i += 1
            else:
                time.sleep(10)
                break

        visualizer.stop()

    _main()
