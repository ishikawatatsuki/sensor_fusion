import os
import sys
import uuid
import logging
import itertools
import numpy as np
import pandas as pd
from PIL import Image
import seaborn as sns
from queue import PriorityQueue
import threading
from threading import Event as ThreadingEvent
from typing import List, Tuple, Dict
import time
from enum import IntEnum, auto
import matplotlib.pyplot as plt
from matplotlib import axes, figure
import matplotlib.patches as mpatches
from collections import namedtuple
from dataclasses import dataclass
import multiprocessing as mp
from multiprocessing import Event
from multiprocessing.connection import wait

from ..extended_common import (
    VisualizationConfig, FilterConfig, Pose, VisualizationDataType, VisualizerInterface, VisualizationQueue, VisualizationData
)

@dataclass
class VisualizationInnerData:
    counter: int
    type: VisualizationDataType
    data: VisualizationData

    def __gt__(self, other: VisualizationDataType):
        return self.counter > other.counter
    def __lt__(self, other: VisualizationDataType):
        return self.counter < other.counter

WindowParam = namedtuple(
    'WindowParam',
    ['label', 'xlabel', 'ylabel', 'tick_left', 'tick_bottom', 'patches'])

x_color = "red"
y_color = "blue"
z_color = "green"
colors = ["black", "red", "blue", "green", "pink", "yellow"]

LINE_GRAPH_TYPES = [VisualizationDataType.ACCELEROMETER, VisualizationDataType.GYROSCOPE, VisualizationDataType.ANGLE, VisualizationDataType.VELOCITY]
XY_GRAPH_TYPES = [VisualizationDataType.GPS, VisualizationDataType.VO, VisualizationDataType.ESTIMATION, VisualizationDataType.GROUND_TRUTH, VisualizationDataType.LEICA]


REFRESH_RATE = 0.5  # seconds

class RealtimeVisualizer:
    x_color = "red"
    y_color = "blue"
    z_color = "green"

    colors = ["black", "red", "blue", "green", "pink", "yellow"]
    color_map = {
        VisualizationDataType.GROUND_TRUTH: "black",
        VisualizationDataType.GPS: "black",
        VisualizationDataType.VO: "red",
        VisualizationDataType.LEICA: "red",
        VisualizationDataType.ESTIMATION: "blue"
    }
    window_params = {
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

    def __init__(self, config: VisualizationConfig):
        self.config = config
        self.interface = self._get_interface()
        self.figure, self.windows = self._create_window()
        self.renderer = self.figure.canvas.renderer
        self.visualization_queues = [q.queue for q in self.interface.queues]

        self.inactivity_timeout = 5.
        self.debugging = self.config.realtime

        self.frame = 0
        self.output_frame_path = os.path.join(self.config.output_filepath,
                                              "frames")
        if not os.path.exists(self.output_frame_path):
            os.makedirs(self.output_frame_path)

        self.prev_datapoint = {
            VisualizationDataType.ACCELEROMETER: None,
            VisualizationDataType.GYROSCOPE: None,
            VisualizationDataType.ANGLE: None,
            VisualizationDataType.VELOCITY: None,
            VisualizationDataType.ESTIMATION: None,
            VisualizationDataType.GPS: None,
            VisualizationDataType.VO: None,
            VisualizationDataType.GROUND_TRUTH: None,
        }

        self.data_points = {
            VisualizationDataType.ACCELEROMETER: [],
            VisualizationDataType.GYROSCOPE: [],
            VisualizationDataType.ANGLE: [],
            VisualizationDataType.VELOCITY: [],
            VisualizationDataType.ESTIMATION: [],
            VisualizationDataType.GPS: [],
            VisualizationDataType.VO: [],
            VisualizationDataType.GROUND_TRUTH: [],
            VisualizationDataType.LEICA: [],
        }

        self.state = None

        self.initial_pose = Pose(R=np.eye(3), t=np.zeros(3))

        self.stop_event = Event()
        self.process = mp.Process(target=self.run)

        # self.visualization_process = threading.Thread(target=self._visualize)
        # self.visualization_process.daemon = True

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
            self.window_params[VisualizationDataType.ESTIMATION],
            legend_loc='upper right')

        window[VisualizationDataType.ACCELEROMETER] = _setup_window(
            figure.add_subplot(gs[:2, 6:]),
            self.window_params[VisualizationDataType.ACCELEROMETER])

        window[VisualizationDataType.GYROSCOPE] = _setup_window(
            figure.add_subplot(gs[2:4, 6:]),
            self.window_params[VisualizationDataType.GYROSCOPE])

        window[VisualizationDataType.ANGLE] = _setup_window(
            figure.add_subplot(gs[4:6, 6:]),
            self.window_params[VisualizationDataType.ANGLE])

        window[VisualizationDataType.VELOCITY] = _setup_window(
            figure.add_subplot(gs[6:8, 6:]),
            self.window_params[VisualizationDataType.VELOCITY])

        figure.tight_layout()

        return figure, window

    def _visualize(self):
        """Visualize the data in the window"""
        # Visualize line graphs
        for t in LINE_GRAPH_TYPES:
            data = np.array(self.data_points[t])
            if data.shape[0] == 0:
                continue
            self.windows[t].plot(data[:, 0], data[:, 1], color=self.x_color)
            self.windows[t].plot(data[:, 0], data[:, 2], color=self.y_color)
            self.windows[t].plot(data[:, 0], data[:, 3], color=self.z_color)

            self.data_points[t] = self.data_points[t][-1:]

        # Visualize xy graphs
        for t in XY_GRAPH_TYPES:
            data = np.array(self.data_points[t])

            if data.shape[0] < 2:
                continue
            self.windows[VisualizationDataType.ESTIMATION].plot(data[:, 0], data[:, 1], color=self.color_map.get(t), lw=1)
            self.data_points[t] = self.data_points[t][-1:]


        plt.draw()
        plt.pause(interval=0.01)
        

    def _get_interface(self) -> VisualizerInterface:
        queues = []
        for field in self.config.fields:
            logging.info(f"Creating queue for {field.name}")
            queues.append(
                VisualizationQueue(name=field.name, type=field, queue=mp.Queue()))
            
        return VisualizerInterface(queues=queues, num_cols=1, window_title="Trajectory comparison")

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

            plt.draw()
            plt.pause(interval=0.01)
        self._visualize()
        self._save_current_frame()

    def _set_general_data(self, t: VisualizationDataType,
                          data: VisualizationData):
        """Visualize IMU data & Angle, Velocity estimation."""

        if t not in LINE_GRAPH_TYPES:
            return
        
        self.data_points[t].append([data.timestamp, *data.data])

        # if self.prev_datapoint[t] is None:
        #     self.prev_datapoint[t] = VisualizationData(
        #         data=data.data, timestamp=data.timestamp)
        #     return

        # point = [[
        #     prev, current
        # ] for prev, current in zip(self.prev_datapoint[t].data, data.data)]
        # ts = [self.prev_datapoint[t].timestamp, data.timestamp]

        # self.windows[t].plot(ts, point[0], color=self.x_color)
        # self.windows[t].plot(ts, point[1], color=self.y_color)
        # self.windows[t].plot(ts, point[2], color=self.z_color)

        # self.prev_datapoint[t] = data
        # plt.draw()
        # plt.pause(interval=0.01)

    def _set_estimation(self, t: VisualizationDataType,
                        data: VisualizationData):
        """Visualize GPS, VO and Estimated position"""
        if t not in XY_GRAPH_TYPES:
            return

        self.data_points[t].append([data.data[0], data.data[1]])

        # color = self.color_map.get(t)
        # if color is None:
        #     return

        # if self.prev_datapoint[t] is None:
        #     self.prev_datapoint[t] = VisualizationData(
        #         data=data.data, timestamp=data.timestamp)
        #     return

        # window_t = VisualizationDataType.ESTIMATION
        # points = [[
        #     prev, current
        # ] for prev, current in zip(self.prev_datapoint[t].data, data.data)]

        # self.windows[window_t].plot(points[0], points[1], color=color, lw=1)

        # self.prev_datapoint[t] = VisualizationData(data=data.data,
        #                                            timestamp=data.timestamp)
        # plt.draw()
        # plt.pause(interval=0.01)

    def _set_beacon_data(self, t: VisualizationDataType, data: VisualizationData):

        window_t = VisualizationDataType.ESTIMATION

        size = data.data[0] # Distance from the beacon to the receiver in meters
        beacon_x = data.data[1]
        beacon_y = data.data[2]

        self.windows[window_t].scatter(
            beacon_x, 
            beacon_y, 
            color="blue", 
            s=size, 
            label=f"Beacon"
        )
        plt.draw()
        plt.pause(interval=0.01)

    def _set_state_text(self, t: VisualizationDataType, data: VisualizationData):
        window_t = VisualizationDataType.ESTIMATION
        self.windows[window_t].text(1, 0, 'Current State: {}'.format(data.extra), 
            verticalalignment='bottom', 
            horizontalalignment='right', 
            transform=self.windows[window_t].transAxes,
            fontsize=12, 
            bbox=dict(facecolor='white', alpha=0.5))

        plt.draw()
        plt.pause(interval=0.01)


    def _interface_mp_to_thread(self, th_queue: PriorityQueue, thread_stop_event: ThreadingEvent):
        """Interface between multiprocessing and threading"""
        counter = itertools.count()

        stop_dequeue_process = False
        while not stop_dequeue_process:
            for viz_queue in self.interface.queues:
                if viz_queue.queue.empty():
                    continue

                try:
                    data = viz_queue.queue.get_nowait()
                    th_queue.put((data.timestamp, VisualizationInnerData(
                        counter=next(counter), type=viz_queue.type, data=data,
                    )))
                    
                except Exception as e:
                    logging.warning(f"Process queue error: {e}")

        time.sleep(5)
        thread_stop_event.set()

    def run(self):
        """Data visualization running on different process"""

        priority_queue = PriorityQueue()
        thread_stop_event = ThreadingEvent()
        mp_to_threading_process = threading.Thread(target=self._interface_mp_to_thread, args=(priority_queue, thread_stop_event, ), daemon=True)
        mp_to_threading_process.start()

        def _get_data():
            try:
                return priority_queue.get()
            except:
                return None
        
        time.sleep(1)
        last_refresh = time.time()
        count = 0
        while True:
            try:
                # Drain all available data from the queue
                while not priority_queue.empty():
                    visualization_data = _get_data()
                    if visualization_data is None:
                        continue

                    _, data = visualization_data

                    match (data.type):
                        case VisualizationDataType.IMAGE:
                            self._set_frame(data=data.data)
                        case VisualizationDataType.ACCELEROMETER |\
                                VisualizationDataType.GYROSCOPE |\
                                VisualizationDataType.VELOCITY |\
                                VisualizationDataType.ANGLE:
                            self._set_general_data(t=data.type, data=data.data)
                        case VisualizationDataType.GPS |\
                                VisualizationDataType.VO |\
                                VisualizationDataType.ESTIMATION |\
                                VisualizationDataType.GROUND_TRUTH |\
                                VisualizationDataType.LEICA:
                            self._set_estimation(t=data.type, data=data.data)
                        case VisualizationDataType.BEACON:
                            self._set_beacon_data(t=data.type, data=data.data)
                        case VisualizationDataType.STATE:
                            self._set_state_text(t=data.type, data=data.data)

                # Refresh the plot every REFRESH_RATE seconds
                now = time.time()
                if now - last_refresh >= REFRESH_RATE:
                    self._visualize()
                    last_refresh = now

                # If no data, wait a bit and count inactivity
                if priority_queue.empty():
                    time.sleep(0.05)
                    count += 1
                    if count > 200:  # 200 * 0.05s = 10s
                        print("No data received for 10 seconds, stop receiving data.")
                        break
                else:
                    count = 0

            except Exception as e:
                print(f"Exception: {e}")

        if not self.debugging:
            self._visualize()
            plt.draw()
            plt.pause(interval=1)

        time.sleep(1)
        print("Stopping visualization process...")
        mp_to_threading_process.join()

    def _drain_queues(self):
        """Drain all queues before stopping"""
        for viz_queue in self.interface.queues:
            while not viz_queue.queue.empty():
                try:
                    data = viz_queue.queue.get_nowait()
                    logging.debug(f"Draining: {data}")  # Process remaining data
                except viz_queue.queue.empty():
                    break  # Queue is empty

    def start(self):
        """Starts the visualization process."""
        self.process.start()

    def stop(self):
        """Waits for the visualization process to complete."""
        self.process.join()
        self.process.terminate()

    @staticmethod
    def show_innovation_history(innovations):
        """Show innovation history"""
        innovations = np.array(innovations)
        fig, axs = plt.subplots(3, 1, figsize=(10, 6))
        axs[0].set_title("Innovation History")
        for i, label in enumerate(["X", "Y", "Z"]):
            axs[i].set_title(f"innovation in {label}-axis")
            axs[i].plot(np.arange(innovations.shape[0]), innovations[:, i])
            axs[i].grid()

        fig.tight_layout()
        fig.savefig('./innovation_history.png')   # save the figure to file


if __name__ == "__main__":
    filter_config = FilterConfig(
        type="ekf",
        dimension=3,
        motion_model="kinematics",
        noise_type=False,
        params=None,
        innovation_masking=False,
        vo_velocity_only_update_when_failure=False,
    )

    config = VisualizationConfig(realtime=True,
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
                                     VisualizationDataType.IMAGE
                                 ])

    def _main():
        visualizer = RealtimeVisualizer(config=config, )
        zline = np.linspace(0, 15, 1949)
        xline = np.sin(zline)
        yline = np.cos(zline)

        xline2 = np.cos(zline)
        yline2 = np.sin(zline)

        queues = visualizer.interface.queues

        stereo_queue = [
            q.queue for q in queues if q.type == VisualizationDataType.IMAGE
        ].pop()
        estimation_queue = [
            q.queue for q in queues
            if q.type == VisualizationDataType.ESTIMATION
        ].pop()
        vo_queue = [
            q.queue for q in queues if q.type == VisualizationDataType.VO
        ].pop()

        visualizer.start()

        zline = zline[1:]
        xline = xline[1:]
        yline = yline[1:]
        xline2 = xline2[1:]
        yline2 = yline2[1:]

        i = 0
        while True:

            vo = np.array([xline[i], yline[i], zline[i]])
            estimate = np.array([xline2[i], yline2[i], zline[i]])

            if estimation_queue is not None:
                estimation_queue.put(
                    VisualizationData(timestamp=i, data=estimate))
            if vo_queue is not None:
                vo_queue.put(VisualizationData(timestamp=i, data=vo))
            image_path = os.path.join(
                f"../../../data/KITTI/seq_09/image_00/data/{i+1:010}.png")
            if stereo_queue is not None:
                stereo_queue.put(
                    VisualizationData(timestamp=i, data=image_path))

            time.sleep(0.1)
            if i < len(zline) - 1:
                i += 1
            else:
                time.sleep(10)
                break

        # visualizer.stop()

    _main()
