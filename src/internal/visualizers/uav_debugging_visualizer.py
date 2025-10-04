import os
import sys
import logging
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import axes, figure
import matplotlib.patches as mpatches

from ..extended_common import VisualizationConfig

from ...common.config import FilterConfig

class UAV_DebuggingVisualizer:
    
    x_color = "red"
    y_color = "blue"
    z_color = "green"
    
    m1_color = "#4285F4"
    m2_color = "#34A853"
    m3_color = "#FBBC05"
    m4_color = "#EA4335"
    
    def __init__(
        self,
        config: VisualizationConfig,
    ):
        self.config = config
        
        self.frame = 0
        self.output_frame_path = os.path.join(self.config.output_filepath, "frames")
        
        self.figure = None
        self.trajectory_window = None
        self.tracking_cam_window = None
        self.acc_window = None
        self.gyro_window = None
        self.motor_output_window = None
        
        
        self.previous_data_point = {
            'gps': None,
            'imu': None,
            'thrust': None 
        }
        
        self.prev_timestamp = {
            'gps': None,
            'imu': None,
            'thrust': None 
        }
        
    def start(self):
        
        plt.ion()  # Enable interactive mode
        
        self.figure = plt.figure(figsize=(14, 9))
        gs = self.figure.add_gridspec(ncols=14, nrows=9)
        
        self.tracking_cam_window = self.figure.add_subplot(gs[:4, :5]) 
        self.trajectory_window = self.figure.add_subplot(gs[4:, :5], projection='3d') 
        gps = mpatches.Patch(color='black', label="GPS trajectory")
        self.trajectory_window.legend(handles=[gps])
        
        self.acc_window = self.figure.add_subplot(gs[:3, 5:]) 
        self.acc_window.set_title("Acceleration [m/s2]")
        self.acc_window.set(xlabel=None, ylabel="m/s2")
        self.acc_window.tick_params(left=True, bottom=False)
    
        acc_x = mpatches.Patch(color=self.x_color, label="Acc X [m/s^2]")
        acc_y = mpatches.Patch(color=self.y_color, label="Acc Y [m/s^2]")
        acc_z = mpatches.Patch(color=self.z_color, label="Acc Z [m/s^2]")
        self.acc_window.legend(handles=[acc_x, acc_y, acc_z], loc='upper left')
        
        
        self.gyro_window = self.figure.add_subplot(gs[3:6, 5:]) 
        self.gyro_window.set_title("Gyroscope [rad/s]")
        self.gyro_window.set(xlabel=None, ylabel="rad/s")
        self.gyro_window.tick_params(left=True, bottom=False)
        
        gyr_x = mpatches.Patch(color=self.x_color, label="Gyro X [rad/s]")
        gyr_y = mpatches.Patch(color=self.y_color, label="Gyro Y [rad/s]")
        gyr_z = mpatches.Patch(color=self.z_color, label="Gyro Z [rad/s]")
        self.gyro_window.legend(handles=[gyr_x, gyr_y, gyr_z], loc='upper left')
        
        self.motor_output_window = self.figure.add_subplot(gs[6:, 5:]) 
        self.motor_output_window.set_title("Motor output [rpm]")
        self.motor_output_window.set(xlabel="Timestamp (s)", ylabel="rpm")
        self.motor_output_window.tick_params(left=True, bottom=True)
        
        output1 = mpatches.Patch(color=self.m1_color, label="Motor 1 output")
        output2 = mpatches.Patch(color=self.m2_color, label="Motor 2 output")
        output3 = mpatches.Patch(color=self.m3_color, label="Motor 3 output")
        output4 = mpatches.Patch(color=self.m4_color, label="Motor 4 output")
        self.motor_output_window.legend(handles=[output1, output2, output3, output4], loc='upper left')
        
        plt.tight_layout()
        logging.debug("Creating VIO plot frame.")
        
    def stop(self):
        
        self.trajectory_window = None
        self.tracking_cam_window = None
        self.acc_window = None
        self.gyro_window = None
        self.motor_output_window = None
        plt.close(self.figure)
    
    def _save_current_frame(self):
        if self.config.save_frames and self.figure is not None:
            self.figure.savefig(os.path.join(self.output_frame_path, f"{str(self.frame)}.png"))
            self.frame += 1
        
    def set_frame(self, image_path):
        image_path = os.path.abspath(image_path)
        if os.path.exists(image_path):
            image = np.asarray(Image.open(image_path))
            self.tracking_cam_window.clear()
            self.tracking_cam_window.imshow(image)
            self.tracking_cam_window.axis("off")  # Hide axes for better viewing
            plt.draw()
            plt.pause(0.01)
            
        
        self._save_current_frame()
    
    def set_gps(self, value):
        if self.previous_data_point['gps'] is None:
            self.previous_data_point['gps'] = value
            return
        
        x = [self.previous_data_point['gps'][0], value[0]]
        y = [self.previous_data_point['gps'][1], value[1]]
        z = [self.previous_data_point['gps'][2], value[2]]
        self.trajectory_window.plot(x, y, z, label="GPS trajectory", color="black")
        plt.pause(0.01)
        
        self.previous_data_point['gps'] = value
        
        
    def set_thrust(self, timestamp: int, value: np.ndarray):
        if self.previous_data_point['thrust'] is None:
            self.previous_data_point['thrust'] = value
            self.prev_timestamp['thrust'] = timestamp
            return
        
        data = [[prev, current] for prev, current in zip(self.previous_data_point['thrust'], value)]
        
        ts = [self.prev_timestamp['thrust'], timestamp]
        
        self.motor_output_window.plot(ts, data[0], color=self.m1_color)
        self.motor_output_window.plot(ts, data[1], color=self.m2_color)
        self.motor_output_window.plot(ts, data[2], color=self.m3_color)
        self.motor_output_window.plot(ts, data[3], color=self.m4_color)
        
        
        self.previous_data_point['thrust'] = value
        self.prev_timestamp['thrust'] = timestamp
        
    def set_imu(self, timestamp: int, value: np.ndarray):
        if self.previous_data_point['imu'] is None:
            self.previous_data_point['imu'] = value
            self.prev_timestamp['imu'] = timestamp
            return
        
        data = [[prev, current] for prev, current in zip(self.previous_data_point['imu'], value)]
        ts = [self.prev_timestamp['imu'], timestamp]
        
        acc, gyr = data[:3], data[3:]
        self.acc_window.plot(ts, acc[0], color=self.x_color)
        self.acc_window.plot(ts, acc[1], color=self.y_color)
        self.acc_window.plot(ts, acc[2], color=self.z_color)
        
        self.gyro_window.plot(ts, gyr[0], color=self.x_color)
        self.gyro_window.plot(ts, gyr[1], color=self.y_color)
        self.gyro_window.plot(ts, gyr[2], color=self.z_color)
        
        self.previous_data_point['imu'] = value
        self.prev_timestamp['imu'] = timestamp
        
if __name__ == "__main__":
    import time
    from tqdm import tqdm
    from PIL import Image

    filter_config = FilterConfig(
        type="ekf",
        dimension=3,
        motion_model="kinematics",
        noise_type=False,
        params=None,
        innovation_masking=False,
        vo_velocity_only_update_when_failure=False,
    )
    
    config = VisualizationConfig(
        realtime=True,
        output_filepath="./",
        save_trajectory=True,
        show_end_result=True,
        save_frames=False,
        show_vo_trajectory=False,
        show_vio_frame=False,
        show_particles=False,
        set_lim_in_plot=False,
        show_innovation_history=False,
        show_angle_estimation=False,
        limits=[]
    )
    
    visualizer = UAV_DebuggingVisualizer(
        config=config
    )
    
    visualizer.start()
    visualizer.set_frame(image_path="../../data/UAV/log0001/run/mpa/qvio_overlay/00000.png")
    
    num_points = 300  # Number of points
    radius = 1         # Radius of the helix
    height = 10        # Total height of the helix
    turns = 5          # Number of turns
    # Generate helix data
    theta = np.linspace(0, 2 * np.pi * turns, num_points)  # Angle along the helix
    heights = np.linspace(0, height, num_points)                # Height along the helix
    
    for i, angle in enumerate(theta):
        x = radius * np.cos(angle)                            # X-coordinates
        y = radius * np.sin(angle)                            # Y-coordinates
        z = heights[i]
        visualizer.set_gps([x, y, z])

        point = np.random.normal(0, 0.1, 6)
        visualizer.set_imu(timestamp=i, value=point)
        thrust = np.random.normal(0, 0.3, 4)
        visualizer.set_thrust(timestamp=i, value=thrust)
        
    
    visualizer.stop()
    
    