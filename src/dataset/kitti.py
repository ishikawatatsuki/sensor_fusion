import os
import sys
import pykitti
import numpy as np
import pandas as pd
from datetime import datetime
from collections import namedtuple

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from utils import get_oxts_gyroscope_noise, get_oxts_acceleration_noise
from interfaces import State, Pose

SENSOR_DELAY = 0.000001

class OXTS_IMUDataReader:
    def __init__(self, root_path, date, drive, starttime=-float('inf')):
        self.kitti_dataset = pykitti.raw(root_path, date, drive)
        self.starttime = starttime
        self.field = namedtuple('data', ['timestamp', 'a', 'w'])
        
        self.gyro_noise  = get_oxts_gyroscope_noise()
        self.acc_noise = get_oxts_acceleration_noise()

    def __iter__(self):
      for i, oxts in enumerate(self.kitti_dataset.oxts):
          packet = oxts.packet
          timestamp = datetime.timestamp(self.kitti_dataset.timestamps[i])
          if timestamp < self.starttime:
              continue
          
          a = np.array([packet.ax, packet.ay, packet.az]) - np.random.normal(0, self.acc_noise, 3)
          w = np.array([packet.wx, packet.wy, packet.wz]) - np.random.normal(0, self.gyro_noise, 3)
          yield self.field(timestamp, a, w)

    def start_time(self):
      return self.kitti_dataset.timestamps[0]

    def set_starttime(self, starttime):
        self.starttime = starttime
        
    @staticmethod
    def get_noise_vector():
        acc_noise = get_oxts_acceleration_noise()
        gyro_noise  = get_oxts_gyroscope_noise()
        
        return acc_noise, gyro_noise
      
class OXTS_GPSDataReader:
    def __init__(self, root_path, date, drive, starttime=-float('inf')):
        self.kitti_dataset = pykitti.raw(root_path, date, drive)
        self.starttime = starttime
        self.field = namedtuple('data', ['timestamp', 'lon', 'lat', 'alt'])

    def __iter__(self):
      for i, oxts in enumerate(self.kitti_dataset.oxts):
          packet = oxts.packet
          timestamp = datetime.timestamp(self.kitti_dataset.timestamps[i])
          if timestamp < self.starttime:
              continue
          yield self.field(timestamp, packet.lon, packet.lat, packet.alt)

    def start_time(self):
      return self.kitti_dataset.timestamps[0]

    def set_starttime(self, starttime):
        self.starttime = starttime

class KITTI_GroundTruthDataReader:
    filename_map = {
      "0027": "00.txt",
      "0042": "01.txt",
      "0034": "02.txt",
      "0067": "03.txt",
      "0016": "04.txt",
      "0018": "05.txt",
      "0020": "06.txt",
      "0027": "07.txt",
      "0028": "08.txt",
      "0033": "09.txt",
      "0034": "10.txt",
      "0098": "curve_cropped_09.txt",
      "0099": "straight_cropped_09.txt",
      "0100": "straight_cropped_09_2.txt",
    }
  
    def __init__(self, root_path, date, drive, starttime=-float('inf')):
      gt_filename = self.filename_map.get(drive)
      assert gt_filename is not None, "Please specify proper drive dataset"

      self.kitti_dataset = pykitti.raw(root_path, date, drive)
      self.timestamps = self.kitti_dataset.timestamps
      self.path = os.path.join(root_path, "ground_truth", gt_filename)
      self.starttime = starttime
      
      self.initial_pose = None
      self.field = namedtuple('data', ['timestamp', 'R', 't'])
      
    def parse(self, i, line):
      """
      line:
        1.000000e+00 9.043680e-12 2.326809e-11 5.551115e-17 9.043683e-12 1.000000e+00 2.392370e-10 3.330669e-16 2.326810e-11 2.392370e-10 9.999999e-01 -4.440892e-16
      """
      
      if len(self.timestamps) <= i:
        return None
      line = np.array([float(_) for _ in line.strip().split(' ')])
      mat = line.reshape(3, 4)
      R = mat[:3, :3]
      t = mat[:, 3]
      
      timestamp = datetime.timestamp(self.timestamps[i])

      return self.field(timestamp, R=R, t=t)

    def __iter__(self):
      with open(self.path, 'r') as f:
        for row in enumerate(f):
          
          data = self.parse(*row)
          if data is None or data.timestamp < self.starttime:
            continue
          
          yield data
    
    def start_time(self):
      return self.kitti_dataset.timestamps[0]

    def set_starttime(self, starttime):
      self.starttime = starttime
      
    def get_initial_state(self) -> State:
      with open(self.path, 'r') as f:
        line = f.readline()
        line = np.array([float(_) for _ in line.strip().split(' ')])
        mat = line.reshape(3, 4)
        R = mat[:, :3]
        t = mat[:, 3]
        gt_pose = Pose(R=R, t=t)
        self.initial_pose = gt_pose
        f.close()
        
        _oxts = self.kitti_dataset.oxts[0]
        packet = _oxts.packet
        
        p = _oxts.T_w_imu[:3, 3].reshape(-1, 1)
        v = np.array([
          packet.vf,
          packet.vl,
          packet.vu
        ]).reshape(-1, 1)
        # angle = np.array([packet.roll, packet.pitch, packet.yaw])
        # print(angle)
        # q = State.get_quaternion_from_euler_angle(w=angle)
        # q /= np.linalg.norm(q)
        
        p = np.zeros((3, 1))
        # v = np.zeros((3, 1))
        q = np.array([1., 0., 0., 0]).reshape(-1, 1)
      return State(p=p, v=v, q=q)

class OXTS_INSDataReader:
    def __init__(self, root_path, date, drive, starttime=-float('inf')):
        self.kitti_dataset = pykitti.raw(root_path, date, drive)
        self.starttime = starttime
        self.field = namedtuple('data', [
            'timestamp', 
            'ax',
            'ay',
            'az',
            'wx',
            'wy',
            'wz',
            'roll', 
            'pitch', 
            'yaw', 
            'vf', 
            'vl', 
            'vu'
          ])

    def __iter__(self):
      for i, oxts in enumerate(self.kitti_dataset.oxts):
          packet = oxts.packet
          timestamp = datetime.timestamp(self.kitti_dataset.timestamps[i])
          if timestamp < self.starttime:
              continue
          yield self.field(
              timestamp, 
              packet.ax,
              packet.ay,
              packet.az,
              packet.wx,
              packet.wy,
              packet.wz,
              packet.roll,
              packet.pitch,
              packet.yaw,
              packet.vf,
              packet.vl,
              packet.vu
            )

    def start_time(self):
      return self.kitti_dataset.timestamps[0]

    def set_starttime(self, starttime):
        self.starttime = starttime

class KITTI_StereoFrameReader:
  def __init__(self, root_path, date, drive, iter_offset=0, starttime=-float('inf')):
      self.kitti_dataset = pykitti.raw(root_path, date, drive)
      self.starttime = starttime
      self.field = namedtuple('data', ['timestamp', 'left_frame_id', 'right_frame_id'])
      
      self.kitti_image_data_path = os.path.join(date, f"{date}_drive_{drive}_sync")
      self.image_root_path = os.path.join(root_path, self.kitti_image_data_path)
      self.iter_offset = iter_offset
  
  def __iter__(self):
      for i, _ in enumerate(self.kitti_dataset.oxts):
          id = i + self.iter_offset
          timestamp = datetime.timestamp(self.kitti_dataset.timestamps[i])
          if timestamp < self.starttime:
              continue
          
          left_frame_id = f"image_00/data/{id:010}.png"  
          right_frame_id = f"image_01/data/{id:010}.png"
          if not os.path.exists(os.path.join(self.image_root_path, left_frame_id)) or\
              not os.path.exists(os.path.join(self.image_root_path, right_frame_id)):
            continue
          
          yield self.field(
            timestamp, 
            os.path.join(self.kitti_image_data_path, left_frame_id), 
            os.path.join(self.kitti_image_data_path, right_frame_id)
          )

  def start_time(self):
    return self.kitti_dataset.timestamps[0]

  def set_starttime(self, starttime):
    self.starttime = starttime
    
class KITTI_CustomVisualOdometry:
  def __init__(
      self, 
      root_path, 
      date, 
      drive, 
      window_size=None,
      starttime=-float('inf')
    ):
    self.kitti_dataset = pykitti.raw(root_path, date, drive)
    self.df = pd.read_csv(
      os.path.join(root_path, "vo_estimates", drive, "estimations.csv"),
      names=["x", "y", "z"]
    )
    self.starttime = starttime
    self.field = namedtuple('data', 
        ['timestamp', 'x', 'y', 'z', 'vx', 'vy', 'vz'])
    
    self.last_position = None
    self.last_timestamp = None
    self.window_size = window_size
    self.buffer = []
    
  def rolling_average(self, velocity):
      self.buffer.append(velocity)
      if len(self.buffer) > self.window_size:
          mean = np.mean(self.buffer, axis=0)
          self.buffer = self.buffer[-self.window_size:]
          return mean
          
      return velocity
    
  def __iter__(self):
    for i, timestamp in enumerate(self.kitti_dataset.timestamps[:self.df.shape[0]]):
      timestamp = datetime.timestamp(timestamp)
      if timestamp < self.starttime or self.last_timestamp is None:
        self.last_timestamp = timestamp
        self.last_position = self.df.iloc[i]
        continue
      position = self.df.iloc[i]
      dt = timestamp - self.last_timestamp
      delta_p = position - self.last_position
      velocity = delta_p.values / dt
      
      self.last_timestamp = timestamp
      self.last_position = position
      
      if self.window_size is not None:
        velocity = self.rolling_average(velocity=velocity)
      
      yield self.field(
          timestamp, 
          x=position.x, 
          y=position.y, 
          z=position.z,
          vx=velocity[0],
          vy=velocity[1],
          vz=velocity[2]
        )

  def start_time(self):
    return self.kitti_dataset.timestamps[0]

  def set_starttime(self, starttime):
    self.starttime = starttime


class KITTI_ColorImageDataReader:
  def __init__(self, root_path, date, drive, starttime=-float('inf')):
      self.kitti_dataset = pykitti.raw(root_path, date, drive)
      self.starttime = starttime
      self.field = namedtuple('data', ['timestamp', 'image_path'])
      
      self.image_root_path = os.path.join(root_path, date, f"{date}_drive_{drive}_sync")
  
  def __iter__(self):
      for i, _ in enumerate(self.kitti_dataset.oxts):
          timestamp = datetime.timestamp(self.kitti_dataset.timestamps[i])
          if timestamp < self.starttime:
              continue
          
          image_path = os.path.join(self.image_root_path, f"image_02/data/{i:010}.png")
          if not os.path.exists(image_path):
            continue
          
          yield self.field(timestamp, image_path=image_path)

  def start_time(self):
    return self.kitti_dataset.timestamps[0]

  def set_starttime(self, starttime):
    self.starttime = starttime

class KITTI_UpwardLeftwardVelocityDataReader:
  def __init__(self, root_path, date, drive, starttime=-float('inf')):
      self.kitti_dataset = pykitti.raw(root_path, date, drive)
      self.starttime = starttime
      self.field = namedtuple('data', ['timestamp', 'vl', 'vu'])
      
  
  def __iter__(self):
      for i, _ in enumerate(self.kitti_dataset.oxts):
          timestamp = datetime.timestamp(self.kitti_dataset.timestamps[i])
          if timestamp < self.starttime:
              continue
          
          yield self.field(timestamp, vl=0., vu=0.)

  def start_time(self):
    return self.kitti_dataset.timestamps[0]

  def set_starttime(self, starttime):
    self.starttime = starttime
    

if __name__ == "__main__":

  import matplotlib.pyplot as plt
  from utils import KITTI_GeometricTransformer
  from custom_types import SensorType
  from config import Config, DatasetConfig
  import matplotlib.patches as mpatches
  import matplotlib.lines as mlines
  
  drive = "0033"
  
  imu = OXTS_IMUDataReader(
    root_path="../../data/KITTI",
    date="2011_09_30",
    drive=drive
  )
  
  gps = OXTS_GPSDataReader(
    root_path="../../data/KITTI",
    date="2011_09_30",
    drive=drive
  )
  
  ins = OXTS_INSDataReader(
    root_path="../../data/KITTI",
    date="2011_09_30",
    drive=drive
  )
  
  stereo = KITTI_StereoFrameReader(
    root_path="../../data/KITTI",
    date="2011_09_30",
    drive=drive
  )
  
  vo = KITTI_CustomVisualOdometry(
    root_path="../../data/KITTI",
    date="2011_09_30",
    drive=drive,
    window_size=1
  )
  
  gt = KITTI_GroundTruthDataReader(
    root_path="../../data/KITTI",
    date="2011_09_30",
    drive=drive
  )
  
  # i = 0
  # vo_data = iter(vo)
  # ins_data = iter(ins)
  
  # vf_vo = []
  # vo_ts = []
  # vf_ins = []
  # ins_ts = []
  # while True:
  #     try:
  #         _vo_data = next(vo_data)
  #         _ins_data = next(ins_data)
  #         if _vo_data is not None and _ins_data is not None:
  #           vo_ts.append(_vo_data.timestamp)
  #           ins_ts.append(_ins_data.timestamp)
  #           vf_vo.append(np.linalg.norm(np.array([_vo_data.vx, _vo_data.vy, _vo_data.vz])))
  #           vf_ins.append(_ins_data.vf)
  #     except StopIteration:
  #         break
  #     i += 1
      
  # fig, ax = plt.subplots(1, 1, figsize=(8, 8))
  # ax.set_title("Forward velocity comparison")
  # ax.plot(vo_ts, vf_vo, label="Forward velocity (VO)")
  # ax.plot(ins_ts, vf_ins, label="Forward velocity (INS)")
  # ax.set_xlabel('X [m]')
  # ax.set_ylabel('Y [m]')
  # plt.legend()
  # plt.show()
  
  
  # MAX_CONSECUTIVE_DROPOUT_RATIO = 0.15
  
  # dropout_ratio = 0.7
  # dataset = list(iter(stereo))
  # iter_length = len(dataset)
  # np.random.seed(777)
  
  # dp_list = [MAX_CONSECUTIVE_DROPOUT_RATIO for i in range(int(dropout_ratio // MAX_CONSECUTIVE_DROPOUT_RATIO))]
  
  # if dropout_ratio % MAX_CONSECUTIVE_DROPOUT_RATIO != 0.0 and 0.001 < dropout_ratio % MAX_CONSECUTIVE_DROPOUT_RATIO < MAX_CONSECUTIVE_DROPOUT_RATIO:
  #   dp_list.append(round(dropout_ratio % MAX_CONSECUTIVE_DROPOUT_RATIO, 3))
    
  # print(dp_list)
  # dp_list.reverse()
  # dropout_length = int(iter_length * dp_list.pop())
  # start_dropping_at = np.random.randint(int(iter_length * 0.2))
  # end_dropping_at = start_dropping_at + dropout_length
  
  # print(start_dropping_at, end_dropping_at)
    
  # i = 0
  # while i < len(dataset):
  #   try:
  #       data = dataset[i]
  #       i += 1
  #   except StopIteration:
  #       break
    
  #   data_dropped = start_dropping_at < i <= end_dropping_at
  #   if not data_dropped:
  #     # print(i)
  #     ...
    
  #   if end_dropping_at < i and len(dp_list):
  #     dropout_length = int(iter_length * dp_list.pop())
  #     rest_iter = iter_length - end_dropping_at
  #     start_dropping_at = end_dropping_at + np.random.randint(int(rest_iter * 0.2))
  #     end_dropping_at = start_dropping_at + dropout_length

  dataset_config = DatasetConfig(
    type="kitti",
    mode="stream",
    root_path="../../data/KITTI",
    variant=drive,
    sensors=[SensorType.OXTS_IMU]
  )

  geo_transformer = KITTI_GeometricTransformer(
    dataset_config=dataset_config,
  )



  def _show_vo_drop():
    # 500 - 600 - - 800 - 899
    # 1100 - 1200 - - 1400 - 1499
    
    gt_data = np.array([(geo_transformer.T_from_cam_to_imu @ np.hstack([pose.t, np.array([1])]))[:3] for pose in list(iter(gt))])

    initial_point = gt_data[0, :]
    
    for i, (pre, curr) in enumerate(zip(gt_data[:-1, :], gt_data[1:, :])):
      if 600 <= i < 800:
        # curved
        p_x, p_y, _ = pre
        c_x, c_y, _ = curr
        if i == 600:
          plt.scatter(c_x, c_y, s=50, color="red")
          
        color = "blue"
        plt.plot([p_x, c_x], [p_y, c_y], color=color, lw=2)
        
      elif 1200 <= i < 1400:
        p_x, p_y, _ = pre
        c_x, c_y, _ = curr
        if i == 1200:
          plt.scatter(c_x, c_y, s=50, color="red")
          
        color = "blue"
        plt.plot([p_x, c_x], [p_y, c_y], color=color, lw=2)
        
      else:
        p_x, p_y, _ = pre
        c_x, c_y, _ = curr
        plt.plot([p_x, c_x], [p_y, c_y], color="gray", lw=2)
        
      
    plt.scatter(initial_point[0], initial_point[1], s=50, color="black", marker=">")
    plt.xlabel('X [m]', fontsize=16)
    plt.ylabel('Y [m]', fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=12)
    # entire_trajectory = mpatches.Patch(color='gray', label='Ground truth trajectory')
    # # vo_loss_start = mpatches.Patch(color='red', hatch='+', label='VO data loss start point')
    # vo_loss_start = mlines.Line2D([], [], color='red', marker='o',
    #                       markersize=14, label='VO data loss start point')
    # vo_loss_trajectory = mpatches.Patch(color='blue', label='VO data loss')
    # plt.legend(handles=[entire_trajectory, vo_loss_start, vo_loss_trajectory])
      
    plt.title("A trajectory of VO data loss experiment", size=18)
    plt.grid()
    plt.show()
    # print(gt_data[:-1, :].shape)
    # print(gt_data[1:, :].shape)
    
    
  
  def _show_real_life_scenario():
    # 500 - 600 - - 800 - 899
    # 1100 - 1200 - - 1400 - 1499
    
    # vo_data = np.array([pose.t for pose in list(iter(vo))])
    gt_data = np.array([(geo_transformer.T_from_cam_to_imu @ np.hstack([pose.t, np.array([1])]))[:3] for pose in list(iter(gt))])
    ins_data = list(iter(ins))
    
    vo_drops = [(350, 400), (650, 750), (1200, 1300)]
    gps_drops = [(200, 300), (700, 900), (1200, 1400)]
    points = [320, 630, 900]
    linestyle = (0, (3, 10, 1, 10, 1, 10))
    dashes = (1, 100)
    
    initial_point = gt_data[0, :]
    
    for i, (pre, curr) in enumerate(zip(gt_data[:-1, :], gt_data[1:, :])):
      p_x, p_y, _ = pre
      c_x, c_y, _ = curr
      is_gps_drop = True in (drops[0] <= i < drops[1] for drops in gps_drops) 
      is_vo_drop = True in (drops[0] <= i < drops[1] for drops in vo_drops) 
      is_all_drop = is_gps_drop and is_vo_drop
      
      if is_all_drop:
        plt.plot([p_x, c_x], [p_y, c_y], color="purple", lw=3)
      elif is_vo_drop:
        plt.plot([p_x, c_x], [p_y, c_y], color="red", lw=3)
      elif is_gps_drop:
        plt.plot([p_x, c_x], [p_y, c_y], color="blue", lw=3)
      else:    
        plt.plot([p_x, c_x], [p_y, c_y], color="gray", lw=2)

      # if i % 100 == 0:
      #   plt.scatter(c_x, c_y, s=50, color="red", marker=">")
        
    plt.scatter(initial_point[0], initial_point[1], s=50, color="black", marker=">")
    plt.xlabel('X [m]', fontsize=16)
    plt.ylabel('Y [m]', fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=12)
    
    gps_vo_loss = mpatches.Patch(color='purple', label='GPS and VO data loss')
    vo_loss = mpatches.Patch(color='red', label='VO data loss')
    gps_loss = mpatches.Patch(color='blue', label='GPS data loss')
    all_available = mpatches.Patch(color='gray', label='ALL data is available')
    plt.legend(handles=[gps_vo_loss, vo_loss, gps_loss, all_available])
    
    plt.title("A trajectory of a real life scenario.", size=18)
    plt.grid()
    plt.show()
    
    
    
  _show_vo_drop()
  # _show_real_life_scenario()