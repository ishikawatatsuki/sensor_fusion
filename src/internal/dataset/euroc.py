import os
import sys
import numpy as np
from collections import namedtuple

if __name__ == "__main__":
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
    from src.common import (
        GyroSpecification,
        AccelSpecification,
        FilterConfig,
        State
    )
else:
    from ...common import (
        GyroSpecification,
        AccelSpecification,
        FilterConfig,
        State
    )

DIVIDER = 1_000_000_000  # ns to s
class EuRoC_IMUDataReader:
    """
        Read IMU data from EuRoC dataset.
    """
    def __init__(
            self, 
            root_path: str,
            starttime=-float('inf'),
            window_size=None
        ):

        self.root_path = os.path.join(root_path, "imu0/data.csv")
        self.starttime = starttime
        self.field = namedtuple('data', 
            ['timestamp', 'a', 'w'])
        
        self.window_size = window_size
        self.buffer = []

    def parse(self, line):
        """
        line: 
            #timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]
        """
        line = [float(_) for _ in line.strip().split(',')]

        timestamp = int(line[0]) / DIVIDER
        w = np.array(line[1:4])
        a = np.array(line[4:7])
        return self.field(timestamp, a, w)
    
    def rolling_average(self, data):
        
        d = np.hstack([data.w, data.a])
        self.buffer.append(d)
        if len(self.buffer) > self.window_size:
            mean = np.mean(self.buffer, axis=0)
            self.buffer = self.buffer[-self.window_size:]
            return self.field(timestamp=data.timestamp, w=mean[:3], a=mean[3:])
            
        return self.field(timestamp=data.timestamp, w=data.w, a=data.a)
    
    def __iter__(self):
        with open(self.root_path, 'r') as f:
            next(f)
            for line in f:
                data = self.parse(line)
                if data.timestamp < self.starttime:
                    continue
                if self.window_size is not None:
                    data = self.rolling_average(data)
                yield data

    def start_time(self):
        with open(self.root_path, 'r') as f:
            next(f)
            for line in f:
                return self.parse(line).timestamp

    def set_starttime(self, starttime):
        self.starttime = starttime
    
class EuRoC_StereoFrameReader:
    def __init__(self, root_path: str, starttime=-float('inf')):
        self.starttime = starttime
        self.root_path = root_path
        self.left_image_path = os.path.join(root_path, "cam0")
        self.right_image_path = os.path.join(root_path, "cam1")
        self.field = namedtuple('data', ['timestamp', 'left_frame_id', 'right_frame_id'])
        
    def parse(self, left_line: str, right_line: str):
        """
        line: 
            #timestamp [ns],filename
        """
        l_line = [_ for _ in left_line.strip().split(',')]
        r_line = [_ for _ in right_line.strip().split(',')]

        timestamp = int(l_line[0]) / DIVIDER
        left_frame_id = os.path.join(self.left_image_path, 'data', l_line[1])
        right_frame_id = os.path.join(self.right_image_path, 'data', r_line[1])
        if not os.path.exists(left_frame_id) or\
            not os.path.exists(right_frame_id):
            return None

        return self.field(timestamp, left_frame_id, right_frame_id)

    def __iter__(self):
        with open(os.path.join(self.left_image_path, "data.csv"), 'r') as left_f, open(os.path.join(self.right_image_path, "data.csv"), 'r') as right_f:
            next(left_f)
            next(right_f)
            for left_line, right_line in zip(left_f, right_f):
                data = self.parse(left_line, right_line)
                if data is None or data.timestamp < self.starttime:
                    continue
                yield data

    def start_time(self):
        with open(self.gyro_path, 'r') as f:
            next(f)
            for line in f:
                return self.parse(line).timestamp

    def set_starttime(self, starttime):
        self.starttime = starttime
    
class EuRoC_LeiCaDataReader:
    
    def __init__(self, root_path: str, starttime=-float('inf')):
        self.root_path = os.path.join(root_path, "leica0/data.csv")
        self.starttime = starttime
        self.field = namedtuple('data', 
            ['timestamp', 'p_x', 'p_y', 'p_z'])

    def parse(self, line):
        """
        line: 
            #timestamp [ns],p_RS_R_x [m],p_RS_R_y [m],p_RS_R_z [m]
        """
        line = [float(_) for _ in line.strip().split(',')]
        
        timestamp = int(line[0]) / DIVIDER
        p_x = line[1]
        p_y = line[2]
        p_z = line[3]
        
        return self.field(timestamp, p_x, p_y, p_z)

    def __iter__(self):
        with open(self.root_path, 'r') as f:
            next(f)
            for line in f:
                data = self.parse(line)
                if data.timestamp < self.starttime:
                    continue
                yield data

    def start_time(self):
        with open(self.root_path, 'r') as f:
            next(f)
            for line in f:
                return self.parse(line).timestamp

    def set_starttime(self, starttime):
        self.starttime = starttime

class EuRoC_GroundTruthDataReader:
    def __init__(
            self, 
            root_path: str,
            starttime=-float('inf'),
            window_size=None,
        ):
        self.root_path = os.path.join(root_path, "state_groundtruth_estimate0/data.csv")
        self.starttime = starttime
        self.field = namedtuple('data', 
            ['timestamp', 'p', 'v', 'q', 'b_w', 'b_a'])
        
        self.window_size = window_size
        self.buffer = []
    
    def parse(self, line: str):
        """
        #timestamp, p_RS_R_x [m], p_RS_R_y [m], p_RS_R_z [m], q_RS_w [], q_RS_x [], q_RS_y [], q_RS_z [], v_RS_R_x [m s^-1], v_RS_R_y [m s^-1], v_RS_R_z [m s^-1], b_w_RS_S_x [rad s^-1], b_w_RS_S_y [rad s^-1], b_w_RS_S_z [rad s^-1], b_a_RS_S_x [m s^-2], b_a_RS_S_y [m s^-2], b_a_RS_S_z [m s^-2]
        """
        line_list = [float(_) for _ in line.strip().split(',')]

        timestamp = int(line_list[0]) / DIVIDER
        p = np.array(line_list[1:4])
        q = np.array(line_list[4:8])
        v = np.array(line_list[8:11])
        b_w = np.array(line_list[11:14])
        b_a = np.array(line_list[14:17])
        
        return self.field(timestamp, p, v, q, b_w, b_a)
    
    def rolling_average(self, data):
        
        d = np.hstack([data.w, data.a])
        self.buffer.append(d)
        if len(self.buffer) > self.window_size:
            mean = np.mean(self.buffer, axis=0)
            self.buffer = self.buffer[-self.window_size:]
            return self.field(timestamp=data.timestamp, w=mean[:3], a=mean[3:])
            
        return self.field(timestamp=data.timestamp, w=data.w, a=data.a)
    
    def __iter__(self):
        with open(self.root_path, 'r') as f:
            next(f)
            for line in f:
                data = self.parse(line)
                if data.timestamp < self.starttime:
                    continue
                if self.window_size is not None:
                    data = self.rolling_average(data)
                yield data

    def start_time(self):
        with open(self.root_path, 'r') as f:
            next(f)
            for line in f:
                return self.parse(line).timestamp

    def set_starttime(self, starttime):
        self.starttime = starttime

    def get_initial_state(self, filter_config: FilterConfig) -> State:
        with open(self.root_path, 'r') as f:
            next(f)
            line = f.readline()
            data = self.parse(line)
            return State(p=data.p, v=np.zeros((3, 1)), q=data.q, b_w=data.b_w, b_a=data.b_a)


if __name__ == "__main__":
    import yaml
    imu_configs = None
    base_dir = "../../../data/EuRoC"
    data_root_path = os.path.join(base_dir, "mav_01")
    imu_config_path = os.path.join(base_dir, "configs/imu_config.yaml")
    with open(imu_config_path, "r") as f:
        imu_configs = yaml.safe_load(f)
        f.close()
        
    imu_config = namedtuple('IMU_Configs', ["adis_16448"])(**imu_configs)
    imu_path = os.path.join(data_root_path, "imu0/data.csv")
    leica_path = os.path.join(data_root_path, "leica0/data.csv")
    image_root_path = data_root_path


    imu = EuRoC_IMUDataReader(
        root_path=data_root_path,
    )
    stereo = EuRoC_StereoFrameReader(
        root_path=data_root_path,
    )
    leica = EuRoC_LeiCaDataReader(
        root_path=leica_path,
    )
    gt = EuRoC_GroundTruthDataReader(
        root_path=data_root_path
    )

    initial_state = gt.get_initial_state(filter_config=FilterConfig(
        type="ekf",
        dimension=2,
        motion_model="velocity",
        noise_type="default",
        innovation_masking=False,
        params=None,
    ))

    def _main():
        i = 0
        # dataset = iter(imu)
        # dataset = iter(stereo)
        # dataset = iter(leica)
        dataset = iter(gt)
        while True:
            try:
                # IMU
                # data = next(dataset)
                # print(data.p, data.v, data.q)

                # Stereo
                # data = next(dataset)

                # Leica
                data = next(dataset)

                # Ground Truth
                # data = next(dataset)

                print(data)

            except StopIteration:
                break
            if i > 10:
                break
            i += 1
    
    _main()