import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from ahrs.filters import Madgwick
class BaseFilter:
    
    mu_x = None
    mu_y = None
    mu_z = None

    errors = []
    
    madgwick = Madgwick()

    def get_diagonal_matrix(self, vector):
        i = np.eye(len(vector))
        return np.array([[val * num for num in i[ind]] for ind, val in enumerate(vector)])

    def compute_norm_w(self, w):
        return np.sqrt(np.sum(w**2))
    
    def get_quaternion(self, u, q_prev):
        return self.madgwick.updateIMU(q_prev.flatten(), gyr=u[3:], acc=u[:3])
    
    def get_euler_angle_from_quaternion(self, q):
        w, x, y, z = q
        roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x**2 + y**2))
        pitch = -np.pi/2 + 2*np.arctan2(np.sqrt(1 + 2*(w*y - x*z)), np.sqrt(1-2*(w*y - x*z)))
        yaw = np.arctan2(2*(w*z + x*y), 1-2*(y**2 + z**2))
        
        return np.array([roll, pitch, yaw])
    
    def get_rotation_matrix(self, q):
        q0, q1, q2, q3 = q[:, 0]
        # https://ahrs.readthedocs.io/en/latest/filters/ekf.html
        # https://www.iri.upc.edu/people/jsola/JoanSola/objectes/notes/kinematics.pdf
        return np.array([
            [q0**2 + q1**2 - q2**2 - q3**2, 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)],
            [2*(q1*q2 + q0*q3), q0**2 - q1**2 + q2**2 - q3**2, 2*(q2*q3 - q0*q1)],
            [2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), q0**2 - q1**2 - q2**2 + q3**2]
        ])

    def get_quaternion_update_matrix(self, w):
        wx, wy, wz = w[:, 0]
        # https://ahrs.readthedocs.io/en/latest/filters/ekf.html
        # https://www.iri.upc.edu/people/jsola/JoanSola/objectes/notes/kinematics.pdf
        return np.array([ # w, x, y, z
            [0, -wx, -wy, -wz],
            [wx, 0, wz, -wy],
            [wy, -wz, 0, wx],
            [wz, wy, -wx, 0]
        ])
        # return np.array([ # w, x, y, z
        #     [0, -wx, -wy, -wz],
        #     [wx, 0, -wz, wy],
        #     [wy, wz, 0, -wx],
        #     [wz, -wy, wx, 0]
        # ])

    def get_estimated_trajectory(self):
        return np.concatenate([
            np.array(self.mu_x).reshape(-1, 1), 
            np.array(self.mu_y).reshape(-1, 1), 
            np.array(self.mu_z).reshape(-1, 1)], axis=1)
    
    def visualize_trajectory(self, data, dimension=2, title=None, xlim=None, ylim=None, interval=None):
        if dimension == 2:
            fig, ax1 = plt.subplots(1, 1, figsize=(6, 6))
            xs, ys, _ = data.GPS_measurements_in_meter.T
            ax1.plot(xs, ys, lw=2, label='ground-truth trajectory', color='black')
            xs, ys, _ = data.VO_measurements.T
            ax1.plot(xs, ys, lw=2, label='VO trajectory', color='b')
            ax1.plot(
                self.mu_x, self.mu_y, lw=2, 
                label='estimated trajectory', color='r')
            if title is not None:
                ax1.title.set_text(title)
            if xlim is not None:
                ax1.set_xlim(xlim)
            if ylim is not None:
                ax1.set_ylim(ylim)
            ax1.set_xlabel('X [m]')
            ax1.set_ylabel('Y [m]')
            ax1.legend()
            ax1.grid()
        else:
            fig = plt.figure()
            ax1 = fig.add_subplot(111, projection='3d')
            ax1.set_title("ground-truth trajectory (GPS)")
            
            xs, ys, zs = data.GPS_measurements_in_meter.T
            ax1.plot(xs, ys, zs, label='ground-truth trajectory (GPS)', color='black')
            
            xs, ys, zs = data.VO_measurements.T
            ax1.plot(xs, ys, zs, label='Visual odometry trajectory', color='blue')
            
            ax1.plot(self.mu_x, self.mu_y, self.mu_z, label='Estimated trajectory', color='red')

            ax1.set_xlabel('X [m]', fontsize=14)
            ax1.set_ylabel('Y [m]', fontsize=14)
            ax1.set_zlabel('Z [m]', fontsize=14)

            fig.tight_layout()
            ax1.legend(loc='best', bbox_to_anchor=(1.1, 0., 0.2, 0.9))
        
        if interval is not None:
            plt.pause(interval=interval)

    def plot_error(self):
        plt.plot([i for i in range(len(self.errors))], self.errors, label='Error', color='r')

class TestFilter(BaseFilter):

    def __init__(self):
        pass

    def test(self):
        print(self.get_diagonal_matrix([0, 1, 2, 3, 4]))
        
    def test_get_quaternion(self, data):
        x, _, _, _, _, _ = data.get_initial_data(setup=SetupEnum.SETUP_1,filter_type=FilterEnum.EKF, noise_type=NoiseTypeEnum.CURRENT)
        Q = np.tile(x[6:].reshape(-1), (data.N, 1)) # Allocate for quaternions
        for t_idx in tqdm(range(1, data.N)):
            ax, ay, az = data.IMU_acc_with_noise[t_idx]
            wx, wy, wz = data.IMU_angular_velocity_with_noise[t_idx]
            u = np.array([
                ax,
                ay,
                az,
                wx,
                wy,
                wz
            ])
            Q[t_idx] = self.get_quaternion(u, q_prev=Q[t_idx -1])
            w = self.get_euler_angle_from_quaternion(Q[t_idx])
            print(w)
        


if __name__ == "__main__":
    import os
    import sys
    sys.path.append('../../src')
    from data_loader import DataLoader
    from configs import  SetupEnum, FilterEnum, NoiseTypeEnum
    from tqdm import tqdm

    root_path = "../../"
    file_export_path = os.path.join(root_path, "exports/_sequences/04")
    kitti_root_dir = os.path.join(root_path, "data")
    vo_root_dir = os.path.join(root_path, "vo_estimates")
    noise_vector_dir = os.path.join(root_path, "exports/_noise_optimizations/noise_vectors")
    kitti_date = '2011_09_30'
    kitti_drive = '0033'

    data = DataLoader(sequence_nr=kitti_drive, 
                    kitti_root_dir=kitti_root_dir, 
                    vo_root_dir=vo_root_dir,
                    noise_vector_dir=noise_vector_dir,
                    vo_dropout_ratio=0.0, 
                    gps_dropout_ratio=0.0)
    
    
    tf = TestFilter()
    tf.test_get_quaternion(data=data)