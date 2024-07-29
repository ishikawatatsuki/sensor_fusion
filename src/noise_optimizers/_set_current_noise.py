import os
import sys
import numpy as np
sys.path.append('../../src')
from data_loader import DataLoader


"""
Refs:
 - [1] OxTS-RT500-RT3000-Manual-240215.pdf
 - [2] https://github.com/yfcube/kitti-devkit-raw
 - [3] https://www.vectornav.com/resources/inertial-navigation-primer/specifications--and--error-budgets/specs-imuspecs
 - [4] https://www.insed.de/wp-content/uploads/2017/02/HG4930-Environmental-Performance-Manual_10-22-2017.pdf
 - [5] https://danceswithcode.net/engineeringnotes/quaternions/quaternions.html
"""

def compute_norm_w(w):
    return np.sqrt(np.sum(w**2))

def get_quaternion_update_matrix(w):
    wx, wy, wz = w
    return np.array([
        [0, wz, -wy, wx],
        [-wz, 0, wx, wy],
        [wy, -wx, 0, wz],
        [-wx, -wy, -wz, 0]
    ])

def main():
  """
  A high-accuracy GPS setup using dual-frequency (L1 and L2) signals, which can minimize ionospheric errors, represents the delay caused by the ionosphere depends on the electron density in the ionosphere layer. 

  The heading accuracy that can be achieved by the dual antenna system in the RTs in Table 5 is 0.2° per metre of separation in ideal, open sky conditions. The system can provide these accuracies in static and dynamic conditions -[1], page 18. The hading is referred to as yaw angle according to the Kitti raw data -[2].

  Digital anti-aliasing filters and coning/sculling motion compensation algorithms are run on the DSP. Calibration of the accelerometers and angular rate sensors also takes place in the DSP; this includes very high-precision alignment matrices that ensure that the direction of the acceleration and angular rate measurements is accurate to better than 0.01° -[1], page 100.

  The scale factor will not be perfectly calibrated and will have some error in the estimated ratio. This error is categorized as one of two equivalent values, either as a parts per million error (ppm), or as an error percentage. As shown in Figure 3.1, the scale factor error causes the output reported to be different from the true output. For example, if the z-axis of an accelerometer only measures gravity (9.81 m/s2), then the bias-corrected sensor output should be 9.81 m/s2. However, with a scale factor error of 0.1%, or 1,000 ppm, the output value from the sensor will instead be 9.82 m/s2.

  """
  gps_position_error = 0.01 # 0.01m L1/L2 -> 1cm -[1], page 15.
  forward_velocity_error_in_meter_per_second = 0.05 * 1000 / (60 * 60) # 0.05km/h -> m/s -[1], page 15.
  roll_pitch_angle_error = 0.03 * np.pi / 180 # 0.03° -> rad -[1], page 15.
  heading_angle_error = 0.2 * np.pi / 180 # 0.2° -> rad -[1], page 18.

  angular_velocity_error = 0.01 * np.pi / 180 # 0.01°/s -> rad/s -[1], page 100.

  # w = np.array([angular_velocity_error, angular_velocity_error, angular_velocity_error])
  # omega = get_quaternion_update_matrix(w)
  # norm_w = compute_norm_w(w)
  # A = np.cos(norm_w/2) * np.eye(4)
  # B = (1/norm_w)*np.sin(norm_w/2) * omega
  # q = np.array(A + B) @ np.array([1, 0, 0, 0]) # multiply with the identity quaternion, q = (1, 0, 0, 0). It represents no rotation. - [5].
  # q0_noise, q1_noise, q2_noise, q3_noise = q
  kitti_root_dir = '../../data'
  vo_root_dir = '../../vo_estimates'
  noise_vector_dir = '../../exports/_noise_optimizations/noise_vectors'
  kitti_date = '2011_09_30'
  kitti_drive = '0033'

  data = DataLoader(sequence_nr=kitti_drive, 
                    kitti_root_dir=kitti_root_dir, 
                    vo_root_dir=vo_root_dir,
                    noise_vector_dir=noise_vector_dir,
                    vo_dropout_ratio=0.0, 
                    gps_dropout_ratio=0.0)
  q0_noise, q1_noise, q2_noise, q3_noise = data.quaternion_process_noise

  linear_acceleration_error = 0.01 # arbitrary value

  vo_error = gps_position_error
  

  ekf_noise_vector_1_2 = np.array([
    linear_acceleration_error, 
    linear_acceleration_error, 
    linear_acceleration_error,
    heading_angle_error, 
    heading_angle_error, 
    heading_angle_error,
    vo_error, 
    vo_error,
    gps_position_error, 
    gps_position_error
  ])
  
  ekf_noise_vector_3 = np.array([
    forward_velocity_error_in_meter_per_second, 
    heading_angle_error,
    vo_error, 
    vo_error,
    gps_position_error, 
    gps_position_error
  ])

  other_noise_vector_1_2 = np.array([
    0.1, 
    0.1, 
    0.1,
    0.1, 
    0.1, 
    0.1,
    q0_noise, 
    q1_noise, 
    q2_noise, 
    q3_noise,
    vo_error, 
    vo_error,
    gps_position_error, 
    gps_position_error
  ])
    
  other_noise_vector_3 = np.array([
    forward_velocity_error_in_meter_per_second, 
    forward_velocity_error_in_meter_per_second,
    heading_angle_error,
    vo_error, 
    vo_error,
    gps_position_error,
    gps_position_error,
  ])

  # large value for measurement noise is set to the PF since PF uses the measurement error as a covariance matrix to estimate likelihood. The small value of the measurement error results in narrow multi-variate normal distribution leading filter divergence easily.
  pf_noise_vector_1_2 = np.array([
    0.1, 
    0.1, 
    0.1,
    0.1, 
    0.1, 
    0.1,
    q0_noise, 
    q1_noise, 
    q2_noise, 
    q3_noise,
    1.0, 
    1.0, 
    1.0, 
    1.0, 
  ])
    
  pf_noise_vector_3 = np.array([
    forward_velocity_error_in_meter_per_second, 
    forward_velocity_error_in_meter_per_second,
    heading_angle_error,
    1.0, 
    1.0,
    1.0, 
    1.0, 
  ])


  noise_vector_dir = '../../exports/_noise_optimizations/noise_vectors'

  ekf_noise_vector_1_2.dump(os.path.join(noise_vector_dir, "ekf/setup_1_current.npy"))
  ekf_noise_vector_1_2.dump(os.path.join(noise_vector_dir, "ekf/setup_2_current.npy"))
  ekf_noise_vector_3.dump(os.path.join(noise_vector_dir, "ekf/setup_3_current.npy"))

  other_noise_vector_1_2.dump(os.path.join(noise_vector_dir, "ukf/setup_1_current.npy"))
  other_noise_vector_1_2.dump(os.path.join(noise_vector_dir, "ukf/setup_2_current.npy"))
  other_noise_vector_3.dump(os.path.join(noise_vector_dir, "ukf/setup_3_current.npy"))

  pf_noise_vector_1_2.dump(os.path.join(noise_vector_dir, "pf/setup_1_current.npy"))
  pf_noise_vector_1_2.dump(os.path.join(noise_vector_dir, "pf/setup_2_current.npy"))
  pf_noise_vector_3.dump(os.path.join(noise_vector_dir, "pf/setup_3_current.npy"))

  other_noise_vector_1_2.dump(os.path.join(noise_vector_dir, "enkf/setup_1_current.npy"))
  other_noise_vector_1_2.dump(os.path.join(noise_vector_dir, "enkf/setup_2_current.npy"))
  other_noise_vector_3.dump(os.path.join(noise_vector_dir, "enkf/setup_3_current.npy"))

  other_noise_vector_1_2.dump(os.path.join(noise_vector_dir, "ckf/setup_1_current.npy"))
  other_noise_vector_1_2.dump(os.path.join(noise_vector_dir, "ckf/setup_2_current.npy"))
  other_noise_vector_3.dump(os.path.join(noise_vector_dir, "ckf/setup_3_current.npy"))

"""
Acceleration
- Bias stability 5 μg 1σ
  The bias stability is a measure of random variation in bias as computed over a specified sample time and averaging time interval.
  The unit "µg" means micro-g (where g is the acceleration due to gravity).
  1σ" denotes one standard deviation, implying that the measured bias stability is given as a statistical value with 68% confidence.
  
- Linearity 0.01%
  In ideal case, the relation between acceleration input and output voltage is linear. However, in reality, a deviation of an accelerometer response happens due to many reason, such as temperature. The linearity value, 0.01%, means that any non-linear error will be less than 0.01%.

- Scale factor 0.1% 
  Scale factor is a multiplier on a signal that is comprised of a ratio of the output to the input over the measurement range. This factor is dependent on temperature. For example,an acceleration sensor that has a scale factor of 0.1 % detecting an actual acceleration of 2 g (19.61 m/s2) may output a value of 19.63 m/s2. (19.61 * 0.001 + 19.61 = 19.629)

- Range 100 m/s2
  The maximum measurable acceleration range of the accelerometer.

Angular rate
- Bias 0.01°/s
  The bias is a constant offset of the output value from the input value.

- Scale factor 0.1%
  Same explanation as accelerometer.

- Range 100°/s
  The maximum measurable angular velocity range of the gyroscope. 

"""



if __name__ == "__main__":

  main()