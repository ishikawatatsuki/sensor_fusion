import pywt
import logging
import numpy as np
from enum import Enum
from dataclasses import dataclass
from scipy.signal import butter, butter, lfilter, savgol_filter
from ahrs.filters import EKF

from ..common import (
    SignalType,
    SensorType,
    SensorDataField,
    HardwareConfig,
    FilterConfig
)

class SignalProcessor:
    
    def __init__(
          self, 
          filter_config: FilterConfig,
          hardware_config: HardwareConfig
      ):
        
        self.filter_config = filter_config
        self.hardware_config = hardware_config
        # TODO: Make these parameters configurable by yaml files.
        self.freq = hardware_config.imu_config.frequency
        self.power_cutoff = 3
        
        self.wavelet = "db4"
        self.wavelet_level = 1
        
        self.sg_window_size = 5,
        self.polyorder = 2
        

        self.moving_avg_window_size = 100
        self.filter_buffer_size = 100
        
        self.avg_buffer = {
          SignalType.ACC: [],
          SignalType.GYRO: [],
        }
        
        self.buffer = {
          SignalType.ACC: [],
          SignalType.GYRO: [],
        }

        self.uav_buffer = []
        # TODO: Set frequency dynamically
        self.ekf = EKF(frequency=48)
        self.q = np.array([1, 0, 0, 0])  # Initial quaternion
        
        self._set_filter_params(low_cutoff=2.5, high_cutoff=6., order=4.)
      
    def clear(self):
        for k in self.buffer.keys():
            self.buffer[k].clear()
    
    def _set_filter_params(self, low_cutoff, high_cutoff, order=5):
        nyq = 0.5 * self.freq
        low = low_cutoff/nyq
        high = high_cutoff/nyq
        self.b_low, self.a_low = butter(order, low, btype='lowpass')
        self.b_high, self.a_high = butter(order, high, btype='highpass')
        self.b_band, self.a_band = butter(order, [low, high], btype='band')
      
    def _madev(self, d, axis=None):
        """ Mean absolute deviation of a signal """
        return np.mean(np.absolute(d - np.mean(d, axis)), axis)
    
    def apply_moving_avg(self, data: np.ndarray, t: SignalType):
      
        self.buffer[t].append(data)
        if len(self.buffer[t]) > self.moving_avg_window_size:
            mean = np.mean(self.buffer[t], axis=0)
            self.buffer[t] = self.buffer[t][-self.moving_avg_window_size:]
            return mean
        
        return data
      
    def _process_filter(self, data: np.ndarray, t: SignalType, b, a):
        self.buffer[t].append(data)
        
        if len(self.buffer[t]) > self.filter_buffer_size:
            filtered = np.array([lfilter(b, a, value) for value in np.array(self.buffer[t]).T])
            self.buffer[t] = self.buffer[t][-self.filter_buffer_size:]
            return filtered[:, -1]
        
        return data
    
    def apply_lowpass_filter(self, data: np.ndarray, t: SignalType):
        return self._process_filter(data, t, self.b_low, self.a_low)
      
    def apply_highpass_filter(self, data: np.ndarray, t: SignalType):
        return self._process_filter(data, t, self.b_high, self.a_high)

    def apply_bandpass_filter(self, data: np.ndarray, t: SignalType):
        return self._process_filter(data, t, self.b_band, self.a_band)
    
    def _process_psd_filter(self, data: np.ndarray):
        n = data.shape[0] 
        fhat = np.fft.fft(data, n) # Compute the FFT
        PSD = fhat * np.conj(fhat) / n    # Power spectrum (power per frequency)
        
        indices = PSD > self.power_cutoff
        fhat_filtered = indices * fhat
        ffilt = np.fft.ifft(fhat_filtered).real
        
        return ffilt

    def apply_psd_filter(self, data: np.ndarray, t: SignalType):
        self.buffer[t].append(data)
        
        if len(self.buffer[t]) > self.filter_buffer_size:
            filtered = np.array([self._process_psd_filter(value) for value in np.array(self.buffer[t]).T])
            self.buffer[t] = self.buffer[t][-self.filter_buffer_size:]
            return filtered[:, -1]
        
        return data
    
    def apply_wavelet_filter(self, data: np.ndarray):
        coeff = pywt.wavedec(data, self.wavelet, mode="per")
        sigma = (1/0.6745) * self._madev(coeff[-self.wavelet_level])
        uthresh = sigma * np.sqrt(2 * np.log(len(data)))
        coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])
        return pywt.waverec(coeff, self.wavelet, mode='per')
    
    def apply_SG_filter(self, data: np.ndarray):
        """Savitzky-Golay filter
        https://ieeexplore.ieee.org/document/8713728
        """
        return savgol_filter(data, self.sg_window_size, self.polyorder)

    def _detect_anomaly(self, data: np.ndarray, buffer: list, threshold: float = 3.0) -> bool:
        """Detects anomalies in the data using z-score method

        Args:
            data (np.ndarray): Input data
            threshold (float, optional): Z-score threshold. Defaults to 3.0.

        Returns:
            bool: True if anomaly is detected, False otherwise
        """
        if len(buffer) < self.moving_avg_window_size:
            return False
        
        mean = np.mean(buffer, axis=0)
        std = np.std(buffer, axis=0)
        z_scores = np.abs((data - mean) / std)
        
        return np.any(z_scores > threshold)
    
    def _apply_actuator_preprocessing(self, sensor_data: SensorDataField) -> np.ndarray:

        assert self.hardware_config.drone_hardware_config is not None, "Hardware config is not set"

        motor_speed = sensor_data.data.u.flatten()
        self.uav_buffer.append(motor_speed)
        
        if self._detect_anomaly(data=motor_speed, buffer=self.uav_buffer, threshold=3.0):
            logging.warning("Anomaly detected in motor speed data")
            return None
            
        if len(self.uav_buffer) > self.moving_avg_window_size:
            motor_speed = np.mean(self.uav_buffer, axis=0)
            self.uav_buffer = self.uav_buffer[-self.moving_avg_window_size:]
        
        rpm_to_rad = (2*np.pi)/60
        omega1, omega2, omega3, omega4 = motor_speed * rpm_to_rad  # Convert RPM to rad/s
        omega_r = -omega1 + omega2 - omega3 + omega4 # Compute relative rotor speed

        return np.array([omega1, omega2, omega3, omega4, omega_r])
    
    def _apply_imu_preprocessing(self, sensor_data: SensorDataField) -> np.ndarray:
        """Returns preprocessed IMU data

        Args:
            sensor_data (SensorDataField): IMU sensor data

        Returns:
            np.ndarray: Control input vector u
        """
        # TODO: Make the filter process configurable.
        imu = sensor_data.data.u
        a, w = imu[:3], imu[3:]
        
        if self._detect_anomaly(data=a, buffer=self.buffer[SignalType.ACC], threshold=3.0) or \
            self._detect_anomaly(data=w, buffer=self.buffer[SignalType.GYRO], threshold=3.0):
            logging.warning("Anomaly detected in IMU data")
            return None
        
        if self.filter_config.use_imu_preprocessing:
            # NOTE: Apply lowpass filter
            a = self.apply_lowpass_filter(data=a, t=SignalType.ACC)
            w = self.apply_lowpass_filter(data=w, t=SignalType.GYRO)
            # NOTE: Apply rolling average
            a = self.apply_moving_avg(data=a, t=SignalType.ACC)
            w = self.apply_moving_avg(data=w, t=SignalType.GYRO)
        
        return np.hstack([a, w])

    def get_control_input(self, sensor_data: SensorDataField) -> np.ndarray:
        """Generates control input based on sensor type

        Args:
            sensor_data (SensorDataField): Data of Sensors, including IMU and Rotor speed

        Returns:
            np.ndarray: control input vector u
        """
        if SensorType.is_motor_data(sensor_data.type):
            return self._apply_actuator_preprocessing(sensor_data=sensor_data)
        else:
            return self._apply_imu_preprocessing(sensor_data=sensor_data)

    def get_angle_for_correction(self, sensor_data: SensorDataField) -> np.ndarray:
        """Returns the angle for correction based on sensor data

        Args:
            sensor_data (SensorDataField): IMU sensor data

        Returns:
            np.ndarray: Angle for correction
        """
        if not SensorType.is_imu_data_for_correction(sensor_data.type):
            return np.array([1., 0., 0., 0.]).reshape(-1, 1)
        
        imu = sensor_data.data.z.flatten()
        acc, gyro = imu[:3], imu[3:6]
        if imu.shape[0] > 6:
            mag = imu[6:9]
            mag = np.array([mag[2], mag[0], mag[1]])  # Adjust magnetometer axes
            q = self.ekf.update(self.q, gyr=gyro, acc=acc, mag=mag) 
        else:
            q = self.ekf.update(self.q, gyr=gyro, acc=acc)
        self.q = q
        return q.reshape(-1, 1)

    def set_initial_angle(self, q: np.ndarray):
        """Sets the initial angle for the EKF

        Args:
            q (np.ndarray): Initial quaternion
        """
        self.q = q