import numpy as np


def get_gyroscope_noise(noise, sample_rate=50):
  """
    Assuming the noise is measured in dps/√Hz.
    e.g.: gyroscope noise = 0.0028 dps/√Hz
    * Converting degree into radian: 
    0.0028 * pi / 180 rad/s/√Hz = 4.89*10^-5 rad/s/√Hz
    * Calculating gyroscope noise in radian/second
    4.89*10^-5 * √50Hz = 0.0003457 rad/s 
  """
  noise *= np.pi / 180
  return noise * np.sqrt(sample_rate)


def get_acceleration_noise(noise, sample_rate=50):
  """
    Assuming the noise is measured in g/√Hz
    e.g.: gyroscope noise = 0.00007 g/√Hz
    * Converting g into m/s^2
    0.00007 * 9.81m/s^2 = 6.867*10^-4 m/s^2/√Hz
    * Calculating accelerometer noise in m/s^2
    6.867*10^-4 * √50Hz = 0.00486 m/s^2
  """
  noise *= 9.81
  return noise * np.sqrt(sample_rate)