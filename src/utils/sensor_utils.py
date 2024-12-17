import numpy as np


def get_gyroscope_noise(noise, sample_rate=50) -> np.float32:
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


def get_acceleration_noise(noise, sample_rate=50) -> np.float32:
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

def get_oxts_gyroscope_noise() -> np.float32:
  """
    OXTS gyroscope noise is specified in d/√hr
    e.g.: noise = 0.2
    * convert degree into radian:
    0.2 * pi / 180 = 0.00349 rad/√hr
    * power of 2
    0.00348**2 = 0.0000122 rad/hr
    * divided by 3600 to get second
    0.0439 rad/s
  """
  noise = 0.2 * np.pi / 180
  return noise**2 / 3600

def get_oxts_acceleration_noise() -> np.float32:
  """
  OXTS acceleration noise is represented in m/s2/√hr
  e.g.: acceleration noise = 0.005
  * Multiply √3600 
  0.005 * √3600 = 0.3 m/s2
  """
  return 0.005 * np.sqrt(3600)