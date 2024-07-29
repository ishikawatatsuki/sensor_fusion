import serial
import numpy as np

def is_float_try_except(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def connection(port, baud_rate):
  try:
    ser = serial.Serial(port, baudrate=baud_rate, timeout=1)
    return ser
  except serial.SerialException as e:
    return None

def get_imu_data(config):
  ser1 = connection(config["port1"], baud_rate=config["baud_rate"])
  ser2 = connection(config["port2"], baud_rate=config["baud_rate"])

  while True:
    if ser1:
      try:
        data1 = ser1.readline().decode('utf-8').strip()
        if data1:
          val1 = np.array([np.float32(val) for val in data1.split(",") if is_float_try_except(val)])
          # val1 = np.array([0.1, 0., 0., 0., 0., 0.])
          yield {
            "label": "imu1",
            "value": val1,
          }
      except (serial.SerialException, OSError) as e:
        ser1.close()
        ser1 = connection(config["port1"], baud_rate=config["baud_rate"])
    else:
        ser1 = connection(config["port1"], baud_rate=config["baud_rate"])
      
      
    if ser2:
      try:
        data2 = ser2.readline().decode('utf-8').strip()
        if data2:
          val2 = np.array([np.float32(val) for val in data2.split(",") if is_float_try_except(val)])
          # val2 = np.array([0.1, 0., 0., 0., 0., 0.])
          yield {
            "label": "imu2",
            "value": val2,
          }
      except (serial.SerialException, OSError) as e:
        ser2.close()
        ser2 = connection(config["port2"], baud_rate=config["baud_rate"])
    else:
        ser2 = connection(config["port2"], baud_rate=config["baud_rate"])
      

if __name__ == "__main__":
  PORT1 = "/dev/cu.usbmodem1201"
  PORT2 = "/dev/cu.usbmodem11401"
  DEFAULT_BAUD_RATE = 115200
  
  config = {
    "port1": PORT1,
    "port2": PORT2,
    "baud_rate": DEFAULT_BAUD_RATE,
  }
  for value in get_imu_data(config):
    print(value)