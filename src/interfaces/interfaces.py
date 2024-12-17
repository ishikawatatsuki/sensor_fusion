import os 
import sys
import numpy as np
from scipy.spatial.transform import Rotation   


sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from custom_types import MotionModel

class State:
    
    def __init__(
        self,
        p: np.ndarray,
        q: np.ndarray,
        v: np.ndarray = None,
    ):
        """ 
          state to estimate: 
            kinematic motion model: [px, py, pz, vx, vy, vz, qw, qx, qy, qz] 
            velocity motion model : [px, py, pz, qw, qx, qy, qz]
        """
        self.p = p
        self.v = v
        self.q = q
        
    @classmethod
    def get_new_state_from_array(cls, x: np.ndarray):
      """create new state object

      Args:
          x (np.ndarray): current state vector as a numpy array
          motion_model (MotionModel): motion model enum

      Returns:
          State: new state object
      """
      x = x.flatten()
      return cls(p=x[:3].reshape(-1, 1), v=x[3:6].reshape(-1, 1), q=x[6:].reshape(-1, 1))
    
    def get_state_vector(self) -> np.ndarray:
      return np.vstack([self.p, self.v, self.q])
    
    def get_vector_size(self):
      return np.vstack([self.p, self.v, self.q]).shape[0]
    
    def get_rotation_matrix(self, q=None):
        if q is None:
          q = self.q
          
        return State.get_rotation_matrix_from_quaternion_vector(q.flatten())
    
    def get_euler_angle_from_quaternion(self, q=None):
        if q is None:
          q = self.q.flatten()
        
        return State.get_euler_angle_from_quaternion_vector(q.flatten())
      
    @staticmethod
    def get_euler_angle_from_quaternion_vector(q) -> np.ndarray:
      w, x, y, z = q
      
      t0 = +2.0 * (w * x + y * z)
      t1 = +1.0 - 2.0 * (x**2 + y**2)
      phi = np.arctan2(t0, t1)
    
      t2 = +2.0 * (w * y - z * x)
      t2 = +1.0 if t2 > +1.0 else t2
      t2 = -1.0 if t2 < -1.0 else t2
      theta = np.arcsin(t2)
    
      t3 = +2.0 * (w * z + x * y)
      t4 = +1.0 - 2.0 * (y**2 + z**2)
      psi = np.arctan2(t3, t4)
    
      return np.array([phi, theta, psi])
      
    @staticmethod
    def get_rotation_matrix_from_quaternion_vector(q) -> np.ndarray:
      q0, q1, q2, q3 = q
      # https://ahrs.readthedocs.io/en/latest/filters/ekf.html
      # https://www.iri.upc.edu/people/jsola/JoanSola/objectes/notes/kinematics.pdf
      return np.array([
          [q0**2 + q1**2 - q2**2 - q3**2, 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)],
          [2*(q1*q2 + q0*q3), q0**2 - q1**2 + q2**2 - q3**2, 2*(q2*q3 - q0*q1)],
          [2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), q0**2 - q1**2 - q2**2 + q3**2]
      ])

    @staticmethod
    def get_quaternion_from_rotation_matrix(R: np.ndarray) -> np.ndarray:
      assert R.shape == np.eye(3).shape, "Please provide 3x3 matrix"
      trace = np.trace(R)

      if trace > 0:
          S = 2.0 * np.sqrt(trace + 1.0)
          w = 0.25 * S
          x = (R[2, 1] - R[1, 2]) / S
          y = (R[0, 2] - R[2, 0]) / S
          z = (R[1, 0] - R[0, 1]) / S
      elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
          S = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
          w = (R[2, 1] - R[1, 2]) / S
          x = 0.25 * S
          y = (R[0, 1] + R[1, 0]) / S
          z = (R[0, 2] + R[2, 0]) / S
      elif R[1, 1] > R[2, 2]:
          S = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
          w = (R[0, 2] - R[2, 0]) / S
          x = (R[0, 1] + R[1, 0]) / S
          y = 0.25 * S
          z = (R[1, 2] + R[2, 1]) / S
      else:
          S = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
          w = (R[1, 0] - R[0, 1]) / S
          x = (R[0, 2] + R[2, 0]) / S
          y = (R[1, 2] + R[2, 1]) / S
          z = 0.25 * S

      return np.array([w, x, y, z]).reshape(-1, 1)
    
      # r11, r12, r13 = R[0, :]
      # r21, r22, r23 = R[1, :]
      # r31, r32, r33 = R[2, :]
      # _q0 = np.sqrt((1+r11+r22+r33)/4)
      # _q1 = np.sqrt((1+r11-r22-r33)/4)
      # _q2 = np.sqrt((1-r11+r22-r33)/4)
      # _q3 = np.sqrt((1-r11-r22+r33)/4)
      # if _q0 > _q1 and _q0 > _q2 and _q0 > _q3:
      #   div = 4*_q0
      #   _q1 = (r31-r23)/div
      #   _q2 = (r13-r31)/div
      #   _q3 = (r21-r12)/div
      # elif _q1 > _q0 and _q1 > _q2 and _q1 > _q3:
      #   div = 4*_q1
      #   _q0 = (r32-r23)/div
      #   _q2 = (r12-r21)/div
      #   _q3 = (r13-r31)/div
      # elif _q2 > _q0 and _q2 > _q1 and _q2 > _q3:
      #   div = 4*_q2
      #   _q0 = (r13-r31)/div
      #   _q1 = (r12-r21)/div
      #   _q3 = (r23-r32)/div
      # else:
      #   div = 4*_q3
      #   _q0 = (r21-r12)/div
      #   _q1 = (r13-r31)/div
      #   _q2 = (r23-r32)/div
        
      # return np.array([_q0, _q1, _q2, _q3]).reshape(-1, 1)
    
    @staticmethod
    def get_euler_angle_from_rotation_matrix(R: np.ndarray) -> np.ndarray:
      assert R.shape == np.eye(3).shape, "Please provide 3x3 matrix"
      
      r = Rotation.from_matrix(R)
      angles = r.as_euler('xyz', degrees=False)
      
      return angles.reshape(-1, 1)
    
    @staticmethod
    def get_quaternion_from_euler_angle(w: np.ndarray) -> np.ndarray:
      w = w.flatten()
      assert w.shape[0] == 3, "Please provide 3d vector"
      
      normalize_euler = lambda angle: (angle + np.pi) % (2 * np.pi) - np.pi
      
      roll, pitch, yaw = w
      
      roll = normalize_euler(roll)
      pitch = normalize_euler(pitch)
      yaw = normalize_euler(yaw)
        
      cr = np.cos(roll * 0.5)
      sr = np.sin(roll * 0.5)
      cp = np.cos(pitch * 0.5)
      sp = np.sin(pitch * 0.5)
      cy = np.cos(yaw * 0.5)
      sy = np.sin(yaw * 0.5)

      w = cr * cp * cy + sr * sp * sy
      x = sr * cp * cy - cr * sp * sy
      y = cr * sp * cy + sr * cp * sy
      z = cr * cp * sy - sr * sp * cy

      return np.array([w, x, y, z]).reshape(-1, 1)
      
class Pose:
  """
  3d rigid transform.
  """
  def __init__(self, R, t):
    assert R.shape == np.eye(3).shape, "Please set 3x3 rotation matrix"
    assert t.shape[0] == 3, "Please set 3x1 translation vector"
    
    self.R = R
    self.t = t if len(t.shape) == 1 else t.reshape(-1, )

  def matrix(self, pose_only=False):
    m = np.zeros((3, 4)) if pose_only else np.identity(4)
    m[:3, :3] = self.R
    m[:3, 3] = self.t
    return m

  def inverse(self):
    return Pose(self.R.T, -self.R.T @ self.t)

  def __mul__(self, T1):
    R = self.R @ T1.R
    t = self.R @ T1.t + self.t
    return Pose(R, t)

  @classmethod
  def from_state(cls, state: State):
    R = state.get_rotation_matrix()
    t = state.p.flatten()
    return cls(R=R, t=t)
  
  def from_ned_to_enu(self):
    return Pose(R=np.array([
      [0, 1, 0],
      [1, 0, 0],
      [0, 0, -1]
    ]), t=np.zeros((3, 1))) * Pose(R=self.R, t=self.t)
    
  def from_enu_to_ned(self):
    return Pose(R=np.array([
      [0, 1, 0],
      [1, 0, 0],
      [0, 0, -1]
    ]), t=np.zeros((3, 1))) * Pose(R=self.R, t=self.t)

if __name__ == "__main__":

  def _main1():
    t1 = np.array([0.5, 0.5, 0.5])
    t2 = np.array([1., 1., 1.])
    R1 = np.eye(3)
    R2 = np.eye(3)
    p1 = Pose(R=R1, t=t1)
    p2 = Pose(R=R2, t=t2)
    p3 = p1 * p2
    
    t1 = np.array([0.5, 0.5, 0.5])
    t2 = np.array([1., 1., 1.])
    R1 = np.eye(3)
    R2 = np.eye(3)
    p1 = Pose(R=R1, t=t1)
    p2 = Pose(R=R2, t=t2)
    p3 = p1 * p2
    euler1 = State.get_euler_angle_from_rotation_matrix(p1.R)
    euler2 = State.get_euler_angle_from_rotation_matrix(p3.R)
    
    expected_delta_translation = t2
    expected_delta_angle = euler2 - euler1
    
    result = p1.inverse() * p3
    calculated_delta_translation = result.t
    calculated_delta_angle = State.get_euler_angle_from_rotation_matrix(result.R).flatten()
    
    print(f"Expected translation: {expected_delta_translation}, calculated: {calculated_delta_translation}")
    print(f"Expected angle: {expected_delta_angle.flatten()}, calculated: {calculated_delta_angle}")
    
    
    t1 = np.array([1.6378729, 4.406781, -6.663874])
    t2 = np.array([1.6137391, 4.4345922, -6.661531])
    q1 = np.array([-0.51108545, 0.06378497, -0.020988196, -0.85690296])
    q2 = np.array([-0.5117634, 0.07358955, -0.020333096, -0.8557274])
    q1 /= np.linalg.norm(q1)
    q2 /= np.linalg.norm(q2)
    R1 = State.get_rotation_matrix_from_quaternion_vector(q1)
    R2 = State.get_rotation_matrix_from_quaternion_vector(q2)
    p1 = Pose(R=R1, t=t1)
    p3 = Pose(R=R2, t=t2)
    
    euler1 = State.get_euler_angle_from_quaternion_vector(q1)
    euler2 = State.get_euler_angle_from_quaternion_vector(q2)
    
    expected_delta_translation = np.linalg.inv(R1) @ (t2 - t1)
    expected_delta_angle = euler2 - euler1
    
    p2 = p1.inverse() * p3
    calculated_delta_translation = p2.t
    calculated_delta_angle = State.get_euler_angle_from_rotation_matrix(p2.R).flatten()
    
    print(f"Expected translation: {expected_delta_translation}, calculated: {calculated_delta_translation}")
    print(f"Expected angle: {expected_delta_angle.flatten()}, calculated: {calculated_delta_angle}")
    
    
  def _main2():
    p1 = Pose(R=np.array([
      [0.866, -0.5, 0.],
      [0.5, 0.866, 0.],
      [0., 0., 1.]
    ]), t=np.ones((3, 1)))
    
    p2 = p1.from_ned_to_enu()
    p3 = p2.from_enu_to_ned()
    print(p2.R)
    print(p3.R)
    
    
  # _main1()
  
  _main2()