import os
import sys
import numpy as np
from ahrs.filters import EKF

from .base_filter import BaseFilter
from ..common import (
    State, MeasurementUpdateField,
    MeasurementUpdateField,
)

class ExtendedKalmanFilter(BaseFilter):
    
    def __init__(
        self, 
        *args,
        **kwargs,
        ):
        super().__init__(
            *args, 
            **kwargs
        )
        self.predict = self._get_motion_model()
    
    def kinematics_motion_model(self, u: np.ndarray, dt: float, Q: np.ndarray):
        """estimate x and P based on previous stete of x and control input u
        Args:
            u  (numpy.array): control input u
            dt (numpy.array): difference of current time and previous time
            Q  (numpy.array): process noise 
        """
        # propagate state x
        p = self.x.p
        v = self.x.v
        q = self.x.q
        b_w = self.x.b_w
        b_a = self.x.b_a

        a = u[:3]
        w = u[3:]
        a = a.reshape(-1, 1) 
        w = w.reshape(-1, 1)

        # Take into account the IMU sensor error
        imu_sensor_error = self.get_imu_sensor_error()
        b_a_k = imu_sensor_error.acc_bias + b_a
        b_w_k = imu_sensor_error.gyro_bias + b_w

        a -= b_a_k + imu_sensor_error.acc_noise
        w -= b_w_k + imu_sensor_error.gyro_noise

        R = self.x.get_rotation_matrix()
        omega = self.get_quaternion_update_matrix(w)
        norm_w = self.compute_norm_w(w)

        A = np.cos(norm_w*dt/2) * np.eye(4)
        B = (1/norm_w)*np.sin(norm_w*dt/2) * omega
        
        acc_val = (R @ a - self.g)
        p_k = p + v * dt #+ acc_val*dt**2 / 2
        v_k = v + acc_val * dt
        q_k = np.array(A + B) @ q
        q_k /= np.linalg.norm(q_k)

        self.x = State(p=p_k, v=v_k, q=q_k, b_w=b_w_k, b_a=b_a_k)
        
        ax, ay, az, wx, wy, wz = u
        q1, q2, q3, q4 = q.flatten()
        dt2 = dt**2
        cos_w = np.cos(norm_w*dt/2)
        sin_w = np.sin(norm_w*dt/2)/norm_w
        wz_sin = wz*sin_w
        wy_sin = wy*sin_w
        wx_sin = wx*sin_w

        # Jacobian matrix of function f(x,u) with respect to the state variables.
        F = np.eye(self.x.get_vector_size())
        F[0, 3] = dt
        F[0, 6] = dt2*(2*ax*q1-2*ay*q4+2*az*q3)/2
        F[0, 7] = dt2*(2*ax*q2+2*ay*q3+2*az*q4)/2
        F[0, 8] = dt2*(-2*ax*q3+2*ay*q2+2*az*q1)/2
        F[0, 9] = dt2*(-2*ax*q4-2*ay*q1+2*az*q2)/2

        F[1, 4] = dt
        F[1, 6] = dt2*(2*ax*q4+2*ay*q1-2*az*q2)/2
        F[1, 7] = dt2*(2*ax*q3-2*ay*q2-2*az*q1)/2
        F[1, 8] = dt2*(2*ax*q2+2*ay*q3+2*az*q4)/2
        F[1, 9] = dt2*(2*az*q1-2*ay*q4+2*az*q3)/2

        F[2, 5] = dt
        F[2, 6] = dt2*(-2*ax*q3+2*ay*q2+2*az*q1)/2
        F[2, 7] = dt2*(2*ax*q4+2*ay*q1-2*az*q2)/2
        F[2, 8] = dt2*(-2*ax*q1+2*ay*q4-2*az*q3)/2
        F[2, 9] = dt2*(2*ax*q2+2*ay*q3+2*az*q4)/2

        F[3, 6] = dt*(2*ax*q1-2*ay*q4+2*az*q3)
        F[3, 7] = dt*(2*ax*q2+2*ay*q3+2*az*q4)
        F[3, 8] = dt*(-2*ax*q3+2*ay*q2+2*az*q1)
        F[3, 9] = dt*(-2*ax*q4-2*ay*q1+2*az*q2)

        F[4, 6] = dt*(2*ax*q4+2*ay*q1-2*az*q2)
        F[4, 7] = dt*(2*ax*q3-2*ay*q2-2*az*q1)
        F[4, 8] = dt*(2*ax*q2+2*ay*q3+2*az*q4)
        F[4, 9] = dt*(2*ax*q1-2*ay*q4+2*az*q3)

        F[5, 6] = dt*(-2*ax*q3+2*ay*q2+2*az*q1)
        F[5, 7] = dt*(2*ax*q4+2*ay*q1-2*ay*q2)
        F[5, 8] = dt*(-2*ax*q1+2*ay*q4-2*az*q3)
        F[5, 9] = dt*(2*ax*q2+2*ay*q3+2*az*q4)

        F[6, 6] = cos_w
        F[6, 7] = wz_sin
        F[6, 8] = -wy_sin
        F[6, 9] = wx_sin

        F[7, 6] = -wz_sin
        F[7, 7] = cos_w
        F[7, 8] = wx_sin
        F[7, 9] = wy_sin

        F[8, 6] = wy_sin
        F[8, 7] = -wx_sin
        F[8, 8] = cos_w
        F[8, 9] = wz_sin

        F[9, 6] = -wx_sin
        F[9, 7] = -wy_sin
        F[9, 8] = -wz_sin
        F[9, 9] = cos_w
        
        # Jacobian matrix of function f(x,u) with respect to the control input variables.
        # G = np.array([
        #     [dt2*(q1_2+q2_2-q3_2-q4_2)/2, dt2*(-2*q1*q4+2*q2*q3)/2, dt2*(2*q1*q3+2*q2*q4)/2, 0., 0., 0.],
        #     [dt2*(2*q1*q4+2*q2*q3)/2, dt2*(q1_2-q2_2+q3_2-q4_2)/2, dt2*(-2*q1*q2+2*q3*q4)/2, 0., 0., 0.],
        #     [dt2*(-2*q1*q3+2*q2*q4)/2, dt2*(2*q1*q2+2*q3*q4)/2, dt2*(q1_2-q2_2-q3_2+q4_2)/2, 0., 0., 0.],
            
        #     [dt*(q1_2+q2_2-q3_2-q4_2), dt*(-2*q1*q4+2*q2*q3), dt*(2*q1*q3+2*q2*q4), 0., 0., 0.],
        #     [dt*(2*q1*q4+2*q2*q3), dt*(q1_2-q2_2+q3_2-q4_2), dt*(-2*q1*q2+2*q3*q4), 0., 0., 0.],
        #     [dt*(-2*q1*q3+2*q2*q4), dt*(2*q1*q2+2*q3*q4), dt*(q1_2-q2_2-q3_2+q4_2), 0., 0., 0.],

        #     [0., 0., 0., q4*sin_w, -q3*sin_w, q2*sin_w],
        #     [0., 0., 0., q3*sin_w, q4*sin_w, -q1*sin_w],
        #     [0., 0., 0., -q2*sin_w, q1*sin_w, q4*sin_w],
        #     [0., 0., 0., -q1*sin_w, -q2*sin_w, -q3*sin_w]
        # ])
        # predict state covariance matrix P
        self.P = F @ self.P @ F.T + Q # G @ Q @ G.T
    '''
        def velocity_motion_model(self, u: np.ndarray, dt: float):
        """estimate x and P based on previous stete of x and control input u
        Args:
            u  (numpy.array): control input u
            dt (numpy.array): difference of current time and previous time
            Q  (numpy.array): process noise 
        """
        p = self.x.p
        q = self.x.q
        
        v, wx, wy, wz = u
        w = np.array([wx, wy, wz]).reshape(-1, 1)
        
        omega = self.get_quaternion_update_matrix(w)
        norm_w = self.compute_norm_w(w)
        phi, _, psi = self.get_euler_angle_from_quaternion(q)
        
        rx = v / wx  # turning radius for x axis
        rz = v / wz  # turning radius for z axis
        
        dphi = wx * dt
        dpsi = wz * dt
        dpx = - rz * np.sin(psi) + rz * np.sin(psi + dpsi)
        dpy = + rz * np.cos(psi) - rz * np.cos(psi + dpsi)
        dpz = + rx * np.cos(phi) - rx * np.cos(phi + dphi)
        
        
        dp = np.array([dpx, dpy, dpz]).reshape(-1, 1)
        
        p_k = p + dp
        
        A = np.cos(norm_w*dt/2) * np.eye(4)
        B = (1/norm_w)*np.sin(norm_w*dt/2) * omega
        
        q_k = np.array(A + B) @ q
        q_k /= np.linalg.norm(q_k)
        
        self.x = State(p=p_k, v=self.x.v, q=q_k)
        

        # propagate covariance P
        qw, qx, qy, qz = q[:, 0]
        # Jacobian of state transition function
        # NOTE: the jacobian of transition matrix F is obtained from setup3_in_3d_coordinate_with_quaternion.ipynb
        
        q_x_squared = qw**2+qx**2-qy**2-qz**2
        q_z_squared = qw**2-qx**2-qy**2+qz**2
        qw_qz_qx_qy = 2*qw*qz+2*qx*qy
        qw_qz_qx_qy_minus = -2*qw*qz-2*qx*qy
        qw_qx_qy_qz = 2*qw*qx+2*qy*qz
        qw_qx_qy_qz_minus = -2*qw*qx-2*qy*qz
        F = np.eye(7)
        F[0, 3] = - (2*qz*v) / (wz*np.sqrt((qw_qz_qx_qy)**2 + (q_x_squared)**2))\
                    - (v*(qw_qz_qx_qy)*(-2*qw*(q_x_squared) -2*qz*(qw_qz_qx_qy))) / wz*((qw_qz_qx_qy)**2 + (q_x_squared)**2)**(3/2) + (v*(2*qw*(qw_qz_qx_qy_minus)/((qw_qz_qx_qy)**2+(q_x_squared)**2)\
                    + 2*qz*(q_x_squared)/((qw_qz_qx_qy)**2+(q_x_squared)**2)) * np.cos(dt*wz + np.arctan2(qw_qz_qx_qy, q_x_squared))) / (wz)
        F[0, 4] = - (2*qy*v)/(wz*np.sqrt((qw_qz_qx_qy)**2 + (q_x_squared)**2))\
                    - (v*(qw_qz_qx_qy)*(-2*qx*(q_x_squared) -2*qy*(qw_qz_qx_qy))) / wz*((qw_qz_qx_qy)**2 + (q_x_squared)**2)**(3/2)\
                    + (v*( (2*qx*(qw_qz_qx_qy_minus))/((qw_qz_qx_qy)**2 + (q_x_squared)**2) + (2*qy*(q_x_squared))/((qw_qz_qx_qy)**2 + (q_x_squared)**2) ) *np.cos(dt*qz + np.arctan2(qw_qz_qx_qy, q_x_squared))) / (wz)
        F[0, 5] = - (2*qx*v) / (wz*np.sqrt((qw_qz_qx_qy)**2 + (q_x_squared)**2))\
                    - (v*(qw_qz_qx_qy)*(-2*qx*(qw_qz_qx_qy)+2*qy*(q_x_squared))) / wz*((qw_qz_qx_qy)**2 + (q_x_squared)**2)**(3/2)\
                    + (v*( (2*qx*(q_x_squared))/((qw_qz_qx_qy)**2+(q_x_squared)**2) - (2*qy*(qw_qz_qx_qy_minus))/((qw_qz_qx_qy)**2 +(q_x_squared)**2) )*np.cos(dt*wz + np.arctan2(qw_qz_qx_qy, q_x_squared))) / (wz)
        F[0, 6] = - (2*qw*v) / (wz*np.sqrt((qw_qz_qx_qy)**2 + (q_x_squared)**2))\
                    - (v*(qw_qz_qx_qy)*(-2*qw*(qw_qz_qx_qy) +2*qz*(q_x_squared))) / wz*((qw_qz_qx_qy)**2 +(q_x_squared)**2)**(3/2)\
                    + (v*((2*qw*(q_x_squared))/((qw_qz_qx_qy)**2+(q_x_squared)**2) - (2*qz*(qw_qz_qx_qy_minus))/((qw_qz_qx_qy)**2 +(q_x_squared)**2))*np.cos(dt*wz +np.arctan2(qw_qz_qx_qy, q_x_squared))) / (wz)

        F[1, 3] = + (2*qw*v) / (wz*np.sqrt((qw_qz_qx_qy)**2 + (q_x_squared)**2))\
                    + (v*(-2*qw*(q_x_squared)-2*qz*(qw_qz_qx_qy))*(q_x_squared)) / wz*((qw_qz_qx_qy)**2 + (q_x_squared)**2)**(3/2)\
                    + (v*(2*qw*(qw_qz_qx_qy_minus)/((qw_qz_qx_qy)**2+(q_x_squared)**2) + 2*qz*(q_x_squared)/((qw_qz_qx_qy)**2+(q_x_squared)**2)) * np.sin(dt*wz + np.arctan2(qw_qz_qx_qy, q_x_squared))) / (wz)
        F[1, 4] = + (2*qx*v)/(wz*np.sqrt((qw_qz_qx_qy)**2 + (q_x_squared)**2))\
                    + (v*(qw_qz_qx_qy)*(-2*qx*(q_x_squared)-2*qy*(qw_qz_qx_qy))) / wz*((qw_qz_qx_qy)**2 + (q_x_squared)**2)**(3/2)\
                    + (v*( (2*qx*(qw_qz_qx_qy_minus))/((qw_qz_qx_qy)**2 + (q_x_squared)**2) + (2*qy*(q_x_squared))/((qw_qz_qx_qy)**2 + (q_x_squared)**2) )*np.sin(dt*qz + np.arctan2(qw_qz_qx_qy, q_x_squared))) / (wz)
        F[1, 5] = - (2*qy*v) / (wz*np.sqrt((qw_qz_qx_qy)**2 + (q_x_squared)**2))\
                    - (v*(-2*qx*(qw_qz_qx_qy)+2*qy*(q_x_squared))*(q_x_squared)) / wz*((qw_qz_qx_qy)**2 + (q_x_squared)**2)**(3/2)\
                    + (v*( (2*qx*(q_x_squared))/((qw_qz_qx_qy)**2+(q_x_squared)**2) - (2*qy*(qw_qz_qx_qy_minus))/((qw_qz_qx_qy)**2 +(q_x_squared)**2) )*np.sin(dt*wz + np.arctan2(qw_qz_qx_qy, q_x_squared))) / (wz)
        F[1, 6] = - (2*qz*v) / (wz*np.sqrt((qw_qz_qx_qy)**2 + (q_x_squared)**2))\
                    + (v*(-2*qw*(qw_qz_qx_qy) +2*qy*(q_x_squared))*(q_x_squared)) / wz*((qw_qz_qx_qy)**2 +(q_x_squared)**2)**(3/2)\
                    + (v*((2*qw*(q_x_squared))/((qw_qz_qx_qy)**2+(q_x_squared)**2) - (2*qz*(qw_qz_qx_qy_minus))/((qw_qz_qx_qy)**2 +(q_x_squared)**2))*np.sin(dt*wz +np.arctan2(qw_qz_qx_qy, q_x_squared))) / (wz)
                    
        F[2, 3] = - (2*qx*v) / (wx*np.sqrt((qw_qx_qy_qz)**2 + (q_z_squared)**2))\
                    - (v*(qw_qx_qy_qz)*(-2*qw*(q_z_squared) -2*qx*(qw_qx_qy_qz))) / wx*((qw_qx_qy_qz)**2 + (q_z_squared)**2)**(3/2)\
                    + (v*(2*qw*(qw_qx_qy_qz_minus)/((qw_qx_qy_qz)**2+(q_z_squared)**2) + 2*qx*(q_z_squared)/((qw_qx_qy_qz)**2+(q_z_squared)**2)) * np.cos(dt*wx + np.arctan2(qw_qx_qy_qz, q_z_squared))) / (wx)
        F[2, 4] = - (2*qw*v)/(wx*np.sqrt((qw_qx_qy_qz)**2 + (q_z_squared)**2))\
                    - (v*(qw_qx_qy_qz)*(-2*qw*(qw_qx_qy_qz)+2*qx*(q_z_squared))) / wx*((qw_qx_qy_qz)**2 + (q_z_squared)**2)**(3/2)\
                    - (v*( (2*qw*(q_z_squared))/((qw_qx_qy_qz)**2 + (q_z_squared)**2) - (2*qx*(qw_qx_qy_qz_minus))/((qw_qx_qy_qz)**2+(q_z_squared)**2) ) *np.cos(dt*qz + np.arctan2(qw_qx_qy_qz, q_z_squared))) / (wz)
        F[2, 5] = - (2*qz*v) / (wx*np.sqrt((2*qw*qx+ 2*qy*qz)**2 + (q_z_squared)**2))\
                    - (v*(2*qw*qx)*(2*qy*(q_z_squared) -2*qz*(qw_qx_qy_qz))) / wx*((qw_qx_qy_qz)**2 + (q_z_squared)**2)**(3/2)\
                    + (v*(-(2*qy*(qw_qx_qy_qz_minus))/((qw_qx_qy_qz)**2 +(q_z_squared)**2) +(2*qz*(q_z_squared))/((qw_qx_qy_qz)**2 +(q_z_squared)**2))*np.cos(dt*wx + np.arctan2(qw_qx_qy_qz, q_z_squared))) / (wz)
        F[2, 6] = - (2*qy*v) / (wx*np.sqrt((qw_qx_qy_qz)**2 + (q_z_squared)**2))\
                    - (v*(qw_qx_qy_qz)*(-2*qy*(qw_qx_qy_qz) -2*qz*(q_z_squared))) / wx*((qw_qx_qy_qz)**2 +(q_z_squared)**2)**(3/2)\
                    + (v*((2*qy*(q_z_squared))/((qw_qx_qy_qz)**2+(q_z_squared)**2) + (2*qz*(qw_qx_qy_qz_minus))/((qw_qx_qy_qz)**2 +(q_z_squared)**2))*np.cos(dt*wx +np.arctan2(qw_qx_qy_qz, q_z_squared))) / (wz)
        F[3, 3] = np.cos(dt*norm_w/2)
        F[3, 4] = wz*np.sin(dt*norm_w) / norm_w
        F[3, 5] = -wy*np.sin(dt*norm_w) / norm_w
        F[3, 6] = wx*np.sin(dt*norm_w) / norm_w
        
        F[4, 3] = -wz*np.sin(dt*norm_w) / norm_w
        F[4, 4] = np.cos(dt*norm_w/2)
        F[4, 5] = wx*np.sin(dt*norm_w/2) / norm_w
        F[4, 6] = wy*np.sin(dt*norm_w/2) / norm_w
        
        F[5, 3] = wy*np.sin(dt*norm_w/2) / norm_w
        F[5, 4] = -wx*np.sin(dt*norm_w/2) / norm_w
        F[5, 5] = np.cos(dt*norm_w/2)
        F[5, 6] = wz*np.sin(dt*norm_w/2) / norm_w
        
        F[6, 3] = -wx*np.sin(dt*norm_w/2) / norm_w
        F[6, 4] = -wy*np.sin(dt*norm_w/2) / norm_w
        F[6, 5] = -wz*np.sin(dt*norm_w/2) / norm_w
        F[6, 6] = np.cos(dt*norm_w/2)

        # Jacobian of state transition function
        G = np.zeros((7, 4))
        
        G[0, 0] = - (qw_qz_qx_qy)/(wz*np.sqrt((qw_qz_qx_qy)**2 + (q_x_squared)**2)) + (np.sin(dt*wz + np.arctan2(qw_qz_qx_qy, q_x_squared)))/wz
        G[0, 3] = (dt*v*np.cos(dt*wz + np.arctan2(qw_qz_qx_qy, q_x_squared)))/wz + (v*(qw_qz_qx_qy))/(wz**2*np.sqrt((qw_qz_qx_qy)**2 + (q_x_squared)**2)) - (v*np.sin(dt*wz + np.arctan2(qw_qz_qx_qy, q_x_squared)))/wz**2
        
        G[1, 0] = - (np.cos(dt*wz + np.arctan2(qw_qz_qx_qy, q_x_squared)))/wz + (q_x_squared)/(wz*np.sqrt((qw_qz_qx_qy)**2 + (q_x_squared)**2))
        G[1, 3] = (dt*v*np.sin(dt*wz +np.arctan2(qw_qz_qx_qy, q_x_squared)))/wz + (v*np.cos(dt*wz+np.arctan2(qw_qz_qx_qy, q_x_squared)))/wz**2 -(v*(q_x_squared))/(wz**2*np.sqrt((qw_qz_qx_qy)**2+(q_x_squared)**2))
        
        G[2, 0] = - (qw_qx_qy_qz)/(wx*np.sqrt((qw_qx_qy_qz)**2+(q_z_squared)**2)) + np.sin(dt*wx+np.arctan2(qw_qx_qy_qz, q_z_squared)) / wx
        G[2, 1] = dt*v*np.cos(dt*wx+np.arctan2(qw_qx_qy_qz, q_z_squared))/wx + v*(qw_qx_qy_qz)/(wx**2*np.sqrt((qw_qx_qy_qz)**2+(q_z_squared)**2)) - v*np.sin(dt*wx+np.arctan2(qw_qx_qy_qz, q_z_squared))/wx**2
        
        G[3, 1] = qz*np.sin(dt*norm_w/2)/norm_w
        G[3, 2] = -qy*np.sin(dt*norm_w/2)/norm_w
        G[3, 3] = qx*np.sin(dt*norm_w/2)/norm_w
        
        G[4, 1] = qy*np.sin(dt*norm_w/2)/norm_w
        G[4, 2] = qz*np.sin(dt*norm_w/2)/norm_w
        G[4, 3] = -qw*np.sin(dt*norm_w/2)/norm_w
        
        G[5, 1] = -qx*np.sin(dt*norm_w/2)/norm_w
        G[5, 2] = qw*np.sin(dt*norm_w/2)/norm_w
        G[5, 3] = qz*np.sin(dt*norm_w/2)/norm_w
        
        G[6, 1] = -qw*np.sin(dt*norm_w/2)/norm_w
        G[6, 2] = -qx*np.sin(dt*norm_w/2)/norm_w
        G[6, 3] = -qy*np.sin(dt*norm_w/2)/norm_w
        
        self.P = F @ self.P @ F.T + G @ Q @ G.T
    
    '''

    def velocity_motion_model(self, u: np.ndarray, dt: float, Q: np.ndarray):
        """estimate x and P based on previous stete of x and control input u
        Args:
            u  (numpy.array): control input u
            dt (numpy.array): difference of current time and previous time
            Q  (numpy.array): process noise 
        """
        p = self.x.p
        v = self.x.v
        q = self.x.q
        b_w = self.x.b_w
        b_a = self.x.b_a
        
        a = u[:3]
        w = u[3:]
        wx, wy, wz = w
        a = a.reshape(-1, 1)
        w = w.reshape(-1, 1)

        # Take into account the IMU sensor error
        imu_sensor_error = self.get_imu_sensor_error()
        b_a_k = imu_sensor_error.acc_bias + b_a
        b_w_k = imu_sensor_error.gyro_bias + b_w

        a -= b_a_k + imu_sensor_error.acc_noise
        w -= b_w_k + imu_sensor_error.gyro_noise
        
        vf = self.get_forward_velocity(v)
        
        omega = self.get_quaternion_update_matrix(w)
        norm_w = self.compute_norm_w(w)
        phi, _, psi = self.get_euler_angle_from_quaternion(q)
        R = self.x.get_rotation_matrix()
        
        acc_val = (R @ a - self.g)
        
        v_k = v + acc_val * dt
        
        rx = vf / wx  # turning radius for x axis
        rz = vf / wz  # turning radius for z axis
        
        dphi = wx * dt
        dpsi = wz * dt
        dpx = - rz * np.sin(psi) + rz * np.sin(psi + dpsi)
        dpy = + rz * np.cos(psi) - rz * np.cos(psi + dpsi)
        dpz = + rx * np.cos(phi) - rx * np.cos(phi + dphi)
        
        
        dp = np.array([dpx, dpy, dpz]).reshape(-1, 1)
        
        p_k = p + dp
        
        A = np.cos(norm_w*dt/2) * np.eye(4)
        B = (1/norm_w)*np.sin(norm_w*dt/2) * omega
        
        q_k = np.array(A + B) @ q
        q_k /= np.linalg.norm(q_k)

        self.x = State(p=p_k, v=v_k, q=q_k, b_w=b_w_k, b_a=b_a_k)
        

        # propagate covariance P
        qw, qx, qy, qz = q[:, 0]
        vx, vy, vz = v[:, 0]
        ax, ay, az = a[:, 0]
        # Jacobian of state transition function
        # NOTE: the jacobian of transition matrix F is obtained from setup3_in_3d_coordinate_with_quaternion.ipynb
        
        q_x_squared = qw**2+qx**2-qy**2-qz**2
        q_z_squared = qw**2-qx**2-qy**2+qz**2
        qw_qz_qx_qy = 2*qw*qz+2*qx*qy
        qw_qz_qx_qy_minus = -2*qw*qz-2*qx*qy
        qw_qx_qy_qz = 2*qw*qx+2*qy*qz
        qw_qx_qy_qz_minus = -2*qw*qx-2*qy*qz
        F = np.eye(self.x.get_vector_size())
        
        F[0, 3] = - vx*(qw_qz_qx_qy)/(wz*np.sqrt(qw_qz_qx_qy**2+q_x_squared**2)*vf)\
                    + vx*np.sin(dt*wz+np.arctan2(qw_qz_qx_qy, q_x_squared)) / (wz*vf)
        F[0, 4] = - vy*(qw_qz_qx_qy)/(wz*np.sqrt(qw_qz_qx_qy**2+q_x_squared**2)*vf)\
                    + vy*np.sin(dt*wz+np.arctan2(qw_qz_qx_qy, q_x_squared)) / (wz*vf)
        F[0, 5] = - vz*(qw_qz_qx_qy)/(wz*np.sqrt(qw_qz_qx_qy**2+q_x_squared**2)*vf)\
                    + vz*np.sin(dt*wz+np.arctan2(qw_qz_qx_qy, q_x_squared)) / (wz*vf)
        
        F[0, 6] = - (2*qz*vf) / (wz*np.sqrt((qw_qz_qx_qy)**2 + (q_x_squared)**2))\
                    - (vf*(qw_qz_qx_qy)*(-2*qw*(q_x_squared) -2*qz*(qw_qz_qx_qy))) / (wz*((qw_qz_qx_qy)**2 + (q_x_squared)**2)**(3/2))\
                    + (vf*(2*qw*(qw_qz_qx_qy_minus)/((qw_qz_qx_qy)**2+(q_x_squared)**2) + 2*qz*(q_x_squared)/((qw_qz_qx_qy)**2+(q_x_squared)**2)) * np.cos(dt*wz + np.arctan2(qw_qz_qx_qy, q_x_squared))) / (wz)
        F[0, 7] = - (2*qy*vf)/(wz*np.sqrt((qw_qz_qx_qy)**2 + (q_x_squared)**2))\
                    - (vf*(qw_qz_qx_qy)*(-2*qx*(q_x_squared) -2*qy*(qw_qz_qx_qy))) / (wz*((qw_qz_qx_qy)**2 + (q_x_squared)**2)**(3/2))\
                    + (vf*( (2*qx*(qw_qz_qx_qy_minus))/((qw_qz_qx_qy)**2 + (q_x_squared)**2) + (2*qy*(q_x_squared))/((qw_qz_qx_qy)**2 + (q_x_squared)**2) ) *np.cos(dt*wz + np.arctan2(qw_qz_qx_qy, q_x_squared))) / (wz)
        F[0, 8] = - (2*qx*vf) / (wz*np.sqrt((qw_qz_qx_qy)**2 + (q_x_squared)**2))\
                    - (vf*(qw_qz_qx_qy)*(-2*qx*(qw_qz_qx_qy)+2*qy*(q_x_squared))) / (wz*((qw_qz_qx_qy)**2 + (q_x_squared)**2)**(3/2))\
                    + (vf*( (2*qx*(q_x_squared))/((qw_qz_qx_qy)**2+(q_x_squared)**2) - (2*qy*(qw_qz_qx_qy_minus))/((qw_qz_qx_qy)**2 +(q_x_squared)**2) )*np.cos(dt*wz + np.arctan2(qw_qz_qx_qy, q_x_squared))) / (wz)
        F[0, 9] = - (2*qw*vf) / (wz*np.sqrt((qw_qz_qx_qy)**2 + (q_x_squared)**2))\
                    - (vf*(qw_qz_qx_qy)*(-2*qw*(qw_qz_qx_qy) +2*qz*(q_x_squared))) / (wz*((qw_qz_qx_qy)**2 +(q_x_squared)**2)**(3/2))\
                    + (vf*((2*qw*(q_x_squared))/((qw_qz_qx_qy)**2+(q_x_squared)**2) - (2*qz*(qw_qz_qx_qy_minus))/((qw_qz_qx_qy)**2 +(q_x_squared)**2))*np.cos(dt*wz +np.arctan2(qw_qz_qx_qy, q_x_squared))) / (wz)


        F[1, 3] = - vx*np.cos(dt*wz+np.arctan2(qw_qz_qx_qy, q_x_squared))/(wz*vf)\
                    + (vx*q_x_squared)/(wz*np.sqrt(qw_qz_qx_qy**2+q_x_squared**2)*vf)
        F[1, 4] = - vy*np.cos(dt*wz+np.arctan2(qw_qz_qx_qy, q_x_squared))/(wz*vf)\
                    + (vy*q_x_squared)/(wz*np.sqrt(qw_qz_qx_qy**2+q_x_squared**2)*vf)
        F[1, 5] = - vz*np.cos(dt*wz+np.arctan2(qw_qz_qx_qy, q_x_squared))/(wz*vf)\
                    + (vz*q_x_squared)/(wz*np.sqrt(qw_qz_qx_qy**2+q_x_squared**2)*vf)
                    
        F[1, 6] = + (2*qw*vf) / (wz*np.sqrt((qw_qz_qx_qy)**2 + (q_x_squared)**2))\
                    + (vf*(-2*qw*(q_x_squared)-2*qz*(qw_qz_qx_qy))*(q_x_squared)) / (wz*((qw_qz_qx_qy)**2 + (q_x_squared)**2)**(3/2))\
                    + (vf*(2*qw*(qw_qz_qx_qy_minus)/((qw_qz_qx_qy)**2+(q_x_squared)**2) + 2*qz*(q_x_squared)/((qw_qz_qx_qy)**2+(q_x_squared)**2)) * np.sin(dt*wz + np.arctan2(qw_qz_qx_qy, q_x_squared))) / (wz)
        F[1, 7] = + (2*qx*vf)/(wz*np.sqrt((qw_qz_qx_qy)**2 + (q_x_squared)**2))\
                    + (vf*(qw_qz_qx_qy)*(-2*qx*(q_x_squared)-2*qy*(qw_qz_qx_qy))) / (wz*((qw_qz_qx_qy)**2 + (q_x_squared)**2)**(3/2))\
                    + (vf*( (2*qx*(qw_qz_qx_qy_minus))/((qw_qz_qx_qy)**2 + (q_x_squared)**2) + (2*qy*(q_x_squared))/((qw_qz_qx_qy)**2 + (q_x_squared)**2) )*np.sin(dt*wz + np.arctan2(qw_qz_qx_qy, q_x_squared))) / (wz)
        F[1, 8] = - (2*qy*vf) / (wz*np.sqrt((qw_qz_qx_qy)**2 + (q_x_squared)**2))\
                    + (vf*(-2*qx*(qw_qz_qx_qy)+2*qy*(q_x_squared))*(q_x_squared)) / (wz*((qw_qz_qx_qy)**2 + (q_x_squared)**2)**(3/2))\
                    + (vf*( (2*qx*(q_x_squared))/((qw_qz_qx_qy)**2+(q_x_squared)**2) - (2*qy*(qw_qz_qx_qy_minus))/((qw_qz_qx_qy)**2 +(q_x_squared)**2) )*np.sin(dt*wz + np.arctan2(qw_qz_qx_qy, q_x_squared))) / (wz)
        F[1, 9] = - (2*qz*vf) / (wz*np.sqrt((qw_qz_qx_qy)**2 + (q_x_squared)**2))\
                    + (vf*(-2*qw*(qw_qz_qx_qy) +2*qz*(q_x_squared))*(q_x_squared)) / (wz*((qw_qz_qx_qy)**2 +(q_x_squared)**2)**(3/2))\
                    + (vf*((2*qw*(q_x_squared))/((qw_qz_qx_qy)**2+(q_x_squared)**2) - (2*qz*(qw_qz_qx_qy_minus))/((qw_qz_qx_qy)**2 +(q_x_squared)**2))*np.sin(dt*wz +np.arctan2(qw_qz_qx_qy, q_x_squared))) / (wz)

                    
        F[2, 3] = - vx*qw_qz_qx_qy/(wx*np.sqrt(qw_qx_qy_qz**2+q_z_squared**2)*vf)\
                    + vx*np.sin(dt*wx+np.arctan2(qw_qx_qy_qz, q_z_squared)) / (wx*vf)
                    
        F[2, 4] = - vy*qw_qz_qx_qy/(wx*np.sqrt(qw_qx_qy_qz**2+q_z_squared**2)*vf)\
                    + vy*np.sin(dt*wx+np.arctan2(qw_qx_qy_qz, q_z_squared)) / (wx*vf)
                    
        F[2, 5] = - vz*qw_qz_qx_qy/(wx*np.sqrt(qw_qx_qy_qz**2+q_z_squared**2)*vf)\
                    + vz*np.sin(dt*wx+np.arctan2(qw_qx_qy_qz, q_z_squared)) / (wx*vf)
                    
        F[2, 6] = - (2*qx*vf) / (wx*np.sqrt((qw_qx_qy_qz)**2 + (q_z_squared)**2))\
                    - (vf*(qw_qx_qy_qz)*(-2*qw*(q_z_squared) -2*qx*(qw_qx_qy_qz))) / (wx*((qw_qx_qy_qz)**2 + (q_z_squared)**2)**(3/2))\
                    + (vf*(2*qw*(qw_qx_qy_qz_minus)/((qw_qx_qy_qz)**2+(q_z_squared)**2) + 2*qx*(q_z_squared)/((qw_qx_qy_qz)**2+(q_z_squared)**2)) * np.cos(dt*wx + np.arctan2(qw_qx_qy_qz, q_z_squared))) / (wx)
        F[2, 7] = - (2*qw*vf)/(wx*np.sqrt((qw_qx_qy_qz)**2 + (q_z_squared)**2))\
                    - (vf*(qw_qx_qy_qz)*(-2*qw*(qw_qx_qy_qz)+2*qx*(q_z_squared))) / (wx*((qw_qx_qy_qz)**2 + (q_z_squared)**2)**(3/2))\
                    + (vf*( (2*qw*(q_z_squared))/((qw_qx_qy_qz)**2 + (q_z_squared)**2) - (2*qx*(qw_qx_qy_qz_minus))/((qw_qx_qy_qz)**2+(q_z_squared)**2) ) *np.cos(dt*wx + np.arctan2(qw_qx_qy_qz, q_z_squared))) / (wz)
        F[2, 8] = - (2*qz*vf) / (wx*np.sqrt((2*qw*qx+ 2*qy*qz)**2 + (q_z_squared)**2))\
                    - (vf*(qw_qx_qy_qz)*(2*qy*(q_z_squared) -2*qz*(qw_qx_qy_qz))) / (wx*((qw_qx_qy_qz)**2 + (q_z_squared)**2)**(3/2))\
                    + (vf*(-(2*qy*(qw_qx_qy_qz_minus))/((qw_qx_qy_qz)**2 +(q_z_squared)**2) +(2*qz*(q_z_squared))/((qw_qx_qy_qz)**2 +(q_z_squared)**2))*np.cos(dt*wx + np.arctan2(qw_qx_qy_qz, q_z_squared))) / (wz)
        
        
        F[2, 9] = - (2*qy*vf) / (wx*np.sqrt((qw_qx_qy_qz)**2 + (q_z_squared)**2))\
                    - (vf*(qw_qx_qy_qz)*(-2*qy*(qw_qx_qy_qz) -2*qz*(q_z_squared))) / (wx*((qw_qx_qy_qz)**2 +(q_z_squared)**2)**(3/2))\
                    + (vf*((2*qy*(q_z_squared))/((qw_qx_qy_qz)**2+(q_z_squared)**2) + (2*qz*(qw_qx_qy_qz_minus))/((qw_qx_qy_qz)**2 +(q_z_squared)**2))*np.cos(dt*wx +np.arctan2(qw_qx_qy_qz, q_z_squared))) / (wz)
                    
        F[3, 6] = dt*(2*ax*qw - 2*ay*qz + 2*az*qy)
        F[3, 7] = dt*(2*ax*qx + 2*ay*qy + 2*az*qz)
        F[3, 8] = dt*(-2*ax*qy + 2*ay*qx + 2*az*qw)
        F[3, 9] = dt*(-2*ax*qz - 2*ay*qw + 2*az*qx)
        
        F[4, 6] = dt*(2*ax*qz + 2*ay*qw - 2*az*qx)
        F[4, 7] = dt*(2*ax*qy - 2*ay*qx - 2*az*qw)
        F[4, 8] = dt*(2*ax*qx + 2*ay*qy + 2*az*qz)
        F[4, 9] = dt*(2*ax*qw - 2*ay*qz + 2*az*qy)
        
        F[5, 6] = dt*(-2*ax*qy + 2*ay*qx + 2*az*qw)
        F[5, 7] = dt*(2*ax*qz + 2*ay*qw - 2*az*qx)
        F[5, 8] = dt*(-2*ax*qw + 2*ay*qz - 2*az*qy)
        F[5, 9] = dt*(2*ax*qx + 2*ay*qy + 2*az*qz)
        
        F[6, 6] = np.cos(dt*norm_w/2)
        F[6, 7] = wz*np.sin(dt*norm_w/2) / norm_w
        F[6, 8] = -wy*np.sin(dt*norm_w/2) / norm_w
        F[6, 9] = wx*np.sin(dt*norm_w/2) / norm_w
        
        F[7, 6] = -wz*np.sin(dt*norm_w/2) / norm_w
        F[7, 7] = np.cos(dt*norm_w/2)
        F[7, 8] = wx*np.sin(dt*norm_w/2) / norm_w
        F[7, 9] = wy*np.sin(dt*norm_w/2) / norm_w
        
        F[8, 6] = wy*np.sin(dt*norm_w/2) / norm_w
        F[8, 7] = -wx*np.sin(dt*norm_w/2) / norm_w
        F[8, 8] = np.cos(dt*norm_w/2)
        F[8, 9] = wz*np.sin(dt*norm_w/2) / norm_w
        
        F[9, 6] = -wx*np.sin(dt*norm_w/2) / norm_w
        F[9, 7] = -wy*np.sin(dt*norm_w/2) / norm_w
        F[9, 8] = -wz*np.sin(dt*norm_w/2) / norm_w
        F[9, 9] = np.cos(dt*norm_w/2)

        '''
        # Jacobian of state transition function
        G = np.zeros((10, 6))
        
        G[0, 5] = (dt*vf*np.cos(dt*wz+np.arctan2(qw_qz_qx_qy, q_x_squared)))/wz\
                    + (qw_qz_qx_qy*vf)/(wz**2*np.sqrt(qw_qz_qx_qy**2+q_x_squared**2))\
                    - (vf*np.sin(dt*wz+np.arctan2(qw_qz_qx_qy, q_x_squared)))/wz**2
                    
        G[1, 5] = (dt*vf*np.sin(dt*wz+np.arctan2(qw_qz_qx_qy, q_x_squared)))/wz\
                    + (vf*np.cos(dt*wz+np.arctan2(qw_qz_qx_qy, q_x_squared)))/wz**2\
                    - (vf*q_x_squared)/(wz**2*np.sqrt(qw_qz_qx_qy**2+q_x_squared**2))
                    
        G[2, 3] = (dt*vf*np.cos(dt*wx+np.arctan2(qw_qx_qy_qz, q_z_squared)))/wx\
                    + (qw_qx_qy_qz*vf)/(wx**2*np.sqrt(qw_qx_qy_qz**2+q_z_squared**2))\
                    - (vf*np.sin(dt*wx+np.arctan2(qw_qx_qy_qz, q_z_squared)))/wx**2
        
        G[3, 0] = dt*q_x_squared
        G[3, 1] = dt*(-2*qw*qz+2*qx*qy)
        G[3, 2] = dt*(2*qw*qy+2*qx*qz)
        
        G[4, 0] = dt*qw_qz_qx_qy
        G[4, 1] = dt*(qw**2-qx**2+qy**2-qz**2)
        G[4, 2] = dt*(-2*qw*qx+2*qy*qz)
        
        G[5, 0] = dt*(-2*qw*qy+2*qx*qz)
        G[5, 1] = dt*(2*qw*qx+2*qy*qz)
        G[5, 2] = dt*(qw**2-qx**2-qy**2+qz**2)
        
        G[6, 3] = qz*np.sin(dt*norm_w/2)/norm_w
        G[6, 4] = -qy*np.sin(dt*norm_w/2)/norm_w
        G[6, 5] = qx*np.sin(dt*norm_w/2)/norm_w
        
        G[7, 3] = qy*np.sin(dt*norm_w/2)/norm_w
        G[7, 4] = qz*np.sin(dt*norm_w/2)/norm_w
        G[7, 5] = -qw*np.sin(dt*norm_w/2)/norm_w
        
        G[8, 3] = -qx*np.sin(dt*norm_w/2)/norm_w
        G[8, 4] = qw*np.sin(dt*norm_w/2)/norm_w
        G[8, 5] = qz*np.sin(dt*norm_w/2)/norm_w
        
        G[9, 3] = -qw*np.sin(dt*norm_w/2)/norm_w
        G[9, 4] = -qx*np.sin(dt*norm_w/2)/norm_w
        G[9, 5] = -qy*np.sin(dt*norm_w/2)/norm_w
        '''
        self.P = F @ self.P @ F.T + Q # G @ Q @ G.T

    def measurement_update(self, data: MeasurementUpdateField):
        z = data.z
        R = data.R
        sensor_type = data.sensor_type
        z_dim = z.shape[0]
        x = self.x.get_state_vector()
        H = self.get_transition_matrix(sensor_type, z_dim=z_dim)
        mask = self.get_innovation_mask(sensor_type=sensor_type, z_dim=z_dim).reshape(-1, 1)
        # compute Kalman gain
        K = self.P @ H.T @ np.linalg.inv(H @ self.P @ H.T + R)
        # update state x
        z_ = np.dot(H, x)  # expected observation from the estimated state 
        residuals = z - z_
        innovation = K @ residuals
        innovation *= mask
        x = x + innovation

        self.x = State.get_new_state_from_array(x)
        # update covariance P
        self.P = self.P - K @ H @ self.P

        # self.innovations.append(np.sum(residuals))
        self.innovations.append(residuals[:3])
