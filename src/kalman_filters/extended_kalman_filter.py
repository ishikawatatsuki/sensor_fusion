import os
import sys
import numpy as np

from .base_filter import BaseFilter
from ..common import (
    FusionData,
    SensorType,
    State, MeasurementUpdateField,
    MeasurementUpdateField,
    GimbalCondition
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

    def _get_kinematics_jacobian(
            self, 
            u: np.ndarray, 
            q: np.ndarray, 
            dt: float, 
            norm_w: float
        ):
        """Compute Jacobian matrix for the kinematics motion model."""

        ax, ay, az, wx, wy, wz = u
        qw, qx, qy, qz = q.flatten()
        dt2 = dt**2
        cos_w = np.cos(norm_w*dt/2)
        sin_w = np.sin(norm_w*dt/2)/norm_w
        wz_sin = wz*sin_w
        wy_sin = wy*sin_w
        wx_sin = wx*sin_w

        # Jacobian matrix of function f(x,u) with respect to the state variables.
        F = np.eye(self.x.get_vector_size())
        F[0, 3] = dt
        F[0, 6] = dt2*(ax*qw-ay*qz+az*qy)
        F[0, 7] = dt2*(ax*qx+ay*qy+az*qz)
        F[0, 8] = dt2*(-ax*qy+ay*qx+az*qw)
        F[0, 9] = dt2*(-ax*qz-ay*qw+az*qx)

        F[1, 4] = dt
        F[1, 6] = dt2*(ax*qz+ay*qw-az*qx)
        F[1, 7] = dt2*(ax*qy-ay*qx-az*qw)
        F[1, 8] = dt2*(ax*qx+ay*qy+az*qz)
        F[1, 9] = dt2*(ax*qw-ay*qz+az*qy)

        F[2, 5] = dt
        F[2, 6] = dt2*(-ax*qy+ay*qx+az*qw)
        F[2, 7] = dt2*(ax*qz+ay*qw-az*qx)
        F[2, 8] = dt2*(-ax*qw+ay*qz-az*qy)
        F[2, 9] = dt2*(ax*qx+ay*qy+az*qz)

        F[3, 6] = dt*(2*ax*qw-2*ay*qz+2*az*qy)
        F[3, 7] = dt*(2*ax*qx+2*ay*qy+2*az*qz)
        F[3, 8] = dt*(-2*ax*qy+2*ay*qx+2*az*qw)
        F[3, 9] = dt*(-2*ax*qz-2*ay*qw+2*az*qx)

        F[4, 6] = dt*(2*ax*qz+2*ay*qw-2*az*qx)
        F[4, 7] = dt*(2*ax*qy-2*ay*qx-2*az*qw)
        F[4, 8] = dt*(2*ax*qx+2*ay*qy+2*az*qz)
        F[4, 9] = dt*(2*ax*qw-2*ay*qz+2*az*qy)

        F[5, 6] = dt*(-2*ax*qy+2*ay*qx+2*az*qw)
        F[5, 7] = dt*(2*ax*qz+2*ay*qw-2*az*qx)
        F[5, 8] = dt*(-2*ax*qw+2*ay*qz-2*az*qy)
        F[5, 9] = dt*(2*ax*qx+2*ay*qy+2*az*qz)

        F[6, 6] = cos_w
        F[6, 7] = -wx_sin
        F[6, 8] = -wy_sin
        F[6, 9] = -wz_sin

        F[7, 6] = wx_sin
        F[7, 7] = cos_w
        F[7, 8] = wz_sin
        F[7, 9] = -wy_sin

        F[8, 6] = wy_sin
        F[8, 7] = -wz_sin
        F[8, 8] = cos_w
        F[8, 9] = wx_sin

        F[9, 6] = wz_sin
        F[9, 7] = wy_sin
        F[9, 8] = -wx_sin
        F[9, 9] = cos_w
        
        # Jacobian matrix of function f(x,u) with respect to the control input variables.
        qw_2, qx_2, qy_2, qz_2 = q.flatten()**2
        G = np.zeros((self.x.get_state_vector().shape[0], 6))
        G[0, :] = [dt2*(qw_2+qx_2-qy_2-qz_2)/2, dt2*(-qw*qz+qx*qy), dt2*(qw*qy+qx*qz), 0., 0., 0.]
        G[1, :] = [dt2*(qw*qz+qx*qy), dt2*(qw_2-qx_2+qy_2-qz_2)/2, dt2*(-qw*qx+qy*qz), 0., 0., 0.]
        G[2, :] = [dt2*(-qw*qy+qx*qz), dt2*(qw*qx+qy*qz), dt2*(qw_2-qx_2-qy_2+qz_2)/2, 0., 0., 0.]

        G[3, :] = [dt*(qw_2+qx_2-qy_2-qz_2), dt*(-2*qw*qz+2*qx*qy), dt*(2*qw*qy+2*qx*qz), 0., 0., 0.]
        G[4, :] = [dt*(2*qw*qz+2*qx*qy), dt*(qw_2-qx_2+qy_2-qz_2), dt*(-2*qw*qx+2*qy*qz), 0., 0., 0.]
        G[5, :] = [dt*(-2*qw*qy+2*qx*qz), dt*(2*qw*qx+2*qy*qz), dt*(qw_2-qx_2-qy_2+qz_2), 0., 0., 0.]

        G[6, :] = [0., 0., 0., -qx*sin_w, -qy*sin_w, -qz*sin_w]
        G[7, :] = [0., 0., 0., qw*sin_w, -qz*sin_w, qy*sin_w]
        G[8, :] = [0., 0., 0., qz*sin_w, qw*sin_w, -qx*sin_w]
        G[9, :] = [0., 0., 0., -qy*sin_w, qx*sin_w, qw*sin_w]

        return F, G

    def _get_velocity_jacobian_deplicated(
            self, 
            a: np.ndarray,
            w: np.ndarray,
            q: np.ndarray, 
            dt: float, 
            norm_w: float, 
            vf: float, 
        ):


        gimbal_condition = self._gimbal_check(q)

        wx, wy, wz = w
        # propagate covariance P
        qw, qx, qy, qz = q[:, 0]
        ax, ay, az = a[:, 0]
        # Jacobian of state transition function
        # NOTE: the jacobian of transition matrix F is obtained from time_update_jacobian.ipynb
        
        q_x_squared = qw**2+qx**2-qy**2-qz**2
        q_z_squared = qw**2-qx**2-qy**2+qz**2
        qw_qz_qx_qy = 2*qw*qz+2*qx*qy
        qw_qz_qx_qy_minus = -2*qw*qz-2*qx*qy
        qw_qx_qy_qz = 2*qw*qx+2*qy*qz
        qw_qx_qy_qz_minus = -2*qw*qx-2*qy*qz

        F = np.eye(self.x.get_vector_size())
        G = np.zeros((self.x.get_state_vector().shape[0], 6))

        def _normal_condition(F: np.ndarray, G: np.ndarray):

            F[0, 6] = - (2*qz*vf) / (wz*np.sqrt((qw_qz_qx_qy)**2 + (q_x_squared)**2))\
                + (vf*(qw_qz_qx_qy)*(-2*qw*(q_x_squared) -2*qz*(qw_qz_qx_qy))) / (wz*((qw_qz_qx_qy)**2 + (q_x_squared)**2)**(3/2))\
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

            F[1, 6] = + (2*qw*vf) / (wz*np.sqrt((qw_qz_qx_qy)**2 + (q_x_squared)**2))\
                        + (vf*(-2*qw*(q_x_squared)-2*qz*(qw_qz_qx_qy))*(q_x_squared)) / (wz*((qw_qz_qx_qy)**2 + (q_x_squared)**2)**(3/2))\
                        + (vf*(2*qw*(qw_qz_qx_qy_minus)/((qw_qz_qx_qy)**2+(q_x_squared)**2) + 2*qz*(q_x_squared)/((qw_qz_qx_qy)**2+(q_x_squared)**2)) * np.sin(dt*wz + np.arctan2(qw_qz_qx_qy, q_x_squared))) / (wz)
            F[1, 7] = + (2*qx*vf)/(wz*np.sqrt((qw_qz_qx_qy)**2 + (q_x_squared)**2))\
                        + (vf*(-2*qx*(q_x_squared)-2*qy*(qw_qz_qx_qy))*q_x_squared) / (wz*((qw_qz_qx_qy)**2 + (q_x_squared)**2)**(3/2))\
                        + (vf*( (2*qx*(qw_qz_qx_qy_minus))/((qw_qz_qx_qy)**2 + (q_x_squared)**2) + (2*qy*(q_x_squared))/((qw_qz_qx_qy)**2 + (q_x_squared)**2) )*np.sin(dt*wz + np.arctan2(qw_qz_qx_qy, q_x_squared))) / (wz)
            F[1, 8] = - (2*qy*vf) / (wz*np.sqrt((qw_qz_qx_qy)**2 + (q_x_squared)**2))\
                        + (vf*(-2*qx*(qw_qz_qx_qy)+2*qy*(q_x_squared))*(q_x_squared)) / (wz*((qw_qz_qx_qy)**2 + (q_x_squared)**2)**(3/2))\
                        + (vf*( (2*qx*(q_x_squared))/((qw_qz_qx_qy)**2+(q_x_squared)**2) - (2*qy*(qw_qz_qx_qy_minus))/((qw_qz_qx_qy)**2 +(q_x_squared)**2) )*np.sin(dt*wz + np.arctan2(qw_qz_qx_qy, q_x_squared))) / (wz)
            F[1, 9] = - (2*qz*vf) / (wz*np.sqrt((qw_qz_qx_qy)**2 + (q_x_squared)**2))\
                        + (vf*(-2*qw*(qw_qz_qx_qy) +2*qz*(q_x_squared))*(q_x_squared)) / (wz*((qw_qz_qx_qy)**2 +(q_x_squared)**2)**(3/2))\
                        + (vf*((2*qw*(q_x_squared))/((qw_qz_qx_qy)**2+(q_x_squared)**2) - (2*qz*(qw_qz_qx_qy_minus))/((qw_qz_qx_qy)**2 +(q_x_squared)**2))*np.sin(dt*wz +np.arctan2(qw_qz_qx_qy, q_x_squared))) / (wz)

            F[2, 6] = - (2*qw*vf) / (wx*np.sqrt((qw_qx_qy_qz)**2 + (q_z_squared)**2))\
                        + (vf*(-2*qw*(q_z_squared)-2*qx*(qw_qx_qy_qz))*(q_z_squared)) / (wx*((qw_qx_qy_qz)**2 + (q_z_squared)**2)**(3/2))\
                        + (vf*(2*qw*(qw_qx_qy_qz_minus)/((qw_qx_qy_qz)**2+(q_z_squared)**2) + 2*qx*(q_z_squared)/((qw_qx_qy_qz)**2+(q_z_squared)**2)) * np.sin(dt*wx + np.arctan2(qw_qx_qy_qz, q_z_squared))) / (wx)
            F[2, 7] = - (2*qx*vf)/(wx*np.sqrt((qw_qx_qy_qz)**2 + (q_z_squared)**2))\
                        + (vf*(-2*qw*(qw_qx_qy_qz)+2*qx*(q_z_squared))*(q_z_squared)) / (wx*((qw_qx_qy_qz)**2 + (q_z_squared)**2)**(3/2))\
                        + (vf*( (2*qw*(q_z_squared))/((qw_qx_qy_qz)**2 + (q_z_squared)**2) - (2*qx*(qw_qx_qy_qz_minus))/((qw_qx_qy_qz)**2+(q_z_squared)**2) ) *np.sin(dt*wx + np.arctan2(qw_qx_qy_qz, q_z_squared))) / (wx)
            F[2, 8] = - (2*qy*vf) / (wx*np.sqrt((qw_qx_qy_qz)**2 + (q_z_squared)**2))\
                        + (vf*(2*qy*(q_z_squared)-2*qz*(qw_qx_qy_qz))*(q_z_squared)) / (wx*((qw_qx_qy_qz)**2 + (q_z_squared)**2)**(3/2))\
                        + (vf*(-(2*qy*(qw_qx_qy_qz_minus))/((qw_qx_qy_qz)**2 + (q_z_squared)**2) + (2*qz*(q_z_squared))/((qw_qx_qy_qz)**2 +(q_z_squared)**2))*np.sin(dt*wx + np.arctan2(qw_qx_qy_qz, q_z_squared))) / (wx)
            F[2, 9] = + (2*qz*vf) / (wx*np.sqrt((qw_qx_qy_qz)**2 + (q_z_squared)**2))\
                        + (vf*(-2*qy*(qw_qx_qy_qz)-2*qz*(q_z_squared))*(q_z_squared)) / (wx*((qw_qx_qy_qz)**2 + (q_z_squared)**2)**(3/2))\
                        + (vf*((2*qy*(q_z_squared))/((qw_qx_qy_qz)**2+(q_z_squared)**2) + (2*qz*(qw_qx_qy_qz_minus))/((qw_qx_qy_qz)**2 +(q_z_squared)**2))*np.sin(dt*wx +np.arctan2(qw_qx_qy_qz, q_z_squared))) / (wx)

            
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
            F[6, 7] = -wx*np.sin(dt*norm_w/2) / norm_w
            F[6, 8] = -wy*np.sin(dt*norm_w/2) / norm_w
            F[6, 9] = -wz*np.sin(dt*norm_w/2) / norm_w
            
            F[7, 6] = wx*np.sin(dt*norm_w/2) / norm_w
            F[7, 7] = np.cos(dt*norm_w/2)
            F[7, 8] = wz*np.sin(dt*norm_w/2) / norm_w
            F[7, 9] = -wy*np.sin(dt*norm_w/2) / norm_w
            
            F[8, 6] = wy*np.sin(dt*norm_w/2) / norm_w
            F[8, 7] = -wz*np.sin(dt*norm_w/2) / norm_w
            F[8, 8] = np.cos(dt*norm_w/2)
            F[8, 9] = wx*np.sin(dt*norm_w/2) / norm_w
            
            F[9, 6] = wz*np.sin(dt*norm_w/2) / norm_w
            F[9, 7] = wy*np.sin(dt*norm_w/2) / norm_w
            F[9, 8] = -wx*np.sin(dt*norm_w/2) / norm_w
            F[9, 9] = np.cos(dt*norm_w/2)

            # Jacobian of state transition function
            G[0, 5] = (dt*vf*np.cos(dt*wz+np.arctan2(qw_qz_qx_qy, q_x_squared)))/wz\
                        + (vf*qw_qz_qx_qy)/(wz**2*np.sqrt(qw_qz_qx_qy**2+q_x_squared**2))\
                        - (vf*np.sin(dt*wz+np.arctan2(qw_qz_qx_qy, q_x_squared)))/wz**2
            G[1, 5] = (dt*vf*np.sin(dt*wz+np.arctan2(qw_qz_qx_qy, q_x_squared)))/wz\
                        + (vf*np.cos(dt*wz+np.arctan2(qw_qz_qx_qy, q_x_squared)))/wz**2\
                        - (vf*q_x_squared)/(wz**2*np.sqrt(qw_qz_qx_qy**2+q_x_squared**2))
            
            G[2, 3] = (dt*vf*np.sin(dt*wx+np.arctan2(qw_qx_qy_qz, q_z_squared)))/wx\
                        + (vf*np.cos(dt*wx+np.arctan2(qw_qx_qy_qz, q_z_squared))) / wx**2\
                        - (vf*q_z_squared)/(wx**2*np.sqrt(qw_qx_qy_qz**2+q_z_squared**2))

            G[3, 0] = dt*q_x_squared
            G[3, 1] = dt*(-2*qw*qz+2*qx*qy)
            G[3, 2] = dt*(2*qw*qy+2*qx*qz)
            
            G[4, 0] = dt*qw_qz_qx_qy
            G[4, 1] = dt*(qw**2-qx**2+qy**2-qz**2)
            G[4, 2] = dt*(-2*qw*qx+2*qy*qz)
            
            G[5, 0] = dt*(-2*qw*qy+2*qx*qz)
            G[5, 1] = dt*(2*qw*qx+2*qy*qz)
            G[5, 2] = dt*q_z_squared
            
            G[6, 3] = -qx*np.sin(dt*norm_w/2)/norm_w
            G[6, 4] = -qy*np.sin(dt*norm_w/2)/norm_w
            G[6, 5] = -qz*np.sin(dt*norm_w/2)/norm_w
            
            G[7, 3] = qw*np.sin(dt*norm_w/2)/norm_w
            G[7, 4] = -qz*np.sin(dt*norm_w/2)/norm_w
            G[7, 5] = qy*np.sin(dt*norm_w/2)/norm_w
            
            G[8, 3] = qz*np.sin(dt*norm_w/2)/norm_w
            G[8, 4] = qw*np.sin(dt*norm_w/2)/norm_w
            G[8, 5] = -qx*np.sin(dt*norm_w/2)/norm_w
            
            G[9, 3] = -qy*np.sin(dt*norm_w/2)/norm_w
            G[9, 4] = qx*np.sin(dt*norm_w/2)/norm_w
            G[9, 5] = qw*np.sin(dt*norm_w/2)/norm_w

            return F, G
        
        def _up_condition(F: np.ndarray, G: np.ndarray):
            F[0, 6] = (2*qx*vf*np.cos(dt*wz-2*np.arctan2(qx,qw)))/(wz*(qw**2+qx**2))\
                - (2*qx*vf*np.cos(2*np.arctan2(qx,qw)))/(wz*(qw**2+qx**2))
            F[0, 7] = - (2*qw*vf*np.cos(dt*wz -2*np.arctan2(qx, qw))/(wz*(qw**2+qx**2)))\
                + (2*qw*vf*np.cos(2*np.arctan2(qx, qw)))/(wz*(qw**2+qx**2))
            
            F[1, 6] = (2*qx*vf*np.sin(dt*wz-2*np.arctan2(qx,qw)))/(wz*(qw**2+qx**2))\
                + (2*qx*vf*np.sin(2*np.arctan2(qx,qw)))/(wz*(qw**2+qx**2))
            F[1, 7] = - (2*qw*vf*np.sin(dt*wz -2*np.arctan2(qx, qw))/(wz*(qw**2+qx**2)))\
                - (2*qw*vf*np.sin(2*np.arctan2(qx, qw)))/(wz*(qw**2+qx**2))
            
            F[3, 6] = dt*(2*ax*qw - 2*ay*qz + 2*az*qy)
            F[3, 7] = dt*(2*ax*qx + 2*ay*qy + 2*az*qz)
            F[3, 8] = dt*(-2*ax*qy + 2*ay*qx + 2*az*qw)
            F[3, 9] = dt*(-2*ax*qz + 2*ay*qw + 2*az*qx)

            F[4, 6] = dt*(2*ax*qz + 2*ay*qw - 2*az*qx)
            F[4, 7] = dt*(2*ax*qy - 2*ay*qx - 2*az*qw)
            F[4, 8] = dt*(2*ax*qx + 2*ay*qy + 2*az*qz)
            F[4, 9] = dt*(2*ax*qw - 2*ay*qz + 2*az*qy)

            F[5, 6] = dt*(-2*ax*qy + 2*ay*qx + 2*az*qw)
            F[5, 7] = dt*(2*ax*qz + 2*ay*qw - 2*az*qx)
            F[5, 8] = dt*(-2*ax*qw + 2*ay*qz - 2*az*qy)
            F[5, 9] = dt*(2*ax*qx + 2*ay*qy + 2*az*qz)

            F[6, 6] = np.cos(dt*norm_w/2)
            F[6, 7] = - (wx*np.sin(dt*norm_w/2))/norm_w
            F[6, 8] = - (wy*np.sin(dt*norm_w/2))/norm_w
            F[6, 9] = - (wz*np.sin(dt*norm_w/2))/norm_w

            F[7, 6] = (wx*np.sin(dt*norm_w/2))/norm_w
            F[7, 7] = np.cos(dt*norm_w/2)
            F[7, 8] = (wz*np.sin(dt*norm_w/2))/norm_w
            F[7, 9] = - (wy*np.sin(dt*norm_w/2))/norm_w

            F[8, 6] = (wy*np.sin(dt*norm_w/2))/norm_w
            F[8, 7] = - (wz*np.sin(dt*norm_w/2))/norm_w
            F[8, 8] = np.cos(dt*norm_w/2)
            F[8, 9] = (wx*np.sin(dt*norm_w/2))/norm_w

            F[9, 6] = (wz*np.sin(dt*norm_w/2))/norm_w
            F[9, 7] = (wy*np.sin(dt*norm_w/2))/norm_w
            F[9, 8] = - (wx*np.sin(dt*norm_w/2))/norm_w
            F[9, 9] = np.cos(dt*norm_w/2)

            G[0, 5] = (dt*vf*np.cos(dt*wz - 2*np.arctan2(qx, qw)))/wz \
                - (vf*np.sin(dt*wz - 2*np.arctan2(qx, qw)))/wz**2 \
                - (vf*np.sin(2*np.arctan2(qx, qw)))/wz**2
            G[1, 5] = (dt*vf*np.sin(dt*wz - 2*np.arctan2(qx, qw)))/wz \
                + (vf*np.cos(dt*wz - 2*np.arctan2(qx, qw)))/wz**2 \
                - (vf*np.cos(2*np.arctan2(qx, qw)))/wz**2
            G[2, 3] = (dt*vf*np.sin(dt*wx))/wx - vf*np.cos(dt*wx)/wx**2 - vf/wx**2

            G[3, 0] = dt*q_x_squared
            G[3, 1] = dt*(-2*qw*qz+2*qx*qy)
            G[3, 2] = dt*(2*qw*qy+2*qx*qz)

            G[4, 0] = dt*(2*qw*qz+2*qx*qy)
            G[4, 1] = dt*(qw**2-qx**2+qy**2-qz**2)
            G[4, 2] = dt*(-2*qw*qx+2*qy*qz)

            G[5, 0] = dt*(-2*qw*qy+2*qx*qz)
            G[5, 1] = dt*(2*qw*qx+2*qy*qz)
            G[5, 2] = dt*q_z_squared

            G[6, 3] = - (qx*np.sin(dt*norm_w/2))/norm_w
            G[6, 4] = - (qy*np.sin(dt*norm_w/2))/norm_w
            G[6, 5] = - (qz*np.sin(dt*norm_w/2))/norm_w

            G[7, 3] = (qw*np.sin(dt*norm_w/2))/norm_w
            G[7, 4] = - (qz*np.sin(dt*norm_w/2))/norm_w
            G[7, 5] = (qy*np.sin(dt*norm_w/2))/norm_w

            G[8, 3] = (qz*np.sin(dt*norm_w/2))/norm_w
            G[8, 4] = (qw*np.sin(dt*norm_w/2))/norm_w
            G[8, 5] = - (qx*np.sin(dt*norm_w/2))/norm_w

            G[9, 3] = - (qy*np.sin(dt*norm_w/2))/norm_w
            G[9, 4] = (qx*np.sin(dt*norm_w/2))/norm_w
            G[9, 5] = (qw*np.sin(dt*norm_w/2))/norm_w

            return F, G

        def _down_condition(F: np.ndarray, G: np.ndarray):

            F[0, 6] = (2*qx*vf*np.cos(dt*wz -2*np.arctan2(qx, qw)))/(wz*(qw**2+qx**2))\
                - (2*qx*vf*np.cos(2*np.arctan2(qx, qw)))/(wz*(qw**2+qx**2))
            F[0, 7] = - (2*qw*vf*np.cos(dt*wz -2*np.arctan2(qx, qw)))/(wz*(qw**2+qx**2))\
                + (2*qw*vf*np.cos(2*np.arctan2(qx, qw)))/(wz*(qw**2+qx**2))
            
            F[1, 6] = (2*qx*vf*np.sin(dt*wz -2*np.arctan2(qx, qw)))/(wz*(qw**2+qx**2))\
                + (2*qx*vf*np.sin(2*np.arctan2(qx, qw)))/(wz*(qw**2+qx**2))
            F[1, 7] = - (2*qw*vf*np.sin(dt*wz -2*np.arctan2(qx, qw)))/(wz*(qw**2+qx**2))\
                - (2*qw*vf*np.sin(2*np.arctan2(qx, qw)))/(wz*(qw**2+qx**2))
            
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
            F[6, 7] = - (wx*np.sin(dt*norm_w/2))/norm_w
            F[6, 8] = - (wy*np.sin(dt*norm_w/2))/norm_w
            F[6, 9] = - (wz*np.sin(dt*norm_w/2))/norm_w

            F[7, 6] = (wx*np.sin(dt*norm_w/2))/norm_w
            F[7, 7] = np.cos(dt*norm_w/2)
            F[7, 8] = (wz*np.sin(dt*norm_w/2))/norm_w
            F[7, 9] = - (wy*np.sin(dt*norm_w/2))/norm_w

            F[8, 6] = (wy*np.sin(dt*norm_w/2))/norm_w
            F[8, 7] = - (wz*np.sin(dt*norm_w/2))/norm_w
            F[8, 8] = np.cos(dt*norm_w/2)
            F[8, 9] = (wx*np.sin(dt*norm_w/2))/norm_w

            F[9, 6] = (wz*np.sin(dt*norm_w/2))/norm_w
            F[9, 7] = (wy*np.sin(dt*norm_w/2))/norm_w
            F[9, 8] = - (wx*np.sin(dt*norm_w/2))/norm_w
            F[9, 9] = np.cos(dt*norm_w/2)

            
            G[0, 5] = (dt*vf*np.cos(dt*wz - 2*np.arctan2(qx, qw)))/wz \
                - (vf*np.sin(dt*wz - 2*np.arctan2(qx, qw)))/wz**2 \
                - (vf*np.sin(2*np.arctan2(qx, qw)))/wz**2
            
            G[1, 5] = (dt*vf*np.sin(dt*wz - 2*np.arctan2(qx, qw)))/wz \
                + (vf*np.cos(dt*wz - 2*np.arctan2(qx, qw)))/wz**2 \
                - (vf*np.cos(2*np.arctan2(qx, qw)))/wz**2
            
            G[2, 3] = (dt*vf*np.sin(dt*wx))/wx + vf*np.cos(dt*wx)/wx**2 - vf/wx**2
            
            G[3, 0] = dt*(qw**2+qx**2-qy**2-qz**2)
            G[3, 1] = dt*(-2*qw*qz+2*qx*qy)
            G[3, 2] = dt*(2*qw*qy+2*qx*qz)
            
            G[4, 0] = dt*(2*qw*qz+2*qx*qy)
            G[4, 1] = dt*(qw**2-qx**2+qy**2-qz**2)
            G[4, 2] = dt*(-2*qw*qx+2*qy*qz)

            G[5, 0] = dt*(-2*qw*qy+2*qx*qz)
            G[5, 1] = dt*(2*qw*qx+2*qy*qz)
            G[5, 2] = dt*(qw**2-qx**2-qy**2+qz**2)

            G[6, 3] = - (qx*np.sin(dt*norm_w/2))/norm_w
            G[6, 4] = - (qy*np.sin(dt*norm_w/2))/norm_w
            G[6, 5] = - (qz*np.sin(dt*norm_w/2))/norm_w

            G[7, 3] = (qw*np.sin(dt*norm_w/2))/norm_w
            G[7, 4] = - (qz*np.sin(dt*norm_w/2))/norm_w
            G[7, 5] = (qy*np.sin(dt*norm_w/2))/norm_w

            G[8, 3] = (qz*np.sin(dt*norm_w/2))/norm_w
            G[8, 4] = (qw*np.sin(dt*norm_w/2))/norm_w
            G[8, 5] = - (qx*np.sin(dt*norm_w/2))/norm_w

            G[9, 3] = - (qy*np.sin(dt*norm_w/2))/norm_w
            G[9, 4] = (qx*np.sin(dt*norm_w/2))/norm_w
            G[9, 5] = (qw*np.sin(dt*norm_w/2))/norm_w

            return F, G
        
        if gimbal_condition == GimbalCondition.NOSE_UP:
            return _up_condition(F, G)
        elif gimbal_condition == GimbalCondition.NOSE_DOWN:
            return _down_condition(F, G)
        
        return _normal_condition(F, G)

    def _get_velocity_jacobian(self, a: np.ndarray, w: np.ndarray, q: np.ndarray, dt: float, norm_w: float, vf: float, v: np.ndarray):

        # propagate covariance P
        qw, qx, qy, qz = q[:, 0]
        vx, vy, vz = v[:, 0]
        ax, ay, az = a[:, 0]
        wx, wy, wz = w[:, 0]
        # Jacobian of state transition function
        # NOTE: the jacobian of transition matrix F is obtained from setup3_in_3d_coordinate_with_quaternion.ipynb
        
        q_x_squared = qw**2+qx**2-qy**2-qz**2
        q_z_squared = qw**2-qx**2-qy**2+qz**2
        qw_qz_qx_qy = 2*qw*qz+2*qx*qy
        qw_qz_qx_qy_minus = -2*qw*qz-2*qx*qy
        qw_qx_qy_qz = 2*qw*qx+2*qy*qz
        qw_qx_qy_qz_minus = -2*qw*qx-2*qy*qz

        F = np.eye(self.x.get_vector_size())
        G = np.zeros((self.x.get_state_vector().shape[0], 6))
        
        F[0, 3] = - vx*(qw_qz_qx_qy)/(wz*np.sqrt(qw_qz_qx_qy**2+q_x_squared**2)*vf)\
                    + vx*np.sin(dt*wz+np.arctan2(qw_qz_qx_qy, q_x_squared)) / (wz*vf)
        F[0, 4] = - vy*(qw_qz_qx_qy)/(wz*np.sqrt(qw_qz_qx_qy**2+q_x_squared**2)*vf)\
                    + vy*np.sin(dt*wz+np.arctan2(qw_qz_qx_qy, q_x_squared)) / (wz*vf)
        F[0, 5] = - vz*(qw_qz_qx_qy)/(wz*np.sqrt(qw_qz_qx_qy**2+q_x_squared**2)*vf)\
                    + vz*np.sin(dt*wz+np.arctan2(qw_qz_qx_qy, q_x_squared)) / (wz*vf)
        
        F[0, 6] = - (2*qz*vf) / (wz*np.sqrt((qw_qz_qx_qy)**2 + (q_x_squared)**2))\
            + (vf*(qw_qz_qx_qy)*(-2*qw*(q_x_squared) -2*qz*(qw_qz_qx_qy))) / (wz*((qw_qz_qx_qy)**2 + (q_x_squared)**2)**(3/2))\
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
                    + (vf*(-2*qx*(q_x_squared)-2*qy*(qw_qz_qx_qy))*q_x_squared) / (wz*((qw_qz_qx_qy)**2 + (q_x_squared)**2)**(3/2))\
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
                    
        F[2, 6] = - (2*qw*vf) / (wx*np.sqrt((qw_qx_qy_qz)**2 + (q_z_squared)**2))\
                    + (vf*(-2*qw*(q_z_squared)-2*qx*(qw_qx_qy_qz))*(q_z_squared)) / (wx*((qw_qx_qy_qz)**2 + (q_z_squared)**2)**(3/2))\
                    + (vf*(2*qw*(qw_qx_qy_qz_minus)/((qw_qx_qy_qz)**2+(q_z_squared)**2) + 2*qx*(q_z_squared)/((qw_qx_qy_qz)**2+(q_z_squared)**2)) * np.sin(dt*wx + np.arctan2(qw_qx_qy_qz, q_z_squared))) / (wx)
        F[2, 7] = - (2*qx*vf)/(wx*np.sqrt((qw_qx_qy_qz)**2 + (q_z_squared)**2))\
                    + (vf*(-2*qw*(qw_qx_qy_qz)+2*qx*(q_z_squared))*(q_z_squared)) / (wx*((qw_qx_qy_qz)**2 + (q_z_squared)**2)**(3/2))\
                    + (vf*( (2*qw*(q_z_squared))/((qw_qx_qy_qz)**2 + (q_z_squared)**2) - (2*qx*(qw_qx_qy_qz_minus))/((qw_qx_qy_qz)**2+(q_z_squared)**2) ) *np.sin(dt*wx + np.arctan2(qw_qx_qy_qz, q_z_squared))) / (wx)
        F[2, 8] = - (2*qy*vf) / (wx*np.sqrt((qw_qx_qy_qz)**2 + (q_z_squared)**2))\
                    + (vf*(2*qy*(q_z_squared)-2*qz*(qw_qx_qy_qz))*(q_z_squared)) / (wx*((qw_qx_qy_qz)**2 + (q_z_squared)**2)**(3/2))\
                    + (vf*(-(2*qy*(qw_qx_qy_qz_minus))/((qw_qx_qy_qz)**2 + (q_z_squared)**2) + (2*qz*(q_z_squared))/((qw_qx_qy_qz)**2 +(q_z_squared)**2))*np.sin(dt*wx + np.arctan2(qw_qx_qy_qz, q_z_squared))) / (wx)
        F[2, 9] = + (2*qz*vf) / (wx*np.sqrt((qw_qx_qy_qz)**2 + (q_z_squared)**2))\
                    + (vf*(-2*qy*(qw_qx_qy_qz)-2*qz*(q_z_squared))*(q_z_squared)) / (wx*((qw_qx_qy_qz)**2 + (q_z_squared)**2)**(3/2))\
                    + (vf*((2*qy*(q_z_squared))/((qw_qx_qy_qz)**2+(q_z_squared)**2) + (2*qz*(qw_qx_qy_qz_minus))/((qw_qx_qy_qz)**2 +(q_z_squared)**2))*np.sin(dt*wx +np.arctan2(qw_qx_qy_qz, q_z_squared))) / (wx)
                    
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

        # Jacobian of state transition function
        
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

        return F, G
    
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
        
        a_world = (R @ a + self.g)
        p_k = p + v * dt + a_world*dt**2 / 2
        v_k = v + a_world * dt
        q_k = np.array(A + B) @ q
        q_k /= np.linalg.norm(q_k)

        self.x = State(p=p_k, v=v_k, q=q_k, b_w=b_w_k, b_a=b_a_k)

        F, G = self._get_kinematics_jacobian(u, q, dt, norm_w)
        # predict state covariance matrix P
        self.P = F @ self.P @ F.T + Q #+ G @ Q @ G.T

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
        wx, wy, wz = w + 1e-17
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
        
        a_world = (R @ a + self.g)
        v_k = v + a_world * dt
        
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
        
        F, G = self._get_velocity_jacobian(a, w, q_k, dt, norm_w, vf, v)

        self.P = F @ self.P @ F.T + Q #+ G @ Q @ G.T

    def measurement_update(self, data: MeasurementUpdateField):
        z = data.z
        R = data.R
        sensor_type = data.sensor_type
        z_dim = z.shape[0]
        x = self.x.get_state_vector()
        self.H = self.get_transition_matrix(sensor_type, z_dim=z_dim)
        H_j = self._get_measurement_jacobian(data)
        mask = self.get_innovation_mask(sensor_type=sensor_type, z_dim=z_dim).reshape(-1, 1)

        # compute Kalman gain
        self.K = self.P @ H_j.T @ np.linalg.inv(H_j @ self.P @ H_j.T + R)
        # update state x
        z_ = np.dot(self.H, x)  # expected observation from the estimated state
        self.innovation = z - z_
        innovation = self.K @ self.innovation
        innovation *= mask

        x = x + innovation

        self.x = State.get_new_state_from_array(x)

        z_ = np.dot(self.H, x)  # expected observation from the estimated state
        self.residual = z - z_

        # update covariance P
        self.P = self.P - self.K @ H_j @ self.P

        # self.innovations.append(np.sum(residuals))
        # self.innovations.append(self.innovation[:3])

    def _get_velocity_update_jacobian_H(self) -> np.ndarray:
        """Jacobian of the velocity update for KITTI upward and leftward velocity sensor."""
        qw, qx, qy, qz = self.x.q.flatten()
        vx, vy, vz = self.x.v.flatten()

        H = np.zeros((3, self.P.shape[0]))  # 3 x 16
        H[0, 3] = qw**2 + qx**2 - qy**2 - qz**2
        H[0, 4] = 2*qw*qz + 2*qx*qy
        H[0, 5] = -2*qw*qz + 2*qx*qy
        H[0, 6] = 2*qw*vx - 2*qy*vz + 2*qz*vy
        H[0, 7] = 2*qx*vx + 2*qy*vy + 2*qz*vz
        H[0, 8] = -2*qw*vz + 2*qx*vy - 2*qy*vx
        H[0, 9] = 2*qw*vy + 2*qx*vz - 2*qz*vx

        H[1, 3] = -2*qw*qz + 2*qx*qy
        H[1, 4] = qw**2 - qx**2 + qy**2 - qz**2
        H[1, 5] = 2*qw*qx + 2*qy*qz
        H[1, 6] = 2*qw*vy + 2*qx*vz - 2*qz*vx
        H[1, 7] = 2*qw*vz - 2*qx*vy + 2*qy*vx
        H[1, 8] = 2*qx*vx + 2*qy*vy + 2*qz*vz
        H[1, 9] = -2*qw*vx + 2*qy*vz - 2*qz*vy

        H[2, 3] = 2*qw*qy + 2*qx*qz
        H[2, 4] = -2*qw*qx + 2*qy*qz
        H[2, 5] = qw**2 - qx**2 - qy**2 + qz**2
        H[2, 6] = 2*qw*vz - 2*qx*vy + 2*qy*vx
        H[2, 7] = -2*qw*vy - 2*qx*vz + 2*qz*vx
        H[2, 8] = 2*qw*vx - 2*qy*vz + 2*qz*vy
        H[2, 9] = 2*qx*vx + 2*qy*vy + 2*qz*vz

        return H

    def _get_measurement_jacobian(self, data: MeasurementUpdateField) -> np.ndarray:
        sensor_type = data.sensor_type
        fusion_fields = self.config.sensors.get(sensor_type, [])
        match(sensor_type.name):
            case SensorType.KITTI_VO.name | SensorType.EuRoC_VO.name:
                H = np.empty((0, self.P.shape[0])) # z_dim x 16
                if FusionData.POSITION in fusion_fields:
                    # [I_3x3, 0_3x3, 0_3x4, 0_3x3, 0_3x3]
                    H = np.vstack((H, self._get_position_update_H()))
                if FusionData.LINEAR_VELOCITY in fusion_fields:
                    # [0_3x3, 0_3x3, {Hj}_3x4, 0_3x3, 0_3x3]
                    H = np.vstack((H, self._get_velocity_update_jacobian_H()))
                if FusionData.ORIENTATION in fusion_fields:
                    # [0_4x3, 0_4x3, I_4x4, 0_4x3, 0_4x3]
                    H = np.vstack((H, self._get_quaternion_update_H()))
                return H
            
            case SensorType.KITTI_UPWARD_LEFTWARD_VELOCITY.name:
                # [0_2x3, 0_2x3, I_2x4, 0_2x3, 0_2x3]
                return self._get_velocity_update_jacobian_H()[:2, :] 
            case _:
                # NOTE: all transition matrix for GPS, UWB, and any position update is handled by this.
                # [I_3x3, 0_3x3, 0_3x4, 0_3x3, 0_3x3]
                return self._get_position_update_H() 