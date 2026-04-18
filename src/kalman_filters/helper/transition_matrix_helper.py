import numpy as np
import sympy as sp
from typing import Union
from collections import namedtuple
from sympy import symbols, Matrix, eye, sqrt, diag

from ...common.config import FilterConfig
from ...common.datatypes import MeasurementUpdateField, MotionModel, FusionData, State, SensorType

PartialDerivativeIMUBasedModelArgs = namedtuple("PartialDerivativeIMUBasedModelArgs", (
    "px", "py", "pz", "vx", "vy", "vz", "qw", "qx", "qy", "qz", "b_ax", "b_ay", "b_az", "b_wx", "b_wy", "b_wz",
    "a_x", "a_y", "a_z", "w_x", "w_y", "w_z",
    "dt",
))
PartialDerivativeThrustModelArgs = namedtuple("PartialDerivativeThrustModelArgs", (
    "px", "py", "pz", "vx", "vy", "vz", "qw", "qx", "qy", "qz", "b_ax", "b_ay", "b_az", "b_wx", "b_wy", "b_wz",
    "x_wx", "x_wy", "x_wz", "omega1", "omega2", "omega3", "omega4", "m", "km", "kf_1", "kf_2", "kf_3", "kf_4", "kt_x", "kt_y", "kt_z", "Ix", "Iy", "Iz", "l", "Ir", "dt",
))

class TransitionMatrixHelper:

    def __init__(self, filter_config: FilterConfig):

        self.filter_config = filter_config
        self.motion_model = MotionModel.get_motion_model(self.filter_config.motion_model)
        self._F_func, self._G_func = self._get_jacobians(self.motion_model)

        self._H_position_func = None
        self._H_velocity_func = None
        self._H_quaternion_func = None
        self._H_yaw_func = None

        self._set_correction_jacobians()

        
    @property
    def is_imu_based(self):
        return self.motion_model in [MotionModel.KINEMATICS, MotionModel.VELOCITY]
        
    def _get_jacobians(self, motion_model: MotionModel):
        if motion_model == MotionModel.KINEMATICS:
            return self._basics_of_kinematics()
        if motion_model == MotionModel.VELOCITY:
            return self._velocity_based_motion_model()
        if motion_model == MotionModel.DRONE_KINEMATICS:
            return self._drone_dynamics_motion_model()
        
        raise ValueError(f"Unsupported motion model: {motion_model}")

    def _R_from_quat(self, qw, qx, qy, qz):
        # Quaternion is scalar-first [qw, qx, qy, qz].
        # This is the standard DCM mapping body->world for a unit quaternion.
        return Matrix([
            [qw**2 + qx**2 - qy**2 - qz**2, 2*(qx*qy - qw*qz),           2*(qw*qy + qx*qz)],
            [2*(qx*qy + qw*qz),           qw**2 - qx**2 + qy**2 - qz**2, 2*(qy*qz - qw*qx)],
            [2*(qx*qz - qw*qy),           2*(qw*qx + qy*qz),           qw**2 - qx**2 - qy**2 + qz**2],
        ])
    
    def _Omega_from_omega(self, wx, wy, wz):
        # For qdot = 0.5 * Omega(omega) * q, with q=[qw,qx,qy,qz].
        return Matrix([
            [0,   -wx, -wy, -wz],
            [wx,   0,   wz, -wy],
            [wy,  -wz,  0,   wx],
            [wz,   wy, -wx,  0 ],
        ])
    
    def _forward_velocity(self, vx, vy, vz):
        return sqrt(vx**2 + vy**2 + vz**2) + 1e-6 # add small term to avoid division by zero in Jacobian
    
    def _get_angle_from_quat(self, qw, qx, qy, qz):
        # Extract yaw (psi), pitch (theta), roll (phi) from quaternion.
        # This is the standard mapping for scalar-first quaternions.
        psi = sp.atan2(2*(qw*qz + qx*qy), 1 - 2*(qy**2 + qz**2))  # yaw
        theta = sp.asin(2*(qw*qy - qz*qx))  # pitch
        phi = sp.atan2(2*(qw*qx + qy*qz), 1 - 2*(qx**2 + qy**2))  # roll
        return psi, theta, phi
    
    def _get_state_symbols(self):
        # state vector (16)
        px, py, pz = symbols('p_x p_y p_z')
        vx, vy, vz = symbols('v_x v_y v_z')
        qw, qx, qy, qz = symbols('q_w q_x q_y q_z')
        b_ax, b_ay, b_az = symbols('b_a_x b_a_y b_a_z')
        b_wx, b_wy, b_wz = symbols('b_w_x b_w_y b_w_z')

        return (px, py, pz, vx, vy, vz, qw, qx, qy, qz, b_ax, b_ay, b_az, b_wx, b_wy, b_wz)

    def _get_state_symbols_thrust_model(self):
        # state vector (16)
        px, py, pz, vx, vy, vz, qw, qx, qy, qz, b_ax, b_ay, b_az, b_wx, b_wy, b_wz = self._get_state_symbols()
        x_wx, x_wy, x_wz = symbols('x_w_x x_w_y x_w_z')


        return (px, py, pz, vx, vy, vz, qw, qx, qy, qz, b_ax, b_ay, b_az, b_wx, b_wy, b_wz, x_wx, x_wy, x_wz)
    
    def _get_control_symbols(self):
        # control input (measured IMU)
        ax, ay, az = symbols('a_x a_y a_z')
        wx, wy, wz = symbols('w_x w_y w_z')

        return (ax, ay, az, wx, wy, wz)
    
    def _get_control_symbols_thrust_model(self):
        # control input (rpm of 4 rotors)
        omega1, omega2, omega3, omega4 = sp.symbols('omega_1 omega_2 omega_3 omega_4') # angular velocities of the 4 rotors

        return (omega1, omega2, omega3, omega4)

    def _basics_of_kinematics(self):     
        # state vector (16)
        px, py, pz, vx, vy, vz, qw, qx, qy, qz, b_ax, b_ay, b_az, b_wx, b_wy, b_wz = self._get_state_symbols()

        # control input (measured IMU)
        ax, ay, az, wx, wy, wz = self._get_control_symbols()

        # delta time
        dt = symbols('dt', positive=True)

        # Process noise (12): accel noise, gyro noise, accel bias RW, gyro bias RW
        n_ax, n_ay, n_az = symbols('n_a_x n_a_y n_a_z')
        n_wx, n_wy, n_wz = symbols('n_w_x n_w_y n_w_z')
        n_bax, n_bay, n_baz = symbols('n_ba_x n_ba_y n_ba_z')
        n_bwx, n_bwy, n_bwz = symbols('n_bw_x n_bw_y n_bw_z')

        p = Matrix([px, py, pz])
        v = Matrix([vx, vy, vz])
        q = Matrix([qw, qx, qy, qz])
        b_a = Matrix([b_ax, b_ay, b_az])
        b_w = Matrix([b_wx, b_wy, b_wz])

        state = Matrix([px, py, pz, vx, vy, vz, qw, qx, qy, qz, b_ax, b_ay, b_az, b_wx, b_wy, b_wz])
        control_input = Matrix([ax, ay, az, wx, wy, wz])

        # -----------------------
        # Propagation ingredients
        # -----------------------
        R_q = self._R_from_quat(qw, qx, qy, qz)

        # Unbiased / noisy IMU signals used by the propagator
        a_m = Matrix([ax, ay, az])
        w_m = Matrix([wx, wy, wz])

        a_used = (a_m + Matrix([n_ax, n_ay, n_az])) - b_a
        w_used = (w_m + Matrix([n_wx, n_wy, n_wz])) - b_w

        g = Matrix([0, 0, -9.81])

        a_world = R_q * a_used + g

        p_new = p + v*dt + sp.Rational(1, 2) * a_world * dt**2
        v_new = v + a_world * dt

        Omega_used = self._Omega_from_omega(w_used[0], w_used[1], w_used[2])
        q_new = (eye(4) + sp.Rational(1, 2) * dt * Omega_used) * q

        # Bias random walks
        b_a_new = b_a + Matrix([n_bax, n_bay, n_baz]) * dt
        b_w_new = b_w + Matrix([n_bwx, n_bwy, n_bwz]) * dt

        fx = Matrix.vstack(p_new, v_new, q_new, b_a_new, b_w_new)
        F_jacobian = fx.jacobian(state)
        G_jacobian = fx.jacobian(control_input)


        # Evaluate Jacobians at zero process noise (mean propagation)
        zero_noise = {
            n_ax: 0.0, n_ay: 0.0, n_az: 0.0,
            n_wx: 0.0, n_wy: 0.0, n_wz: 0.0,
            n_bax: 0.0, n_bay: 0.0, n_baz: 0.0,
            n_bwx: 0.0, n_bwy: 0.0, n_bwz: 0.0,
        }

        F_expr = sp.Matrix(F_jacobian).subs(zero_noise)
        G_expr = sp.Matrix(G_jacobian).subs(zero_noise)

        # Argument order: state (16), control (6), dt (1)
        _args = [
            px, py, pz, vx, vy, vz, qw, qx, qy, qz, b_ax, b_ay, b_az, b_wx, b_wy, b_wz,
            ax, ay, az, wx, wy, wz,
            dt,
        ]

        _F_func = sp.lambdify(_args, F_expr, modules="numpy")
        _G_func = sp.lambdify(_args, G_expr, modules="numpy")
        return _F_func, _G_func
    
    def _velocity_based_motion_model(self):
        """Velocity-based motion model"""
        # state vector (16)
        px, py, pz, vx, vy, vz, qw, qx, qy, qz, b_ax, b_ay, b_az, b_wx, b_wy, b_wz = self._get_state_symbols()

        # control input (measured IMU)
        ax, ay, az, wx, wy, wz = self._get_control_symbols()

        # delta time
        dt = symbols('dt', positive=True)

        # Process noise (12): accel noise, gyro noise, accel bias RW, gyro bias RW
        n_ax, n_ay, n_az = symbols('n_a_x n_a_y n_a_z')
        n_wx, n_wy, n_wz = symbols('n_w_x n_w_y n_w_z')
        n_bax, n_bay, n_baz = symbols('n_ba_x n_ba_y n_ba_z')
        n_bwx, n_bwy, n_bwz = symbols('n_bw_x n_bw_y n_bw_z')


        p = Matrix([px, py, pz])
        v = Matrix([vx, vy, vz])
        q = Matrix([qw, qx, qy, qz])
        b_a = Matrix([b_ax, b_ay, b_az])
        b_w = Matrix([b_wx, b_wy, b_wz])

        state = Matrix([px, py, pz, vx, vy, vz, qw, qx, qy, qz, b_ax, b_ay, b_az, b_wx, b_wy, b_wz])
        control_input = Matrix([ax, ay, az, wx, wy, wz])
        
        vf = self._forward_velocity(vx, vy, vz)
        
        psi, theta, phi = self._get_angle_from_quat(qw, qx, qy, qz)

        R_q = self._R_from_quat(qw, qx, qy, qz)

        # Unbiased / noisy IMU signals used by the propagator
        a_m = Matrix([ax, ay, az])
        w_m = Matrix([wx, wy, wz])

        a_used = (a_m + Matrix([n_ax, n_ay, n_az])) - b_a
        w_used = (w_m + Matrix([n_wx, n_wy, n_wz])) - b_w

        g = Matrix([0, 0, -9.81])

        Omega = self._Omega_from_omega(w_used[0], w_used[1], w_used[2])
        q_new = (eye(4) + sp.Rational(1, 2) * dt * Omega) * q

        a_world = R_q * a_used + g
        v_new = v + a_world * dt

        rx = vf / wx  # turning radius for x axis
        rz = vf / wz  # turning radius for z axis

        dphi = wx * dt
        dpsi = wz * dt
        dpx = - rz * sp.sin(psi) + rz * sp.sin(psi + dpsi)
        dpy = + rz * sp.cos(psi) - rz * sp.cos(psi + dpsi)
        dpz = + rx * sp.cos(phi) - rx * sp.cos(phi + dphi)

        dp = Matrix([dpx, dpy, dpz])
        p_new = p + dp

        # Bias random walks
        b_a_new = b_a + Matrix([n_bax, n_bay, n_baz]) * dt
        b_w_new = b_w + Matrix([n_bwx, n_bwy, n_bwz]) * dt

        fx = Matrix.vstack(p_new, v_new, q_new, b_a_new, b_w_new)
        F_jacobian = fx.jacobian(state)
        G_jacobian = fx.jacobian(control_input)


        # Evaluate Jacobians at zero process noise (mean propagation)
        zero_noise = {
            n_ax: 0.0, n_ay: 0.0, n_az: 0.0,
            n_wx: 0.0, n_wy: 0.0, n_wz: 0.0,
            n_bax: 0.0, n_bay: 0.0, n_baz: 0.0,
            n_bwx: 0.0, n_bwy: 0.0, n_bwz: 0.0,
        }

        F_expr = sp.Matrix(F_jacobian).subs(zero_noise)
        G_expr = sp.Matrix(G_jacobian).subs(zero_noise)

        # Argument order: state (16), control (6), dt (1)
        _args = [
            px, py, pz, vx, vy, vz, qw, qx, qy, qz, b_ax, b_ay, b_az, b_wx, b_wy, b_wz,
            ax, ay, az, wx, wy, wz,
            dt,
        ]

        _F_func = sp.lambdify(_args, F_expr, modules="numpy")
        _G_func = sp.lambdify(_args, G_expr, modules="numpy")

        return _F_func, _G_func
    

    def _drone_dynamics_motion_model(self):
        """Drone-dynamics-based motion model (not implemented yet)"""

        dt = symbols('dt', positive=True) # delta time
        m = symbols('m') # mass of drone
        km = sp.symbols('k_m') # moment coefficient
        kr_x, kr_y, kr_z = sp.symbols('k_r_x k_r_y k_r_z') # drag coefficients for x, y, z axes respectively
        kf_1, kf_2, kf_3, kf_4 = sp.symbols('k_f_1 k_f_2 k_f_3 k_f_4') # thrust coefficients for each rotor
        kt_x, kt_y, kt_z = sp.symbols('k_t_x k_t_y k_t_z') # thrust drag coefficients for x, y, z axes respectively

        Ix, Iy, Iz = sp.symbols('I_x I_y I_z') # moments of inertia around x, y, z axes respectively
        l = sp.symbols('l') # arm length from center of mass to each rotor
        Ir = sp.symbols('I_r') # rotor inertia around its axis of rotation

        px, py, pz, vx, vy, vz, qw, qx, qy, qz, b_ax, b_ay, b_az, b_wx, b_wy, b_wz, x_wx, x_wy, x_wz = self._get_state_symbols_thrust_model()
        omega1, omega2, omega3, omega4 = self._get_control_symbols_thrust_model()

        p = Matrix([px, py, pz])
        v = Matrix([vx, vy, vz])
        b_a = Matrix([b_ax, b_ay, b_az])
        b_w = Matrix([b_wx, b_wy, b_wz])


        drone_state = Matrix([px, py, pz, vx, vy, vz, qw, qx, qy, qz, b_ax, b_ay, b_az, b_wx, b_wy, b_wz, x_wx, x_wy, x_wz])
        drone_control_input = Matrix([omega1, omega2, omega3, omega4])


        omega_r = -omega1 + omega2 - omega3 + omega4  # net rotor speed (for gyroscopic effect)

        u1 = kf_1*omega1**2 + kf_2*omega2**2 + kf_3*omega3**2 + kf_4*omega4**2  # Total thrust (scalar)
        u2 = l * (kf_4*omega4**2 - kf_2*omega2**2)                               # Roll torque
        u3 = l * (-kf_3*omega3**2 + kf_1*omega1**2)                              # Pitch torque
        u4 = km * (omega1**2 - omega2**2 + omega3**2 - omega4**2)                # Yaw torque


        psi, theta, phi = self._get_angle_from_quat(qw, qx, qy, qz)

        # --- Translational dynamics (inertial frame) ---
        ax_world = - 1/m * (kt_x * vx + u1*(sp.sin(phi)*sp.sin(psi) + sp.cos(phi)*sp.cos(psi)*sp.sin(theta)))
        ay_world = - 1/m * (kt_y * vy + u1*(sp.sin(phi)*sp.cos(psi) - sp.sin(psi)*sp.cos(phi)*sp.sin(theta)))
        az_world = - 1/m * (kt_z * vz - m*9.81 + u1*(sp.cos(phi)*sp.cos(theta)))
        a_world = Matrix([ax_world, ay_world, az_world])

        v_new = v + a_world * dt
        p_new = p + v * dt

        # --- Rotational dynamics (body frame) ---
        a1 = (Iy - Iz) / Ix
        a2 = Ir / Ix
        a3 = (Iz - Ix) / Iy
        a4 = Ir / Iy
        a5 = (Ix - Iy) / Iz
        b1 = l / Ix
        b2 = l / Iy
        b3 = l / Iz

        omega_x, omega_y, omega_z = x_wx, x_wy, x_wz

        w_dot_x = b1*u2 - a2*omega_y*omega_r + a1*omega_y*omega_z
        w_dot_y = b2*u3 + a4*omega_x*omega_r + a3*omega_x*omega_z
        w_dot_z = b3*u4 + a5*omega_x*omega_y

        omega_x_new = omega_x + w_dot_x * dt
        omega_y_new = omega_y + w_dot_y * dt
        omega_z_new = omega_z + w_dot_z * dt
        w_new = Matrix([omega_x_new, omega_y_new, omega_z_new])

        # --- Euler rates from body angular velocities ---
        phi_dot   = omega_x + omega_y * sp.sin(phi) * sp.tan(theta) + omega_z * sp.cos(phi) * sp.tan(theta)
        theta_dot = omega_y * sp.cos(phi) - omega_z * sp.sin(phi)
        psi_dot   = omega_y * sp.sin(phi) / sp.cos(theta) + omega_z * sp.cos(phi) / sp.cos(theta)

        euler = Matrix([phi, theta, psi])
        new_euler = euler + Matrix([phi_dot, theta_dot, psi_dot]) * dt

        half_phi   = new_euler[0] / 2
        half_theta = new_euler[1] / 2
        half_psi   = new_euler[2] / 2
        q_new = Matrix([
            sp.cos(half_phi)*sp.cos(half_theta)*sp.cos(half_psi) + sp.sin(half_phi)*sp.sin(half_theta)*sp.sin(half_psi),
            sp.sin(half_phi)*sp.cos(half_theta)*sp.cos(half_psi) - sp.cos(half_phi)*sp.sin(half_theta)*sp.sin(half_psi),
            sp.cos(half_phi)*sp.sin(half_theta)*sp.cos(half_psi) + sp.sin(half_phi)*sp.cos(half_theta)*sp.sin(half_psi),
            sp.cos(half_phi)*sp.cos(half_theta)*sp.sin(half_psi) - sp.sin(half_phi)*sp.sin(half_theta)*sp.cos(half_psi),
        ])

        b_a_new = b_a
        b_w_new = b_w

        fx = Matrix.vstack(p_new, v_new, q_new, b_a_new, b_w_new, w_new)
        F_jacobian = fx.jacobian(drone_state)
        G_jacobian = fx.jacobian(drone_control_input)

        F_expr = sp.Matrix(F_jacobian)
        G_expr = sp.Matrix(G_jacobian)

        # Argument order: drone_state (19), control (4), physical params (14), dt (1)
        _args = [
            px, py, pz, vx, vy, vz, qw, qx, qy, qz, b_ax, b_ay, b_az, b_wx, b_wy, b_wz, x_wx, x_wy, x_wz,
            omega1, omega2, omega3, omega4,
            m, km, kf_1, kf_2, kf_3, kf_4, kt_x, kt_y, kt_z, Ix, Iy, Iz, l, Ir,
            dt,
        ]

        _F_func = sp.lambdify(_args, F_expr, modules="numpy")
        _G_func = sp.lambdify(_args, G_expr, modules="numpy")

        return _F_func, _G_func

    def get_F_jacobian(
            self, 
            list_of_args: Union[PartialDerivativeIMUBasedModelArgs, PartialDerivativeThrustModelArgs]
            ) -> np.ndarray:
        """
        if IMU-based motion model, expects list_of_args to contain:
            x_state: shape (16,) = [p(3), v(3), q(4 scalar-first), b_a(3), b_w(3)]
            u_imu:   shape (6,)  = [a_x, a_y, a_z, w_x, w_y, w_z]
            dt_val: scalar
            list_of_args (23,) = [x_state, u_imu, dt_val]

            Return F = d f / d x as a NumPy (16x16) array.
        
        if thrust-based motion model, expects list_of_args to contain:
            x_state: shape (19,) = [p(3), v(3), q(4 scalar-first), b_a(3), b_w(3), x_w(3)]
            u_thrust: shape (4,) = [T_1, T_2, T_3, T_4]
            drone_params: shape (14,) = [m, km, kf_1, kf_2, kf_3, kf_4, kt_x, kt_y, kt_z, Ix, Iy, Iz, l, Ir]
            dt_val: scalar
            list_of_args (38,) = [x_state, u_thrust, drone_params, dt_val]
            Return F = d f / d x as a NumPy (16x16) array.
        """

        if self.is_imu_based:
            if len(list_of_args) != 23:
                raise ValueError("Invalid number of arguments for IMU-based motion model")
        else: # thrust-based or other non-IMU motion models would go here
            if len(list_of_args) != 38:
                raise ValueError("Invalid number of arguments for thrust-based motion model")
            
        return np.array(self._F_func(*list_of_args), dtype=float)

    def get_G_jacobian(
            self, 
            list_of_args: Union[PartialDerivativeIMUBasedModelArgs, PartialDerivativeThrustModelArgs]
            ) -> np.ndarray:
        """
        if IMU-based motion model, expects list_of_args to contain:
            x_state: shape (16,) = [p(3), v(3), q(4 scalar-first), b_a(3), b_w(3)]
            u_imu:   shape (6,)  = [a_x, a_y, a_z, w_x, w_y, w_z]
            dt_val: scalar
            list_of_args = [x_state, u_imu, dt_val]

            Return G = d f / d w as a NumPy (16x6) array.
        
        if thrust-based motion model, expects list_of_args to contain:
            x_state: shape (19,) = [p(3), v(3), q(4 scalar-first), b_a(3), b_w(3), x_w(3)]
            u_thrust: shape (4,) = [T_1, T_2, T_3, T_4]
            drone_params: shape (14,) = [m, km, kf_1, kf_2, kf_3, kf_4, kt_x, kt_y, kt_z, Ix, Iy, Iz, l, Ir]
            dt_val: scalar
            list_of_args = [x_state, u_thrust, drone_params, dt_val]
            Return G = d f / d w as a NumPy (16x4) array.
        """

        if self.is_imu_based:
            if len(list_of_args) != 23:
                raise ValueError("Invalid number of arguments for IMU-based motion model")

        else: # thrust-based or other non-IMU motion models would go here
            if len(list_of_args) != 38:
                raise ValueError("Invalid number of arguments for thrust-based motion model")
        
        return np.array(self._G_func(*list_of_args), dtype=float)
    
    def _set_correction_jacobians(self):
        px, py, pz = symbols('p_x p_y p_z')
        vx, vy, vz = symbols('v_x v_y v_z')
        qw, qx, qy, qz = symbols('q_w q_x q_y q_z')
        b_ax, b_ay, b_az = symbols('b_a_x b_a_y b_a_z')
        b_wx, b_wy, b_wz = symbols('b_w_x b_w_y b_w_z')

        p = Matrix([px, py, pz])
        v = Matrix([vx, vy, vz])
        q = Matrix([qw, qx, qy, qz])

        state = Matrix([px, py, pz, vx, vy, vz, qw, qx, qy, qz, b_ax, b_ay, b_az, b_wx, b_wy, b_wz])
        
        R_q = self._R_from_quat(qw, qx, qy, qz)
        psi, theta, phi = self._get_angle_from_quat(qw, qx, qy, qz)

        hx_position = p
        hx_velocity: Matrix = R_q.T * v
        hx_quaternion = q
        hx_mag_yaw = Matrix([psi])

        H_jacobian_position = hx_position.jacobian(state)
        H_jacobian_velocity = hx_velocity.jacobian(state)
        H_jacobian_quaternion = hx_quaternion.jacobian(state)
        H_jacobian_yaw = hx_mag_yaw.jacobian(state)

        H_position_expr = sp.Matrix(H_jacobian_position)
        H_velocity_expr = sp.Matrix(H_jacobian_velocity)
        H_quaternion_expr = sp.Matrix(H_jacobian_quaternion)
        H_yaw_expr = sp.Matrix(H_jacobian_yaw)
        

        _args = [
            px, py, pz, vx, vy, vz, qw, qx, qy, qz, b_ax, b_ay, b_az, b_wx, b_wy, b_wz
        ]

        self._H_position_func = sp.lambdify(_args, H_position_expr, modules="numpy")
        self._H_velocity_func = sp.lambdify(_args, H_velocity_expr, modules="numpy")
        self._H_quaternion_func = sp.lambdify(_args, H_quaternion_expr, modules="numpy")
        self._H_yaw_func = sp.lambdify(_args, H_yaw_expr, modules="numpy")


    def get_transition_matrix(self, state: State, data: MeasurementUpdateField) -> np.ndarray:
        
        sensor_type = data.sensor_type

        state_vector = state.get_state_vector().flatten().tolist()
        x_dim = state.get_vector_size()
        H = np.empty((0, x_dim)) # z_dim x 16
        sensor = self.filter_config.sensors.get(sensor_type, {})
        fusion_fields = sensor.get('fields', [])

        match(sensor_type.name):
            case SensorType.KITTI_VO.name | SensorType.EuRoC_VO.name |\
                    SensorType.UAV_VO.name | SensorType.PX4_VO.name:
                if FusionData.POSITION in fusion_fields:
                    p_H = self._H_position_func(*state_vector)
                    H = np.vstack((H, p_H))
                if FusionData.LINEAR_VELOCITY in fusion_fields:
                    v_H = self._H_velocity_func(*state_vector)
                    H = np.vstack((H, v_H))
                if FusionData.ORIENTATION in fusion_fields:
                    q_H = self._H_quaternion_func(*state_vector)
                    H = np.vstack((H, q_H))
            case SensorType.PX4_MAG.name:
                if FusionData.HEADING_ANGLE in fusion_fields:
                    yaw_H = self._H_yaw_func(*state_vector)
                    H = np.vstack((H, yaw_H))
            case SensorType.PX4_CUSTOM_IMU.name:
                if FusionData.ORIENTATION in fusion_fields:
                    q_H = self._H_quaternion_func(*state_vector)
                    H = np.vstack((H, q_H))
            case SensorType.KITTI_UPWARD_LEFTWARD_VELOCITY.name:
                v_H = np.eye(x_dim)[3:6, :] # 3 x 16, directly observe velocity
                H = np.vstack((H, v_H))
            case _:
                p_H = np.eye(x_dim)[:3, :] # all transition matrix for GPS, UWB, and any position update is handled by this. [I_3x3, 0_3x3, 0_3x4, 0_3x3, 0_3x3]
                H = np.vstack((H, p_H))
            
        return H.astype(float)

if __name__ == "__main__":

    motion_model = 'kinematics' # or 'velocity' or 'drone_dynamics'
    jacobian_helper = TransitionMatrixHelper(motion_model)
    x_numpy = np.array([0, 0, 0, 0.1, 0.1, 0.1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)
    u_numpy = np.array([0, 0, 9.81, 0, 0, 0], dtype=float) + 1e-6
    delta_t = 0.1

    P = np.eye(16) * 0.1
    Q = np.diag([0.1]*3 + [0.1]*3)

    vals = PartialDerivativeIMUBasedModelArgs(
        px=x_numpy[0], py=x_numpy[1], pz=x_numpy[2],
        vx=x_numpy[3], vy=x_numpy[4], vz=x_numpy[5],
        qw=x_numpy[6], qx=x_numpy[7], qy=x_numpy[8], qz=x_numpy[9],
        b_ax=x_numpy[10], b_ay=x_numpy[11], b_az=x_numpy[12],
        b_wx=x_numpy[13], b_wy=x_numpy[14], b_wz=x_numpy[15],
        a_x=u_numpy[0], a_y=u_numpy[1], a_z=u_numpy[2],
        w_x=u_numpy[3], w_y=u_numpy[4], w_z=u_numpy[5],
        dt=delta_t,
    )
    F_jacobian_1 = jacobian_helper.get_F_jacobian(vals)
    G_jacobian_1 = jacobian_helper.get_G_jacobian(vals)

    print("Shape of F Jacobian:", F_jacobian_1.shape)
    print("Shape of G Jacobian:", G_jacobian_1.shape)
    P1_new = F_jacobian_1 @ P @ F_jacobian_1.T + G_jacobian_1 @ Q @ G_jacobian_1.T
    print("Shape of new covariance after propagation:", P1_new.shape)


    motion_model = 'velocity' # or 'velocity' or 'drone_dynamics'
    jacobian_helper = TransitionMatrixHelper(motion_model)
    x_numpy = np.array([0, 0, 0, 0.1, 0.1, 0.1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)
    u_numpy = np.array([0, 0, 9.81, 0, 0, 0], dtype=float) + 1e-6
    delta_t = 0.1

    P = np.eye(16) * 0.1
    Q = np.diag([0.1]*3 + [0.1]*3)

    vals = PartialDerivativeIMUBasedModelArgs(
        px=x_numpy[0], py=x_numpy[1], pz=x_numpy[2],
        vx=x_numpy[3], vy=x_numpy[4], vz=x_numpy[5],
        qw=x_numpy[6], qx=x_numpy[7], qy=x_numpy[8], qz=x_numpy[9],
        b_ax=x_numpy[10], b_ay=x_numpy[11], b_az=x_numpy[12],
        b_wx=x_numpy[13], b_wy=x_numpy[14], b_wz=x_numpy[15],
        a_x=u_numpy[0], a_y=u_numpy[1], a_z=u_numpy[2],
        w_x=u_numpy[3], w_y=u_numpy[4], w_z=u_numpy[5],
        dt=delta_t,
    )
    F_jacobian_2 = jacobian_helper.get_F_jacobian(vals)
    G_jacobian_2 = jacobian_helper.get_G_jacobian(vals)

    print("Shape of F Jacobian:", F_jacobian_2.shape)
    print("Shape of G Jacobian:", G_jacobian_2.shape)
    P2_new = F_jacobian_2 @ P @ F_jacobian_2.T + G_jacobian_2 @ Q @ G_jacobian_2.T
    print("Shape of new covariance after propagation:", P2_new.shape)



    motion_model = 'drone_dynamics' # or 'velocity' or 'drone_dynamics'
    jacobian_helper = TransitionMatrixHelper(motion_model)
    x_numpy = np.array([0, 0, 0, 0.1, 0.1, 0.1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0.1, 0.1], dtype=float)
    u_numpy = np.array([500, 500, 500, 500], dtype=float) + 1e-6
    delta_t = 0.1

    P = np.eye(x_numpy.shape[0]) * 0.1
    Q = np.diag([0.1]*4)

    vals = PartialDerivativeThrustModelArgs(
        px=x_numpy[0], py=x_numpy[1], pz=x_numpy[2],
        vx=x_numpy[3], vy=x_numpy[4], vz=x_numpy[5],
        qw=x_numpy[6], qx=x_numpy[7], qy=x_numpy[8], qz=x_numpy[9],
        b_ax=x_numpy[10], b_ay=x_numpy[11], b_az=x_numpy[12],
        b_wx=x_numpy[13], b_wy=x_numpy[14], b_wz=x_numpy[15],
        x_wx=x_numpy[16], x_wy=x_numpy[17], x_wz=x_numpy[18],
        omega1=u_numpy[0], omega2=u_numpy[1], omega3=u_numpy[2], omega4=u_numpy[3],
        m=1.0, km=0.01, kf_1=0.1, kf_2=0.1, kf_3=0.1, kf_4=0.1, kt_x=0.01, kt_y=0.01, kt_z=0.01,
        Ix=0.01, Iy=0.01, Iz=0.02, l=0.1, Ir=0.001,
        dt=delta_t,
    )
    F_jacobian_3 = jacobian_helper.get_F_jacobian(vals)
    G_jacobian_3 = jacobian_helper.get_G_jacobian(vals)

    print("Shape of F Jacobian:", F_jacobian_3.shape)
    print("Shape of G Jacobian:", G_jacobian_3.shape)
    P3_new = F_jacobian_3 @ P @ F_jacobian_3.T + G_jacobian_3 @ Q @ G_jacobian_3.T
    print("Shape of new covariance after propagation:", P3_new.shape)