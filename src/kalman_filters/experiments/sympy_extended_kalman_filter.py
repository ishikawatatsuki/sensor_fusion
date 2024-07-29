import sys
if __name__ == "__main__":
    sys.path.append('../../src')
import numpy as np
import matplotlib.pyplot as plt
import sympy
from sympy import symbols, Matrix

class ExtendedKalmanFilter1_2_sympy:
    """Extended Kalman Filter
    for vehicle whose motion is modeled as eq. (5.9) in [1]
    and with observation of its 2d location (x, y)
    """
    x = None
    P = None
    R = None
    px = symbols('p_{x}')
    py = symbols('p_{y}')
    pz = symbols('p_{z}')
    
    vx = symbols('v_{x}')
    vy = symbols('v_{y}')
    vz = symbols('v_{z}')
    
    q1 = symbols('q1')
    q2 = symbols('q2')
    q3 = symbols('q3')
    q4 = symbols('q4')
    
    ax = symbols('a_{x}')
    ay = symbols('a_{y}')
    az = symbols('a_{z}')
    wx = symbols('w_{x}')
    wy = symbols('w_{y}')
    wz = symbols('w_{z}')

    dt = symbols('dt')
    norm_w = symbols('\|w\|')
    aR1 = symbols('aR_{1}')
    aR2 = symbols('aR_{2}')
    aR3 = symbols('aR_{3}')
    qk = Matrix([
        [1],[0],[0],[0]
    ])
    g = Matrix([
        [0],[0],[9.81]
    ])

    errors = []
    
    def __init__(self, x, P, H):
        """ 
        Args:
            x (numpy.array): state to estimate: [x_, y_, theta]^T
            P (numpy.array): estimation error covariance
        """
        
        R = Matrix([[self.q1**2 + self.q2**2 - self.q3**2 + self.q4**2, 
                     2*(self.q2*self.q3 - self.q1*self.q4), 
                     2*(self.q1*self.q3 + self.q2*self.q4)],
                    [2*(self.q2*self.q3 + self.q1*self.q4), 
                     self.q1**2 - self.q2**2 + self.q3**2 - self.q4**2, 
                     2*(self.q3*self.q4 - self.q1*self.q2)],
                    [2*(self.q2*self.q4 - self.q1*self.q3), 2*(self.q1*self.q2 + self.q3*self.q4), 
                     self.q1**2 - self.q2**2 - self.q3**2 + self.q4**2]
           ])
        Omega = Matrix([
            [0, self.wz, -self.wy, self.wx],
            [-self.wz, 0, self.wx, self.wy],
            [self.wy, -self.wx, 0, self.wz],
            [-self.wx, -self.wy, -self.wz, 0]
        ])
        # Omega = Matrix([
        #     [0, -self.wx, -self.wy, -self.wz],
        #     [self.wz, 0, self.wx, -self.wy],
        #     [self.wy, -self.wz, 0, self.wx],
        #     [self.wz, self.wy, -self.wx, 0]
        # ])
        A = sympy.cos(self.norm_w*self.dt/2) * sympy.eye(4)
        B = (1/self.norm_w)*sympy.sin(self.norm_w*self.dt/2) * Omega
        # A = sympy.eye(4)
        # B = self.dt / 2 * Omega
        # self.fxu = Matrix([
        #     [self.px + self.vx*self.dt + 1/2 * (R[0,0]*self.ax-self.g[0] + R[0,1]*self.ay-self.g[1] + R[0,2]*self.az-self.g[2])*self.dt**2],
        #     [self.py + self.vy*self.dt + 1/2 * (R[1,0]*self.ax-self.g[0] + R[1,1]*self.ay-self.g[1] + R[1,2]*self.az-self.g[2])*self.dt**2],
        #     [self.pz + self.vz*self.dt + 1/2 * (R[2,0]*self.ax-self.g[0] + R[2,1]*self.ay-self.g[1] + R[2,2]*self.az-self.g[2])*self.dt**2],
        #     [self.vx + (R[0,0]*self.ax-self.g[0] + R[0,1]*self.ay-self.g[1] + R[0,2]*self.az-self.g[2])*self.dt],
        #     [self.vy + (R[1,0]*self.ax-self.g[0] + R[1,1]*self.ay-self.g[1] + R[1,2]*self.az-self.g[2])*self.dt],
        #     [self.vz + (R[2,0]*self.ax-self.g[0] + R[2,1]*self.ay-self.g[1] + R[2,2]*self.az-self.g[2])*self.dt],
        #     [(A + B) * self.qk],
        # ])
        self.fxu = Matrix([
            [Matrix([[self.px],[self.py],[self.pz]]) +  Matrix([[self.vx],[self.vy],[self.vz]]) * self.dt + (R * Matrix([[self.ax],[self.ay], [self.az]]) - self.g)*self.dt**2 / 2],
            [Matrix([[self.vx],[self.vy],[self.vz]]) + (R * Matrix([[self.ax],[self.ay],[self.az]]) - self.g) * self.dt],
            [(A + B) * Matrix([[self.q1],[self.q2],[self.q3],[self.q4]])],
        ])
        state_x = Matrix([self.px, self.py, self.pz, self.vx, self.vy, self.vz, self.q1, self.q2, self.q3, self.q4])
        control_input = Matrix([self.ax, self.ay, self.az, self.wx, self.wy, self.wz])

        self.F = self.fxu.jacobian(state_x)
        self.G = self.fxu.jacobian(control_input)
        self.P = P
        self.H = H
        self.x = x

    def compute_norm_w(self, wx_, wy_, wz_):
        return np.sqrt(wx_**2 + wy_**2 + wz_**2)
        
    def predict(self, u, dt, Q):
        """estimate x and P based on previous stete of x and control input u
        Args:
            u  (numpy.array): control input u
            dt (numpy.array): difference of current time and previous time
            Q  (numpy.array): process noise 
        """
        # propagate state x
        px_, py_, pz_ = self.x[:3, 0]
        vx_, vy_, vz_ = self.x[3:6, 0]
        q1_, q2_, q3_, q4_ = self.x[6:, 0]
        ax_, ay_, az_, wx_, wy_, wz_ = u
        norm_w_ = self.compute_norm_w(wx_, wy_, wz_);
        fxu_values = {
            self.dt: dt,
            self.px: px_,
            self.py: py_,
            self.pz: pz_,
            self.vx: vx_,
            self.vy: vy_,
            self.vz: vz_,
            self.q1: q1_, 
            self.q2: q2_, 
            self.q3: q3_, 
            self.q4: q4_,
            self.ax: ax_,
            self.ay: ay_,
            self.az: az_,
            self.wx: wx_,
            self.wy: wy_, 
            self.wz: wz_,
            self.norm_w: norm_w_
        }
        # predict state vector x
        self.x = np.array(self.fxu.evalf(subs=fxu_values)).astype(float)

        # predict state covariance matrix P
        P_hat = self.F @ self.P @ self.F.T + self.G @ Matrix(Q) @ self.G.T
        self.P = np.array(P_hat.evalf(subs=fxu_values)).astype(float)
        
    def update(self, z, R):
        """update x and P based on observation of (x_, y_)
        Args:
            z (numpy.array): measurement for [x_, y_]^T
            R (numpy.array): measurement noise covariance
        """
        # compute Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(self.H @ self.P @ self.H.T + R)
        # update state x
        z_ = np.dot(self.H, self.x)  # expected observation from the estimated state
        self.x = self.x + K @ (z - z_)
        
        self.errors.append(np.sqrt(np.sum((z-z_)**2)))

        # update covariance P
        self.P = self.P - K @ self.H @ self.P

    def plot_error(self):
        plt.plot([i for i in range(len(self.errors))], self.errors, label='Error', color='r')


class ExtendedKalmanFilter3_sympy:
    """Extended Kalman Filter
    for vehicle whose motion is modeled as eq. (5.9) in [1]
    and with observation of its 2d location (x, y)
    """
    x = None
    P = None
    R = None
    H = None
    px = symbols('p_{x}')
    py = symbols('p_{y}')
    theta = symbols('theta')
    v = symbols('v')
    w = symbols('w')
    dt = symbols('dt')
        
    errors = []
    
    def __init__(self, x, P, H):
        """ 
        Args:
            x (numpy.array): state to estimate: [x_, y_, theta]^T
            P (numpy.array): estimation error covariance
        """
        
        self.fxu = Matrix([
            [self.px - self.v*sympy.sin(self.theta)/self.w + self.v*sympy.sin(self.theta + self.w*self.dt)/self.w],
            [self.py + self.v*sympy.cos(self.theta)/self.w - self.v*sympy.cos(self.theta + self.w*self.dt)/self.w],
            [self.theta + self.w*self.dt]
        ])
        state_x = Matrix([self.px, self.py, self.theta])
        control_input = Matrix([self.v, self.w])

        self.F = self.fxu.jacobian(state_x)
        self.G = self.fxu.jacobian(control_input)
        self.P = P
        self.H = H
        self.x = x
        
    def predict(self, u, dt, Q):
        """estimate x and P based on previous stete of x and control input u
        Args:
            u  (numpy.array): control input u
            dt (numpy.array): difference of current time and previous time
            Q  (numpy.array): process noise 
        """
        # propagate state x
        px_, py_ = self.x[:2, 0]
        theta_ = self.x[-1, 0]
        v_, w_ = u
        fxu_values = {
            self.dt: dt,
            self.px: px_,
            self.py: py_,
            self.theta: theta_,
            self.v: v_,
            self.w: w_
        }
        # predict state vector x
        self.x = np.array(self.fxu.evalf(subs=fxu_values)).astype(float)

        # predict state covariance matrix P
        P_hat = self.F @ self.P @ self.F.T + self.G @ Matrix(Q) @ self.G.T
        self.P = np.array(P_hat.evalf(subs=fxu_values)).astype(float)
        
    def update(self, z, R):
        """update x and P based on observation of (x_, y_)
        Args:
            z (numpy.array): measurement for [x_, y_]^T
            R (numpy.array): measurement noise covariance
        """
        # compute Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(self.H @ self.P @ self.H.T + R)
        # update state x
        z_ = np.dot(self.H, self.x)  # expected observation from the estimated state
        self.x = self.x + K @ (z - z_)
        self.errors.append(np.sqrt(np.sum((z-z_)**2)))
        
        # update covariance P
        self.P = self.P - K @ self.H @ self.P

    def plot_error(self):
        plt.plot([i for i in range(len(self.errors))], self.errors, label='Error', color='r')
