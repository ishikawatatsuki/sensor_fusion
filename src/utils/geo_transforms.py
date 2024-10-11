# references:
# [1] https://www.enri.go.jp/~fks442/K_MUSEN/1st/1st060428rev2.pdf

import numpy as np
if __name__ == "__main__":
    import sys
    sys.path.append("../../src")
from configs import CoordinateSystemEnum

# constant parameters defined in [1]
_a = 6378137.
_f = 1. / 298.257223563
_b = (1. - _f) * _a
_e = np.sqrt(_a ** 2. - _b ** 2.) / _a
_e_prime = np.sqrt(_a ** 2. - _b ** 2.) / _b


def Rx(theta):
    """rotation matrix around x-axis
    """
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([
        [1, 0, 0],
        [0, c, s],
        [0, -s, c]
    ])


def Ry(theta):
    """rotation matrix around y-axis
    """
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([
        [c, 0, -s],
        [0, 1, 0],
        [s, 0, c]
    ])


def Rz(theta):
    """rotation matrix around z-axis
    """
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([
        [c, s, 0],
        [-s, c, 0],
        [0, 0, 1]
    ])


def lla_to_ecef(points_lla):
    """transform N x [longitude(deg), latitude(deg), altitude(m)] coords into
    N x [x, y, z] coords measured in Earth-Centered-Earth-Fixed frame.
    """
    lon = np.radians(points_lla[0])  # [N,]
    lat = np.radians(points_lla[1])  # [N,]
    alt = points_lla[2]  # [N,]

    N = _a / np.sqrt(1. - (_e * np.sin(lat)) ** 2.)  # [N,]
    x = (N + alt) * np.cos(lat) * np.cos(lon)
    y = (N + alt) * np.cos(lat) * np.sin(lon)
    z = (N * (1. - _e ** 2.) + alt) * np.sin(lat)

    points_ecef = np.stack([x, y, z], axis=0)  # [3, N]
    return points_ecef


def ecef_to_enu(points_ecef, ref_lla):
    """transform N x [x, y, z] coords measured in Earth-Centered-Earth-Fixed frame into
    N x [x, y, z] coords measured in a local East-North-Up frame.
    """
    lon = np.radians(ref_lla[0])
    lat = np.radians(ref_lla[1])

    ref_ecef = lla_to_ecef(ref_lla)  # [3,]

    relative = points_ecef - ref_ecef[:, np.newaxis]  # [3, N]

    # R = Rz(np.pi / 2.0) @ Ry(np.pi / 2.0 - lat) @ Rz(lon)  # [3, 3]
    R = np.array([
        [-np.sin(lon), np.cos(lon), 0],
        [-np.sin(lat)*np.cos(lon), -np.sin(lat)*np.sin(lon), np.cos(lat)],
        [np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)]
    ])
    points_enu = R @ relative  # [3, N]
    return points_enu

def ecef_to_ned(points_ecef, ref_lla):
    """transform N x [x, y, z] coords measured in Earth-Centered-Earth-Fixed frame into
    N x [x, y, z] coords measured in a local North-East-Down frame.
    """
    lon = np.radians(ref_lla[0])
    lat = np.radians(ref_lla[1])

    ref_ecef = lla_to_ecef(ref_lla)  # [3,]

    relative = points_ecef - ref_ecef[:, np.newaxis]  # [3, N]

    # R = Rz(np.pi / 2.0) @ Ry(np.pi / 2.0 - lat) @ Rz(lon)  # [3, 3]
    R = np.array([
        [-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)],
        [-np.sin(lon), np.cos(lon), 0],
        [-np.cos(lat) * np.cos(lon), -np.cos(lat) * np.sin(lon), -np.sin(lat)]
    ])
    points_ned = R @ relative  # [3, N]
    return points_ned


def lla_to_enu(points_lla, ref_lla):
    """transform N x [longitude(deg), latitude(deg), altitude(m)] coords into
    N x [x, y, z] coords measured in a local East-North-Up frame.
    """
    points_ecef = lla_to_ecef(points_lla)
    points_enu = ecef_to_enu(points_ecef, ref_lla)
    return points_enu

def lla_to_ned(points_lla, ref_lla):
    """transform N x [longitude(deg), latitude(deg), altitude(m)] coords into
    N x [x, y, z] coords measured in a local North-East-Down frame.
    """
    points_ecef = lla_to_ecef(points_lla)
    points_ned = ecef_to_ned(points_ecef, ref_lla)
    return points_ned

def enu_to_ecef(points_enu, ref_lla):
    """transform N x [x, y, z] coords measured in a local East-North-Up frame into
    N x [x, y, z] coords measured in Earth-Centered-Earth-Fixed frame.
    """
    # inverse transformation of `ecef_to_enu`

    lon = np.radians(ref_lla[0])
    lat = np.radians(ref_lla[1])
    alt = ref_lla[2]

    ref_ecef = lla_to_ecef(ref_lla)  # [3,]

    R = Rz(np.pi / 2.0) @ Ry(np.pi / 2.0 - lat) @ Rz(lon)  # [3, 3]
    R = R.T  # inverse rotation
    relative = R @ points_enu  # [3, N]

    points_ecef = ref_ecef[:, np.newaxis] + relative  # [3, N]
    return points_ecef


def ecef_to_lla(points_ecef):
    """transform N x [x, y, z] coords measured in Earth-Centered-Earth-Fixed frame into
    N x [longitude(deg), latitude(deg), altitude(m)] coords.
    """
    # approximate inverse transformation of `lla_to_ecef`
    
    x = points_ecef[0]  # [N,]
    y = points_ecef[1]  # [N,]
    z = points_ecef[2]  # [N,]

    p = np.sqrt(x ** 2. + y ** 2.)  # [N,]
    theta = np.arctan(z * _a / (p * _b))  # [N,]

    lon = np.arctan(y / x)  # [N,]
    lat = np.arctan(
        (z + (_e_prime ** 2.) * _b * (np.sin(theta) ** 3.)) / \
        (p - (_e ** 2.) * _a * (np.cos(theta)) ** 3.)
    )  # [N,]
    N = _a / np.sqrt(1. - (_e * np.sin(lat)) ** 2.)  # [N,]
    alt = p / np.cos(lat) - N  # [N,]

    lon = np.degrees(lon)
    lat = np.degrees(lat)

    points_lla = np.stack([lon, lat, alt], axis=0)  # [3, N]
    return points_lla


def enu_to_lla(points_enu, ref_lla):
    """transform N x [x, y, z] coords measured in a local East-North-Up frame into
    N x [longitude(deg), latitude(deg), altitude(m)] coords.
    """
    points_ecef = enu_to_ecef(points_enu, ref_lla)
    points_lla = ecef_to_lla(points_ecef)
    return points_lla

def get_rigid_transformation(calib_path):
    with open(calib_path, 'r') as f:
        calib = f.readlines()

    R = np.array([float(x) for x in calib[1].strip().split(' ')[1:]]).reshape((3, 3))
    t = np.array([float(x) for x in calib[2].strip().split(' ')[1:]])[:, None]

    T = np.vstack((np.hstack((R, t)), np.array([0, 0, 0, 1])))
    
    return T

# this is the method that pykitti computes the translation vectors using Mercator projection (https://sciencedemonstrations.fas.harvard.edu/presentations/mercator-projection)
def kitti_lla_to_enu(lon, lat, alt, scale):
    er = 6378137.  # earth radius (approx.) in meters
    
    # Use a Mercator projection to get the translation vector
    tx = scale * lon * np.pi * er / 180.
    ty = scale * er * \
        np.log(np.tan((90. + lat) * np.pi / 360.))
    tz = alt
    return np.array([tx, ty, tz])


def get_rotation_matrix(q, simplified=False):
    """
        return rotation matrix obtained from quaternion represented in [w, x, y, z] order.
        
        simplified rotation matrix has a constraint such that the quaternion has to be normalized meaning
        that |q| = 1
        https://ahrs.readthedocs.io/en/latest/filters/ekf.html
        https://www.iri.upc.edu/people/jsola/JoanSola/objectes/notes/kinematics.pdf
    """
    q0, q1, q2, q3 = q[:, 0]
    if simplified:
        return np.array([
            [1 - 2*q2**2 - 2*q3**2, 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)],
            [2*(q1*q2 + q0*q3), 1 - 2*q1**2 - 2*q3**2, 2*(q2*q3 - q0*q1)],
            [2*(q1*q3-q0*q2), 2*(q2*q3 + q0*q1), 1 - 2*q1**2 - 2*q2**2]
        ])
    return np.array([
        [q0**2 + q1**2 - q2**2 - q3**2, 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)],
        [2*(q1*q2 + q0*q3), q0**2 - q1**2 + q2**2 - q3**2, 2*(q2*q3 - q0*q1)],
        [2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), q0**2 - q1**2 - q2**2 + q3**2]
    ])


def get_quaternion_update_matrix(w):
    """
        return quaternion update matrix derived from euler angle.
        The matrix is defined as:
        
        omega = | 0  -w.T  |
                | w  -|w|x |
                
        https://ahrs.readthedocs.io/en/latest/filters/ekf.html
        https://www.iri.upc.edu/people/jsola/JoanSola/objectes/notes/kinematics.pdf
    """
    wx, wy, wz = w[:, 0]
    return np.array([ # w, x, y, z
        [0, -wx, -wy, -wz],
        [wx, 0, wz, -wy],
        [wy, -wz, 0, wx],
        [wz, wy, -wx, 0]
        ])

def compute_norm_w(w):
    """
        return a norm of angle
    """
    return np.sqrt(np.sum(w**2))
    
def quat_mult(p, q):
    """
        return a product of two quaternions
    """
    return np.array([
        p[0] * q[0] - p[1] * q[1] - p[2] * q[2] - p[3] * q[3],
        p[0] * q[1] + p[1] * q[0] + p[2] * q[3] - p[3] * q[2],
        p[0] * q[2] - p[1] * q[3] + p[2] * q[0] + p[3] * q[1],
        p[0] * q[3] + p[1] * q[2] - p[2] * q[1] + p[3] * q[0]
    ])
    
def get_rotation_matrix_from_euler_angle(angles, coord=CoordinateSystemEnum.ENU):
    
    roll, pitch, yaw = angles[:, 0]
    if coord == CoordinateSystemEnum.ENU:
        return Rz(yaw) @ Ry(pitch) @ Rx(roll)
    elif coord == CoordinateSystemEnum.NED:
        return Rx(roll) @ Ry(pitch) @ Rz(yaw)
    
    return np.eye(3)