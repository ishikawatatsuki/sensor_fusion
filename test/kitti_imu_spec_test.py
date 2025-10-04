import math

# Given RT3000 specs (matches trade-in/comparison datasheet):
# Gyro ARW = 0.2 deg/√hr, Accel VRW = 0.005 m/s/√hr
ARW_deg_per_sqrt_hr = 0.2
VRW_mps_per_sqrt_hr = 0.005
fs = 100.0  # Hz

# Convert ARW -> gyro noise density in (deg/s)/√Hz, then to rad/s/√Hz
gyro_nd_deg = ARW_deg_per_sqrt_hr / math.sqrt(3600.0)
gyro_nd_rad = gyro_nd_deg * math.pi / 180.0  # rad/s/√Hz

# Convert VRW -> accel noise density in m/s^2/√Hz
accel_nd = VRW_mps_per_sqrt_hr / math.sqrt(3600.0)  # m/s^2/√Hz

# Per-sample standard deviations at fs
gyro_sigma_sample = gyro_nd_rad * math.sqrt(fs)     # rad/s per sample
accel_sigma_sample = accel_nd * math.sqrt(fs)       # m/s^2 per sample

print({
    "gyroscope_noise_density_rad_s_sqrtHz": gyro_nd_rad,
    "accelerometer_noise_density_m_s2_sqrtHz": accel_nd,
    "gyroscope_sigma_per_sample_rad_s": gyro_sigma_sample,
    "accelerometer_sigma_per_sample_m_s2": accel_sigma_sample
})

# If you later fit the +1/2-slope Allan coefficients (RRW) K_g, K_a:
# K_g in rad/s/√Hz, K_a in m/s^2/√Hz
def bias_random_walk_from_rrw(K_g=None, K_a=None):
    out = {}
    if K_g is not None:
        out["gyroscope_random_walk_rad_s2_sqrtHz"] = math.sqrt(3.0) * K_g
    if K_a is not None:
        out["accelerometer_random_walk_m_s3_sqrtHz"] = math.sqrt(3.0) * K_a
    return out