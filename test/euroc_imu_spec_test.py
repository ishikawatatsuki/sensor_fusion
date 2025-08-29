import math

# ---- Datasheet values (ADIS16448) ----
gyro_nd_deg = 0.0135        # [deg/s]/sqrt(Hz)  (Rate noise density)
gyro_arw_deg = 0.66         # [deg]/sqrt(hr)    (Angular random walk)
accel_nd_mg = 0.23          # [mg]/sqrt(Hz)     (Accel noise density)
accel_vrw = 0.11            # [m/s]/sqrt(hr)    (Velocity random walk)
fs = 200                    # [Hz] default device output rate

# ---- Unit conversions ----
deg2rad = math.pi/180.0
g = 9.80665

gyro_nd = gyro_nd_deg * deg2rad                  # [rad/s]/sqrt(Hz)
accel_nd = (accel_nd_mg * 1e-3) * g              # [m/s^2]/sqrt(Hz)

# Per-sample (discrete-time) std dev from continuous-time density:
gyro_sigma_discrete = gyro_nd * math.sqrt(fs)    # [rad/s] per sample
accel_sigma_discrete = accel_nd * math.sqrt(fs)  # [m/s^2] per sample

# Cross-check ARW <-> noise density (ARW = noise_density * sqrt(3600) in same angle units)
gyro_arw_from_nd_deg = gyro_nd_deg * math.sqrt(3600)  # [deg]/sqrt(hr)
# Should be close to datasheet 0.66 deg/sqrt(hr)
# (Print if you run this: gyro_arw_from_nd_deg ≈ 0.81; datasheet says 0.66 -> typical specs differ, filtering/settings matter.)

# Convert VRW to accel noise density (back-of-envelope): accel_nd ≈ VRW / sqrt(3600)
# But units differ (VRW is m/s/√hr, accel_nd is m/s^2/√Hz), so:
accel_nd_from_vrw = (accel_vrw / math.sqrt(3600.0))    # [m/s]/sqrt(s) = [m/s^2]/sqrt(Hz)
# Compare to datasheet accel_nd; they should be of same order.

# ---- Bias random-walk (you need K from +1/2 slope on Allan) ----
# Example placeholders (PUT YOUR FITTED NUMBERS HERE):
K_g_deg = None  # e.g., 0.002  # [deg/s]/sqrt(Hz) from +1/2 slope fit on gyro Allan
K_a = None      # e.g., 0.0005 # [m/s^2]/sqrt(Hz) from +1/2 slope fit on accel Allan

def bias_rw_from_rrw(K_g_deg=None, K_a=None):
    out = {}
    if K_g_deg is not None:
        K_g = K_g_deg * deg2rad               # [rad/s]/sqrt(Hz)
        gyro_bias_rw = math.sqrt(3.0) * K_g   # [rad/s^2]/sqrt(Hz)
        out['gyroscope_random_walk_rad_s2_sqrtHz'] = gyro_bias_rw
    if K_a is not None:
        accel_bias_rw = math.sqrt(3.0) * K_a  # [m/s^3]/sqrt(Hz)
        out['accelerometer_random_walk_m_s3_sqrtHz'] = accel_bias_rw
    return out

print({
    'gyroscope_noise_density_rad_s_sqrtHz': gyro_nd,
    'accelerometer_noise_density_m_s2_sqrtHz': accel_nd,
    'gyroscope_sigma_per_sample_rad_s': gyro_sigma_discrete,
    'accelerometer_sigma_per_sample_m_s2': accel_sigma_discrete,
    'gyro_ARW_from_noise_deg_sqrtHr': gyro_arw_from_nd_deg,
    'accel_nd_from_VRW_m_s2_sqrtHz': accel_nd_from_vrw
})
# Then call bias_rw_from_rrw(K_g_deg=<fit>, K_a=<fit>) after you fit the +1/2 slope.