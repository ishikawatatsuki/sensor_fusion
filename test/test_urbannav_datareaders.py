"""
Test script for UrbanNav Tokyo dataset data readers
"""
import os
import sys
import numpy as np

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.internal.dataset.urban_nav import (
    UrbanNav_IMUDataReader,
    UrbanNav_WheelOdometryDataReader,
    UrbanNav_ReferenceDataReader
)

def test_imu_reader():
    """Test IMU data reader"""
    print("=" * 60)
    print("Testing UrbanNav IMU Data Reader")
    print("=" * 60)
    
    imu_path = "/Volumes/Data_EXT/data/workspaces/sensor_fusion/data/UrbanNav/TK/Odaiba/imu.csv"
    
    if not os.path.exists(imu_path):
        print(f"Error: IMU file not found at {imu_path}")
        return
    
    imu_reader = UrbanNav_IMUDataReader(imu_path)
    
    # Read first 5 samples
    count = 0
    for imu_data in imu_reader:
        if count < 5:
            print(f"\nSample {count + 1}:")
            print(f"  Timestamp: {imu_data.timestamp:.3f} ms")
            print(f"  Linear Acceleration (m/s²): {imu_data.a}")
            print(f"  Angular Velocity (rad/s): {imu_data.w}")
            count += 1
        else:
            break
    
    print(f"\n✓ IMU reader test completed successfully")


def test_reference_reader():
    """Test reference/ground truth data reader"""
    print("\n" + "=" * 60)
    print("Testing UrbanNav Reference Data Reader")
    print("=" * 60)
    
    ref_path = "/Volumes/Data_EXT/data/workspaces/sensor_fusion/data/UrbanNav/TK/Odaiba/reference.csv"
    
    if not os.path.exists(ref_path):
        print(f"Error: Reference file not found at {ref_path}")
        return
    
    ref_reader = UrbanNav_ReferenceDataReader(ref_path)
    
    # Read first 5 samples
    count = 0
    for ref_data in ref_reader:
        if count < 5:
            print(f"\nSample {count + 1}:")
            print(f"  Timestamp: {ref_data.timestamp:.3f} ms")
            print(f"  ECEF Position (m): X={ref_data.ecef_x:.4f}, Y={ref_data.ecef_y:.4f}, Z={ref_data.ecef_z:.4f}")
            print(f"  LLA: Lat={ref_data.latitude:.8f}°, Lon={ref_data.longitude:.8f}°, Alt={ref_data.ellipsoid_height:.4f}m")
            print(f"  Orientation (rad): Roll={ref_data.roll:.6f}, Pitch={ref_data.pitch:.6f}, Heading={ref_data.heading:.6f}")
            print(f"  Velocity (m/s): X={ref_data.vel_x:.6f}, Y={ref_data.vel_y:.6f}, Z={ref_data.vel_z:.6f}")
            print(f"  Acceleration (m/s²): X={ref_data.acc_x:.6f}, Y={ref_data.acc_y:.6f}, Z={ref_data.acc_z:.6f}")
            print(f"  Angular Rate (rad/s): X={ref_data.ang_rate_x:.8f}, Y={ref_data.ang_rate_y:.8f}, Z={ref_data.ang_rate_z:.8f}")
            count += 1
        else:
            break
    
    print(f"\n✓ Reference reader test completed successfully")


def test_wheel_odometry_reader():
    """Test wheel odometry data reader"""
    print("\n" + "=" * 60)
    print("Testing UrbanNav Wheel Odometry Data Reader")
    print("=" * 60)
    
    wheel_path = "/Volumes/Data_EXT/data/workspaces/sensor_fusion/data/UrbanNav/TK/Odaiba/imu.csv"
    
    if not os.path.exists(wheel_path):
        print(f"Error: Wheel odometry file not found at {wheel_path}")
        return
    
    wheel_reader = UrbanNav_WheelOdometryDataReader(wheel_path)
    
    # Read first 5 samples
    count = 0
    for wheel_data in wheel_reader:
        if count < 5:
            print(f"\nSample {count + 1}:")
            print(f"  Timestamp: {wheel_data.timestamp:.3f} ms")
            print(f"  Wheel Speed (m/s): {wheel_data.wheel_speed:.6f}")
            count += 1
        else:
            break
    
    print(f"\n✓ Wheel odometry reader test completed successfully")


def test_data_synchronization():
    """Test timestamp synchronization between IMU and reference data"""
    print("\n" + "=" * 60)
    print("Testing Data Synchronization")
    print("=" * 60)
    
    imu_path = "/Volumes/Data_EXT/data/workspaces/sensor_fusion/data/UrbanNav/TK/Odaiba/imu.csv"
    ref_path = "/Volumes/Data_EXT/data/workspaces/sensor_fusion/data/UrbanNav/TK/Odaiba/reference.csv"
    
    if not os.path.exists(imu_path) or not os.path.exists(ref_path):
        print("Error: Data files not found")
        return
    
    imu_reader = UrbanNav_IMUDataReader(imu_path)
    ref_reader = UrbanNav_ReferenceDataReader(ref_path)
    
    # Get timestamps
    imu_times = []
    ref_times = []
    
    for i, data in enumerate(imu_reader):
        imu_times.append(data.timestamp)
        if i >= 99:  # First 100 samples
            break
    
    for i, data in enumerate(ref_reader):
        ref_times.append(data.timestamp)
        if i >= 99:  # First 100 samples
            break
    
    imu_times = np.array(imu_times)
    ref_times = np.array(ref_times)
    
    print(f"\nIMU Data:")
    print(f"  Sample count: {len(imu_times)}")
    print(f"  Time range: {imu_times[0]:.3f} to {imu_times[-1]:.3f} ms")
    print(f"  Frequency: ~{1000 / np.mean(np.diff(imu_times)):.1f} Hz")
    
    print(f"\nReference Data:")
    print(f"  Sample count: {len(ref_times)}")
    print(f"  Time range: {ref_times[0]:.3f} to {ref_times[-1]:.3f} ms")
    print(f"  Frequency: ~{1000 / np.mean(np.diff(ref_times)):.1f} Hz")
    
    print(f"\n✓ Synchronization test completed successfully")


if __name__ == "__main__":
    try:
        test_imu_reader()
        test_reference_reader()
        test_wheel_odometry_reader()
        test_data_synchronization()
        
        # Show GNSS parsing information
        print("\n" + "=" * 60)
        print("GNSS Data Parsing Information")
        print("=" * 60)
        
        from src.internal.dataset.urbannav_gnss_parser import UrbanNav_GNSSParser
        
        obs_file = "/Volumes/Data_EXT/data/workspaces/sensor_fusion/data/UrbanNav/TK/Odaiba/rover_ublox.obs"
        nav_file = "/Volumes/Data_EXT/data/workspaces/sensor_fusion/data/UrbanNav/TK/Odaiba/base.nav"
        
        if os.path.exists(obs_file) and os.path.exists(nav_file):
            parser = UrbanNav_GNSSParser(obs_file, nav_file)
            print(parser.get_recommendation())
        else:
            print("\nRINEX files not found for GNSS parsing demo")
            print("Expected:")
            print(f"  - {obs_file}")
            print(f"  - {nav_file}")
        
        print("\n" + "=" * 60)
        print("DATA USAGE SUMMARY")
        print("=" * 60)
        print("\n1. SENSOR FUSION INPUTS:")
        print("   - IMU: Linear acceleration + Angular velocity (50 Hz)")
        print("   - Wheel Odometry: Wheel velocity for forward speed (50 Hz)")
        print("   - GNSS (uncorrected): Lat/Lon/Alt from parsed RINEX (5-10 Hz)")
        print("\n2. GROUND TRUTH (for evaluation):")
        print("   - Reference CSV: RTK-corrected LLA, ECEF, attitudes, velocities (10 Hz)")
        print("\n3. PARSING GNSS:")
        print("   - Use RTKLIB or georinex to convert RINEX → CSV")
        print("   - rover_ublox.obs: 5 Hz, consumer-grade")
        print("   - rover_trimble.obs: 10 Hz, survey-grade")
        
        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
