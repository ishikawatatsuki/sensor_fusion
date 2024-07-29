import os
import numpy as np
import pandas as pd
from datetime import datetime, timezone

# px4 -> log_xx_YYYY-mm-dd-HH-MM-SS-csv
# voxl -> YYYY-mm-dd-logxxxxx
VERSION = "v2"

file_combinations = {
    '0001': {
        'px4': {
            'timestamp': "2024-05-24T13:21:36",
            "folder_name": "log_87_2024-5-24-13-21-36-csv",
            'gps': "log_87_2024-5-24-13-21-36_vehicle_global_position_0.csv", 
            'imu0': "log_87_2024-5-24-13-21-36_vehicle_imu_0.csv",
            'imu1': "log_87_2024-5-24-13-21-36_vehicle_imu_1.csv",
        },
        'voxl': {
            'timestamp': "2024-05-24T13:19:07", #UTC+3 hours
            'folder_name': "2024-05-24-log0001",  
            'imu0':"imu0/data.csv",
            'imu1': "imu1/data.csv",
            'stereo': 'stereo/data.csv'
        },
        'output': f"{VERSION}/2024-05-24-log0001.csv",
    },
    '0005': {
        'px4': {
        'timestamp': "2024-05-29T15:52:22",
            "folder_name": "log_91_2024-5-29-15-52-22-csv",
            'gps': "log_91_2024-5-29-15-52-22_vehicle_global_position_0.csv",
            'imu0': "log_91_2024-5-29-15-52-22_vehicle_imu_0.csv",
            'imu1': "log_91_2024-5-29-15-52-22_vehicle_imu_1.csv",
        },
        'voxl': {
            'timestamp': "2024-05-29T15:50:28", #UTC+3 hours
            'folder_name': "2024-05-29-log0005",
            'imu0':"imu0/data.csv",
            'imu1': "imu1/data.csv",
            'stereo': 'stereo/data.csv'
        },
        'output': f"{VERSION}/2024-05-29-log0005.csv",
    },
    '0006': {
        'px4': {
        'timestamp': "2024-05-29T15:56:42",
            'folder_name': "log_92_2024-5-29-15-56-42-csv",
            'gps': "log_92_2024-5-29-15-56-42_vehicle_global_position_0.csv",
            'imu0': "log_92_2024-5-29-15-56-42_vehicle_imu_0.csv",
            'imu1': "log_92_2024-5-29-15-56-42_vehicle_imu_1.csv",  
        },
        'voxl': {
            'timestamp': "2024-05-29T15:54:40", #UTC+3 hours
            'folder_name': "2024-05-29-log0006",
            'imu0':"imu0/data.csv",
            'imu1': "imu1/data.csv",
            'stereo': 'stereo/data.csv'
        },
        'output': f"{VERSION}/2024-05-29-log0006.csv"
    },
    '0008': {
        'px4': {
            'timestamp': "2024-05-31T14:15:10", #assuming this is log end time
            'folder_name': "log_94_2024-5-31-14-15-10-csv",
            'gps': "log_94_2024-5-31-14-15-10_vehicle_global_position_0.csv",
            'imu0': "log_94_2024-5-31-14-15-10_vehicle_imu_0.csv",
            'imu1': "log_94_2024-5-31-14-15-10_vehicle_imu_1.csv",  
        },
        'voxl': {
            'timestamp': "2024-05-31T14:13:37", #UTC+3 hours
            'folder_name': "2024-05-31-log0008",
            'imu0':"imu0/data.csv",
            'imu1': "imu1/data.csv",
            'stereo': 'stereo/data.csv'
        },
        'output': f"{VERSION}/2024-05-31-log0008.csv"
    }
}

variant = file_combinations["0006"]

EXPORT_ROOT_PATH = "_exports"
EXPORT_FILENAME = os.path.join(EXPORT_ROOT_PATH, variant['output'])

voxl = variant['voxl']
px4 = variant['px4']

SENSOR_DATA_ROOT_PATH = os.path.join(voxl['folder_name'], "run", "mpa")

voxl_imu0_path = os.path.join(SENSOR_DATA_ROOT_PATH, voxl['imu0'])
voxl_imu1_path = os.path.join(SENSOR_DATA_ROOT_PATH, voxl['imu1'])
px4_imu0_path = os.path.join(px4['folder_name'], px4["imu0"])
px4_imu1_path = os.path.join(px4['folder_name'], px4["imu1"])
px4_gps_path = os.path.join(px4['folder_name'], px4["gps"])

stereo_path = os.path.join(SENSOR_DATA_ROOT_PATH, voxl['stereo'])


def iso_to_nanoseconds(iso_time_str):
    # Parse the ISO time string to a datetime object
    dt = datetime.fromisoformat(iso_time_str)
    
    # Ensure the datetime object is in UTC
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    
    # Calculate the number of nanoseconds
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    delta = dt - epoch
    
    # Convert the timedelta to nanoseconds
    nanoseconds = delta.total_seconds() * 1e9 + delta.microseconds * 1e3
    return int(nanoseconds)
    

def get_px4_timestamp(df, PX4_END_TIMESTAMP):
    df["timestamp_delta"] = df["timestamp"].diff()*1000
    df.loc[0, "timestamp_delta"] = df["timestamp_delta"].iloc[1] #filling NaN
    df["timestamp_reverse_cumsum"] = df.loc[::-1, "timestamp_delta"].cumsum()[::-1]
    return df[["timestamp_reverse_cumsum"]].apply(lambda x: PX4_END_TIMESTAMP - x)

def get_voxl_imu_by_timestamp(voxl_imu0, voxl_imu1, timestamp):
    return (np.argmin(np.abs(voxl_imu0["timestamp"] - timestamp)), np.argmin(np.abs(voxl_imu1["timestamp"] - timestamp)))

def get_px4_gps_by_timestamp(px4_gps, timestamp):
    return np.argmin(np.abs(px4_gps["timestamp"] - timestamp))

def pad(index, length=5):
    string = str(index)
    while length > len(string):
        string = "0" + string
    return string
    
def get_stereo_image_path(index):
    file_index = pad(index)
    left = os.path.join(SENSOR_DATA_ROOT_PATH, "stereo", file_index+"l.png")
    right = os.path.join(SENSOR_DATA_ROOT_PATH, "stereo", file_index+"r.png")
    assert os.path.exists(left) and os.path.exists(right), "Image does not exist"
    return (left, right)

def get_upsampled_gps_data(df, gps_indices):
    overlap_count = 0
    _gps_idx = 0
    data = []
    for gps_idx in gps_indices:
        value = df.iloc[gps_idx].values
        if _gps_idx != gps_idx:
            if overlap_count > 0:
                prev_value = df.iloc[_gps_idx].values
                overlap_count += 1
                for upsample in np.linspace(value, prev_value, overlap_count, endpoint=False)[::-1]:
                    data.append(upsample.tolist())
                overlap_count = 0
            else:
                data.append(value.tolist())
            _gps_idx = gps_idx
        else:
            overlap_count += 1

    for _ in range(overlap_count):
        data.append(value)
    
    return data

def main():
    
    voxl_imu0 = pd.read_csv(voxl_imu0_path)
    voxl_imu1 = pd.read_csv(voxl_imu1_path)
    stereo = pd.read_csv(stereo_path)

    px4_gps = pd.read_csv(px4_gps_path)
    px4_imu0 = pd.read_csv(px4_imu0_path)
    px4_imu1 = pd.read_csv(px4_imu1_path)

    PX4_END_TIMESTAMP = iso_to_nanoseconds(px4['timestamp'])
    VOXL_START_TIMESTAMP = iso_to_nanoseconds(voxl['timestamp'])

    # Making timestamp of each table aligned.
    voxl_imu0["timestamp"] = voxl_imu0[["timestamp(ns)"]] \
                            .apply(lambda x:  x - voxl_imu0["timestamp(ns)"].iloc[0]) \
                            .apply(lambda x: VOXL_START_TIMESTAMP + x)
    voxl_imu1["timestamp"] = voxl_imu1[["timestamp(ns)"]] \
                                .apply(lambda x:  x - voxl_imu1["timestamp(ns)"].iloc[0]) \
                                .apply(lambda x: VOXL_START_TIMESTAMP + x)
    stereo["timestamp"] = stereo[["timestamp(ns)"]] \
                            .apply(lambda x:  x - stereo["timestamp(ns)"].iloc[0]) \
                            .apply(lambda x: VOXL_START_TIMESTAMP + x)
                            

    px4_gps["timestamp"] = get_px4_timestamp(px4_gps, PX4_END_TIMESTAMP)
    px4_imu0["timestamp"] = get_px4_timestamp(px4_imu0, PX4_END_TIMESTAMP)
    px4_imu1["timestamp"] = get_px4_timestamp(px4_imu1, PX4_END_TIMESTAMP)

    print("Logging duration of voxl imu0: {} seconds, sample rate: {}Hz"\
        .format(
            ((voxl_imu0["timestamp"].iloc[-1] - voxl_imu0["timestamp"].iloc[0])/ 1e9),
            np.round(voxl_imu0.shape[0] / ((voxl_imu0["timestamp"].iloc[-1] - voxl_imu0["timestamp"].iloc[0]) / 1e9))
        ))
    print("Logging duration of voxl imu1: {} seconds, sample rate: {}Hz"\
        .format(
            ((voxl_imu1["timestamp"].iloc[-1] - voxl_imu1["timestamp"].iloc[0])/ 1e9),
            np.round(voxl_imu1.shape[0] / ((voxl_imu1["timestamp"].iloc[-1] - voxl_imu1["timestamp"].iloc[0]) / 1e9))
        ))
    print("Logging duration of stereo: {} seconds, sample rate: {}Hz"\
        .format(
            ((stereo["timestamp"].iloc[-1] - stereo["timestamp"].iloc[0])/ 1e9),
            np.round(stereo.shape[0] / ((stereo["timestamp"].iloc[-1] - stereo["timestamp"].iloc[0]) / 1e9))
        ))
    print("Logging duration of px4 imu0: {} seconds, sample rate: {}Hz"\
        .format(
            ((px4_imu0["timestamp"].iloc[-1] - px4_imu0["timestamp"].iloc[0])/ 1e9),
            np.round(px4_imu0.shape[0] / ((px4_imu0["timestamp"].iloc[-1] - px4_imu0["timestamp"].iloc[0]) / 1e9))
        ))
    print("Logging duration of px4 imu1: {} seconds, sample rate: {}Hz"\
        .format(
            ((px4_imu1["timestamp"].iloc[-1] - px4_imu1["timestamp"].iloc[0])/ 1e9),
            np.round(px4_imu1.shape[0] / ((px4_imu1["timestamp"].iloc[-1] - px4_imu1["timestamp"].iloc[0]) / 1e9))
        ))
    print("Logging duration of px4 gps: {} seconds, sample rate: {}Hz"\
        .format(
                ((px4_gps["timestamp"].iloc[-1] - px4_gps["timestamp"].iloc[0])/ 1e9),
                np.round(px4_gps.shape[0] / ((px4_gps["timestamp"].iloc[-1] - px4_gps["timestamp"].iloc[0]) / 1e9))
        ))
    
    # Get voxl data resides in px4 log duration
    voxl_imu0 = voxl_imu0.loc[
        (voxl_imu0["timestamp"] >= px4_gps["timestamp"].iloc[0]) &
        (voxl_imu0["timestamp"] <= px4_gps["timestamp"].iloc[-1])
    ]
    voxl_imu1 = voxl_imu1.loc[
        (voxl_imu1["timestamp"] >= px4_gps["timestamp"].iloc[0]) &
        (voxl_imu1["timestamp"] <= px4_gps["timestamp"].iloc[-1])
    ]
    stereo = stereo.loc[
        (stereo["timestamp"] >= px4_gps["timestamp"].iloc[0]) &
        (stereo["timestamp"] <= px4_gps["timestamp"].iloc[-1])
    ]
    imu_columns = ['timestamp', 'AX(m/s2)', 'AY(m/s2)', 'AZ(m/s2)', 'GX(rad/s)', 'GY(rad/s)', 'GZ(rad/s)']
    gps_columns = ['timestamp', 'lat', 'lon', 'alt']
    
    voxl_imu0_new_columns = ["voxl_imu0_" + col for col in imu_columns]
    voxl_imu1_new_columns = ["voxl_imu1_" + col for col in imu_columns]
    
    new_voxl_imu0 = voxl_imu0[imu_columns]
    new_voxl_imu1 = voxl_imu1[imu_columns]
    new_px4_gps = px4_gps[gps_columns]
    
    imu_indices = stereo["timestamp"].apply(lambda x: get_voxl_imu_by_timestamp(voxl_imu0, voxl_imu1, x))
    gps_indices = stereo["timestamp"].apply(lambda x: get_px4_gps_by_timestamp(px4_gps, x))
    image_path = stereo["i"].apply(lambda x: get_stereo_image_path(x))
    
    voxl_imu_df = pd.DataFrame(
        np.array([
            np.concatenate([
                new_voxl_imu0.iloc[idx[0]].values,
                new_voxl_imu1.iloc[idx[1]].values
            ], axis=0) for idx in imu_indices]), 
        columns=np.concatenate([voxl_imu0_new_columns, voxl_imu1_new_columns]).tolist())
    
    _gps_data = get_upsampled_gps_data(new_px4_gps, gps_indices.tolist())
    print(_gps_data)
    
    px4_gps_df = pd.DataFrame(
        np.array(_gps_data), 
        columns=gps_columns)
    
    stereo_df = pd.concat([
            stereo.reset_index()[["timestamp"]],
            pd.DataFrame(np.array([path for path in image_path]), 
                        columns=["stereo image left", "stereo image right"])
        ], axis=1)
    df = pd.concat([
        stereo_df,
        px4_gps_df,
        voxl_imu_df
    ], axis=1)
    
    df.to_csv(EXPORT_FILENAME, index=False)
    
if __name__ == "__main__":
    main()