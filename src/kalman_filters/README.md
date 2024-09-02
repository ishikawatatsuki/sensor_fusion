## Kalman Filters
This directory consists of 5 major Kalman Filters:
  * Extended Kalman Filter (EKF)
  * Unscented Kalman Filter (UKF)
  * Particle Filter (PF)
  * Ensemble Kalman Filter (EnKF)
  * Cubature Kalman Filter (CKF)

## Run filters
To run a filter, type `python3 {}.py` in your terminal to execute the python script. By default, the dataset that all the filters refer to is the KITTI example dataset located under the `example_data` directory. Since example data contains only small amount of KITTI raw dataset, it is used for a testing purpose. 

To change the dataset with entire trajectory provided by KITTI raw dataset, firstly run the `kitti_data_downloader.sh` shell script to download the KITTI dataset in root directory. The dataset consists of 7.7 GB including two sequences of data, `0016` and `0033`. The downloaded KITTI data will be located under the data directory.

After downloading the KITTI dataset, you can comment out example data path and undo comment out the path of entire sequence data in each script as follows:

From:
```
    root_path = "../../"
    kitti_drive = 'example'
    kitti_data_root_dir = os.path.join(root_path, "example_data")
    noise_vector_dir = os.path.join(root_path, "exports/_noise_optimizations/noise_vectors")
    dimension=2

    # Undo comment out this to change example data to entire sequence data
    # root_path = "../../"
    # kitti_drive = '0033'
    # kitti_data_root_dir = os.path.join(root_path, "data")
    # noise_vector_dir = os.path.join(root_path, "exports/_noise_optimizations/noise_vectors")
    # dimension=2

    data = DataLoader(
        sequence_nr=kitti_drive, 
        kitti_root_dir=kitti_data_root_dir, 
        noise_vector_dir=noise_vector_dir,
        vo_dropout_ratio=0., 
        gps_dropout_ratio=0.,
        visualize_data=False,
        dimension=dimension
    )
```

To:
```
    # root_path = "../../"
    # kitti_drive = 'example'
    # kitti_data_root_dir = os.path.join(root_path, "example_data")
    # noise_vector_dir = os.path.join(root_path, "exports/_noise_optimizations/noise_vectors")
    # dimension=2

    # Undo comment out this to change example data to entire sequence data
    root_path = "../../"
    kitti_drive = '0033'
    kitti_data_root_dir = os.path.join(root_path, "data")
    noise_vector_dir = os.path.join(root_path, "exports/_noise_optimizations/noise_vectors")
    dimension=2

    data = DataLoader(
        sequence_nr=kitti_drive, 
        kitti_root_dir=kitti_data_root_dir, 
        noise_vector_dir=noise_vector_dir,
        vo_dropout_ratio=0., 
        gps_dropout_ratio=0.,
        visualize_data=False,
        dimension=dimension
    )
```

You can change the value of vo_dropout_ratio and gps_dropout_ratio in the range 0.0 to 1.0 to dropout some measurement data for Visual Odometry and GPS respectively.
When you change the dropout ratio, change the MeasurementDataEnum from ALL_DATA, which feed entire measurement data without any dropout, with DROPOUT, for instance, to cut the measurement data at time t during filtering process.

For example, in extended_kalman_filter.py, when we change measurement type to DROPOUT: 
```

    # Set 50% dropout for Visual Odometry data and 90% dropout for GPS data
    data = DataLoader(sequence_nr=kitti_drive, 
                  kitti_root_dir=kitti_data_root_dir, 
                  noise_vector_dir=noise_vector_dir,
                  vo_dropout_ratio=0.5, 
                  gps_dropout_ratio=0.9,
                  dimension=dimension)

    # Set measurement type to DROPOUT
    measurement_type=MeasurementDataEnum.DROPOUT

    # Instantiate EKF
    ekf2_0 = ExtendedKalmanFilter(
        x=x_setup2.copy(), 
        P=P_setup2.copy(), 
        H=H_setup2.copy(),
        q=q2,
        r_vo=r_vo2,
        r_gps=r_gps2,
        setup=SetupEnum.SETUP_2
        )
    # Run EKF
    error_ekf2_0 = ekf2_0.run(
        data=data, 
        debug_mode=debug_mode, 
        measurement_type=measurement_type)
    
    # Visualize estimated trajectories to check the effects of dropout
    ekf2_0.visualize_trajectory(
        data=data, 
        dimension=dimension, 
        interval=interval, 
        title="EKF Setup2 trajectories")
```

For COVARIANCE, the extended_kalman_filter.py, for example, will be:
```

    # Set 50% dropout for Visual Odometry data and 90% dropout for GPS data
    data = DataLoader(sequence_nr=kitti_drive, 
                    kitti_root_dir=kitti_data_root_dir, 
                    noise_vector_dir=noise_vector_dir,
                    vo_dropout_ratio=0.5, 
                    gps_dropout_ratio=.9,
                    dimension=dimension)

    # Set measurement type to DROPOUT
    measurement_type=MeasurementDataEnum.COVARIANCE

    # Set uncertainty for each sensor. The measurement error covariance matrix will be constructed based on the uncertainty value.
    data.change_sensor_uncertainty(vo_uncertainty_std=2.0, gps_uncertainty_std=1000.0)

    # Instantiate EKF
    ekf2_0 = ExtendedKalmanFilter(
        x=x_setup2.copy(), 
        P=P_setup2.copy(), 
        H=H_setup2.copy(),
        q=q2,
        r_vo=r_vo2,
        r_gps=r_gps2,
        setup=SetupEnum.SETUP_2
        )
    # Run EKF
    error_ekf2_0 = ekf2_0.run(
        data=data, 
        debug_mode=debug_mode, 
        measurement_type=measurement_type)
    
    # Visualize estimated trajectories to check the effects of change of measurement error.
    ekf2_0.visualize_trajectory(
        data=data, 
        dimension=dimension, 
        interval=interval, 
        title="EKF Setup2 trajectories")
```

