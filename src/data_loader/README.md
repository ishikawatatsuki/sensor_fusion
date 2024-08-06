This directory contains two scripts, which load data from KITTI and UAV datasets by `data_loader.py` and `uav_data_loader.py` respectively.

## DataLoader
DataLoader class loads KITTI dataset and apply preprocessing such as adding a noise and slicing the data to have the same length.
Moreover, user can set dropout ration, scaling from 0.0 to 1.0 for each Visual Odometry and GPS so that some sensor data are unavailable during a filter process.
In terms of dropout, there are two options. The first option, `MeasurementDataEnum.DROPOUT`, indicates that the sensor data is completely unavailable. While the second option, `MeasurementDataEnum.COVARIANCE`, provides the sensor data with large covariance matrix which causes the filter relying on the estimate in time update step. The covariance option enables you to obtain realistic data such as GPS cycle slip and Visual Odometry uncertainty due to the less feature points available during the estimation. 


## UAV_DataLoader
UAV_Dataloader class loads the UAV dataset collected at Taltech in Estonia and apply preprocessing such as conversion from LLA(Latitude, Longitude, and Altitude) into NED(North, East, and Down).
The 