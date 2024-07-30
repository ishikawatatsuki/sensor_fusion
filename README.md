# sensor_fusion

This repository is designed to create various filters and apply them to Kitti datasets and UAV (Unmanned Aerial Vehicle) datasets collected at Taltech.

# Features

## Filters
- Extended Kalman filter
- Unscented Kalman filter
- Cubature Kalman filter
- Particle filter
- Ensemble Kalman filter

# Quick start

## Prerequisite

- docker
- docker-compose

## Pull repository

Firstly pull the github repository in your local environment with the command:
```
git clone https://github.com/ishikawatatsuki/sensor_fusion.git
```

## Create Docker container

Given docker and docker compose installed on your host machine, run the following command to create Docker container on your machine:
```
make jupyterlab_start
```
. The command creates a docker container and start jupyter lab inside the container which can be accessed from your local machine. 

## Testing python script individually

To test python script, given the docker container running, execute the following command to enter the container:
```
make jupyterlab_run
```
. The command enables you to enter the container that runs the jupyter server. Since bash shell script is enabled, you can navigate using linux commands and execute python scripts under the src directory by simply running `python3 {filename}.py`.

## Download Kitti dataset

To download Kitti dataset, executing the shell script `raw_data_downloader.sh` creates a data directory and download the required data with the following command:
```
# Make shell script executable
chmod +x raw_data_downloader.sh

# Run the shell script
./raw_data_downloader.sh
```
You can select which sequence of Kitti dataset to download by editing the script directly.