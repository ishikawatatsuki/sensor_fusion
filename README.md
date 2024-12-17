# Sensor Fusion

This repository is designed to integrate various filters and apply them to KITTI datasets and UAV (Unmanned Aerial Vehicle) datasets collected at Taltech in Estonia.

# Features

## Filters
* Extended Kalman filter
* Unscented Kalman filter
* Cubature Kalman filter
* Particle filter
* Ensemble Kalman filter

# Quick start

## Prerequisites

- docker

## Clone this repository

Firstly pull the github repository in your local environment with the command:

```
git clone https://github.com/ishikawatatsuki/sensor_fusion.git
```

## To create Docker container

Given docker installed on your host machine, run the following command to create Docker container on your machine:

```
make build
```

The command creates a docker container, in which required python libraries are installed.


## To download Kitti datasets

To download Kitti dataset, executing the shell script namely`kitti_data_downloader.sh` creates a directory called `data` and download the Kitti raw dataset under the directory. The command is as follows:

```
# Run the shell script
./kitti_data_downloader.sh
```
If permission error is shown, make shell script executable by following command: `chmod +x kitti_data_downloader.sh`.

You can select which sequence of Kitti dataset to download by editing the script directly.

## To download UAV (Unmanned Aerial Vehicle) data corrected at Taltech
Besides Kitti dataset, we also prepared UAV data corrected at Taltech. To download the dataset, likewise Kitti data, run the following commands:
```
# Run the shell script
./uav_data_downloader.sh
```
If permission error is shown, make shell script executable by following command: `chmod +x uav_data_downloader.sh`.

The commands create a directory called `UAV` under the `data` directory and download 5 UAV datasets.


## To test jupyter notebook

To run jupyter notebooks stored under the notebooks directory, run the following command:
```
make jupyter_up
```
. This command starts jupyter lab inside the docker container and you can access the jupyter lab environment through the browser by entering the URL shown in your terminal.
For example, this is the output shown after you run `make jupyter_up` command:

```
06:31:07 taki@agx sensor_fusion ±|master ✗|→ make jupyter_up
docker run --rm --user root -p 8889:8888 -v .:/app -it sensor_fusion:1.0 jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --notebook-dir=/app --allow-root
[I 2024-07-31 15:31:17.150 ServerApp] jupyter_lsp | extension was successfully linked.
[I 2024-07-31 15:31:17.162 ServerApp] jupyter_server_terminals | extension was successfully linked.
[I 2024-07-31 15:31:17.177 ServerApp] jupyterlab | extension was successfully linked.
[I 2024-07-31 15:31:17.190 ServerApp] notebook | extension was successfully linked.
[I 2024-07-31 15:31:17.194 ServerApp] Writing Jupyter server cookie secret to /root/.local/share/jupyter/runtime/jupyter_cookie_secret
[I 2024-07-31 15:31:18.104 ServerApp] notebook_shim | extension was successfully linked.
[I 2024-07-31 15:31:18.155 ServerApp] notebook_shim | extension was successfully loaded.
[I 2024-07-31 15:31:18.161 ServerApp] jupyter_lsp | extension was successfully loaded.
[I 2024-07-31 15:31:18.164 ServerApp] jupyter_server_terminals | extension was successfully loaded.
[I 2024-07-31 15:31:18.170 LabApp] JupyterLab extension loaded from /usr/local/lib/python3.12/site-packages/jupyterlab
[I 2024-07-31 15:31:18.170 LabApp] JupyterLab application directory is /usr/local/share/jupyter/lab
[I 2024-07-31 15:31:18.171 LabApp] Extension Manager is 'pypi'.
[I 2024-07-31 15:31:18.269 ServerApp] jupyterlab | extension was successfully loaded.
[I 2024-07-31 15:31:18.279 ServerApp] notebook | extension was successfully loaded.
[I 2024-07-31 15:31:18.280 ServerApp] Serving notebooks from local directory: /app
[I 2024-07-31 15:31:18.280 ServerApp] Jupyter Server 2.13.0 is running at:
[I 2024-07-31 15:31:18.280 ServerApp] http://bbc53700ae0c:8888/lab?token=cb1963f12bd2299eb863ea565b564cb4b9f3089f1c14b582
[I 2024-07-31 15:31:18.280 ServerApp]     http://127.0.0.1:8888/lab?token=cb1963f12bd2299eb863ea565b564cb4b9f3089f1c14b582
[I 2024-07-31 15:31:18.280 ServerApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
[C 2024-07-31 15:31:18.291 ServerApp]

    To access the server, open this file in a browser:
        file:///root/.local/share/jupyter/runtime/jpserver-1-open.html
    Or copy and paste one of these URLs:
        http://bbc53700ae0c:8888/lab?token=cb1963f12bd2299eb863ea565b564cb4b9f3089f1c14b582
        http://127.0.0.1:8888/lab?token=cb1963f12bd2299eb863ea565b564cb4b9f3089f1c14b582
```
Copy the URL starting with `http://127.0.0.1:8888/lab?token=` and access the URL on your browser. It navigates to the jupyter lab environment.


## To test python script individually

To test python script individually, execute the following command to enter the container:
```
make container_up
```
The command enables you to enter the container and enables bash shell so that you can enter linux command to freely navigate in the repository. To run python scripts under the src directory, simply type `python3 {filename}.py`.
