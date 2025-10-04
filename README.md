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

## Create a python environment
To create a Python environment, follow these steps:

1. Create a virtual environment:
    ```
    python -m venv venv
    ```

2. Activate the virtual environment:
    - On Linux/Mac:
      ```
      source venv/bin/activate
      ```
    - On Windows:
      ```
      venv\Scripts\activate
      ```

3. Install the required dependencies:
    ```
    pip install -r requirements.txt
    ```

This sets up a clean Python environment with all necessary dependencies installed.




## Download Kitti datasets

To download Kitti dataset, executing the shell script namely`download_kitti_dataset.sh` creates a directory called `data` and download the Kitti raw dataset under the directory. The command is as follows:

```
# Run the shell script
./download_kitti_dataset.sh
```
If permission error is shown, make shell script executable by following command: `chmod +x download_kitti_dataset.sh`.

You can select which sequence of Kitti dataset to download by editing the script directly.

## Download EuRoC datasets
To download EuRoC dataset, run the following commands:
```
# Run the shell script
./download_euroc_dataset.sh
```
If permission error is shown, make shell script executable by following command: `chmod +x download_euroc_dataset.sh`.
The commands create a directory called `EuRoC` under the `data` directory and download 3 UAV datasets.


## Download UAV (Unmanned Aerial Vehicle) data corrected at Taltech
To download a custom UAV dataset, run the following commands:
```
# Run the shell script
./download_uav_dataset.sh
```
If permission error is shown, make shell script executable by following command: `chmod +x download_uav_dataset.sh`.

The commands create a directory called `UAV` under the `data` directory and download 2 UAV datasets.



## Test jupyter notebook

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


## Test python script individually

To test python script individually, run the following commands in root directory:
```
python -m venv venv # This creates new python environment
source venv/bin/activate # Activate the environment
pip install -r requirements.txt # Install all the dependencies
```
The aforementioned commands setup an environment to enable you to run the script.
To run python scripts under the src directory, simply type `python3 {filename}.py`.

## Test pipeline

To test the pipeline, we prepared a Makefile to run the pipeline with specified dataset such that:
```
make run_kitti # Run the pipeline using KITTI dataset

make run_euroc # with EuRoC dataset

make run_uav # with our custom UAV dataset
```
To select sensors used in the pipeline, modify the corresponding configuration file.