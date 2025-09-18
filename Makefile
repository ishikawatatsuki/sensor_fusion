OS := $(shell uname -s)

# Default values
USER_ID := $(shell id -u)
GROUP_ID := $(shell id -g)
DISPLAY_IP := $(shell ifconfig | sed -En 's/127.0.0.1//;s/.*inet (addr:)?(([0-9]*\.){3}[0-9]*).*/\2/p' | sed -n '1p'):0.0

CONTAINER_TAG := sensor_fusion
CONTAINER_KITTI_EVAL_TAG := sensor_fusion_kitti_eval

# Adjust for Windows
ifeq ($(OS), MINGW64_NT)
	USER_ID := 1000
	GROUP_ID := 1000
	DISPLAY_IP := "host.docker.internal:0.0"
endif

build:
	docker build . --file .docker/Dockerfile --tag ${CONTAINER_TAG}:1.0 --build-arg USER_ID=$(USER_ID) --build-arg GROUP_ID=$(GROUP_ID)

kitti_eval_up:
	docker build . --file .docker/Dockerfile.kitti-eval --tag ${CONTAINER_KITTI_EVAL_TAG}:1.0
	docker run \
	--mount type=bind,source=./outputs/KITTI/errors,target=/app/outputs \
	--mount type=bind,source=./libs,target=/app/libs \
	--mount type=bind,source=./data/KITTI/ground_truth,target=/app/data/KITTI/ground_truth \
	--mount type=bind,source=./run_evaluate_kitti_errors.sh,target=/app/run_evaluate_kitti_errors.sh \
	-it ${CONTAINER_KITTI_EVAL_TAG}:1.0

container_up:
	docker run --rm \
	--user root \
	--mount type=bind,source=./src,target=/app/src \
	--mount type=bind,source=./outputs,target=/app/outputs \
	--mount type=bind,source=./data,target=/app/data \
	--volume="/etc/group:/etc/group:ro" \
	--volume="/etc/passwd:/etc/passwd:ro" \
	--volume="/etc/shadow:/etc/shadow:ro" \
	--volume="/etc/sudoers.d:/etc/sudoers.d:ro" \
	--volume="/tmp/.X11-unix:/tmp/.X11-unix" \
	--env="DISPLAY=$(DISPLAY)" \
	--workdir /app/src \
	--name kalman_filter \
	-it ${CONTAINER_TAG}:1.0 bash 

jupyter_up:
	docker run --rm --user root -p 8888:8888 -v .:/app -it ${CONTAINER_TAG}:1.0 jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --notebook-dir=/app --allow-root

run_kitti:
	python -m src.pipeline --config_file ./configs/kitti_config.yaml --log_output .debugging --log_level DEBUG

run_euroc:
	python -m src.pipeline --config_file ./configs/euroc_config.yaml --log_output .debugging --log_level DEBUG

run_vo_pose_2d2d_experiment:
	python -m src._kitti.run_visual_odometry \
		--dataset_path ./data/KITTI \
		--output_path ./outputs/vo_estimates/pose_estimation_2d2d_improved \
		--config_file ./configs/kitti_config.yaml

run_vo_pose_2d3d_experiment:
	python -m src._kitti.run_visual_odometry_2d3d \
		--dataset_path ./data/KITTI \
		--output_path ./outputs/vo_estimates/pose_estimation_2d3d_improved \
		--config_file ./configs/kitti_config.yaml

run_vo_pose_hybrid_experiment:
	python -m src._kitti.run_visual_odometry_hybrid \
		--dataset_path ./data/KITTI \
		--output_path ./outputs/vo_estimates/pose_estimation_hybrid_improved \
		--config_file ./configs/kitti_config.yaml


run_vo_pose_euroc_experiment:
	python -m src._euroc.run_visual_odometry \
		--dataset_path ./data/EuRoC \
		--config_file ./configs/euroc_config.yaml


run_vo_pose_euroc_2d3d_experiment:
	python -m src._euroc.run_visual_odometry_2d3d \
		--dataset_path ./data/EuRoC \
		--config_file ./configs/euroc_config.yaml

run_vo_pose_euroc_hybrid_experiment:
	python -m src._euroc.run_visual_odometry_hybrid \
		--dataset_path ./data/EuRoC \
		--config_file ./configs/euroc_config.yaml


run_all_kitti_experiments:
	python -m src._kitti.run_all_experiments \
		--output_path ./outputs/KITTI/results \
		--log_output ./.debugging/experiments \
		--config_file ./configs/kitti_config_experiment_base.yaml \
		--checkpoint_file ./outputs/KITTI/results/checkpoint.txt

run_kitti_adaptive_noise_experiment:
	echo "Running KITTI adaptive noise experiment..."

help:
	@echo  'build	- Build the docker image.'
	@echo  'container_up	- Start the container and use bash shell.'
	@echo  'jupyter_up	- Start the container that runs jupyter lab server.'
	@echo  ''

.PHONY: jupyterlab_start jupyterlab_run