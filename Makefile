build:
	docker build . --tag sensor_fusion:1.0 --build-arg USER_ID=$(shell id -u) --build-arg GROUP_ID=$(shell id -g)

container_up:
	docker run --rm --user $(shell id -u ${USER}):$(shell id -g ${USER}) -v .:/app -it sensor_fusion:1.0 /bin/bash

jupyter_up:
	docker run --rm --user root -p 8888:8888 -v .:/app -it sensor_fusion:1.0 jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --notebook-dir=/app --allow-root

help:
	@echo  'build	- Build the container.'
	@echo  'container_up	- Start the container.'
	@echo  'jupyter_up	- Start the container that runs jupyter lab server.'
	@echo  ''


.PHONY: jupyterlab_start jupyterlab_run


