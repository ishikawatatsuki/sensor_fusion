CONTAINER_NAME := sensor_fusion-jupyterlab

jupyterlab_start:
	docker-compose up -d

jupyterlab_start_with_build:
	docker-compose up --build -d
	
jupyterlab_run:
	docker exec -it $(shell echo | docker compose ps --format '{{.Name}}' | grep ${CONTAINER_NAME}) /bin/bash

jupyterlab_stop:
	docker-compose down

help:
	@echo  'jupyterlab_start	- Start jupyter lab server inside the container.'
	@echo  'jupyterlab_start_with_build	- Start jupyter lab server with build. Run this whenever you change Dockerfile.'
	@echo  'jupyterlab_stop	- Stop container that runs jupyter lab server.'
	@echo  'jupyterlab_run	- Enter the jupyterlab container, by which you can run python script individually.'
	@echo  ''


.PHONY: jupyterlab_start jupyterlab_run


