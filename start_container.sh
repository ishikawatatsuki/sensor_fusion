#!/bin/bash

TAG="1.0"
docker build . --tag sensor_fusion:$TAG
docker run --rm -it --entrypoint bash sensor_fusion:$TAG