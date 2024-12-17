#!/bin/bash

LOCATION_DIR='./data/UAV'

mkdir -p $LOCATION_DIR

files=(
  log0001.zip
  log0002.zip
  log0003.zip
)

for i in ${files[@]}; do
  shortname=$LOCATION_DIR'/'$i
  fullname=$i
	echo "Downloading: "$shortname
  wget 'https://uav-buckect.s3.eu-north-1.amazonaws.com/'$fullname -P $LOCATION_DIR
  unzip -o $shortname -d $LOCATION_DIR
  rm $shortname
done
