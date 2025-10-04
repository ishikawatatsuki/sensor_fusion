#!/bin/bash

LOCATION_DIR='./data/UAV'

mkdir -p $LOCATION_DIR

files=(
  log0001.zip
  log0002.zip
)

BASE_URL="https://uav-buckect.s3.eu-north-1.amazonaws.com"

for i in ${files[@]}; do
  shortname=$LOCATION_DIR'/'$i
  fullname=$i
	echo "Downloading: "$shortname
  wget 'https://uav-buckect.s3.eu-north-1.amazonaws.com/'$fullname -P $LOCATION_DIR
  unzip -o $shortname -d $LOCATION_DIR
  rm $shortname
done

# Downloading configs
filename=$LOCATION_DIR'/configs.zip'
wget $BASE_URL'/configs.zip' -P $LOCATION_DIR
unzip -o $filename -d $LOCATION_DIR
rm $filename

# Downloading models
filename=$LOCATION_DIR'/models.zip'
wget $BASE_URL'/models.zip' -P $LOCATION_DIR
unzip -o $filename -d $LOCATION_DIR
rm $filename