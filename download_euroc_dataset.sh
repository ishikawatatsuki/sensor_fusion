#!/bin/bash

LOCATION_DIR='./data/EuRoC'

mkdir -p $LOCATION_DIR

files=(
    "machine_hall/MH_01_easy/MH_01_easy"
    "machine_hall/MH_02_easy/MH_02_easy"
    "machine_hall/MH_03_medium/MH_03_medium"
)

BASE_URL="http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset"

for f in "${files[@]}"; do
    zipfile="${f}.zip"
    full_url="${BASE_URL}/${zipfile}"
    dataset_name=$(basename "$f")
    target_zip="$LOCATION_DIR/${dataset_name}.zip"
    extract_dir="$LOCATION_DIR/${dataset_name}"

    echo "Downloading: $full_url"
    # wget -c "$full_url" -O "$target_zip"

    echo "Extracting into: $extract_dir"
    mkdir -p "$extract_dir"
    unzip -o "$target_zip" -d "$extract_dir"

    echo "Removing zip: $target_zip"
    rm "$target_zip"
done


# Downloading visual odometry estimates
vo_files=(
        vo_pose_estimates.zip
        vo_pose_estimates_2d3d.zip
        vo_pose_estimates_hybrid.zip
)
for i in ${vo_files[@]}; do
        shortname=$LOCATION_DIR'/'$i
        fullname=$i
	echo "Downloading: "$shortname
        wget 'https://euroc-vo-estimate-bucket.s3.eu-north-1.amazonaws.com/'$fullname -P $LOCATION_DIR
        unzip -o $shortname -d $LOCATION_DIR
        rm -rf $shortname
done


echo "All EuRoC sequences downloaded and extracted into $LOCATION_DIR"