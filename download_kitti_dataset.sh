#!/bin/bash

LOCATION_DIR='./data/KITTI'

# KITTI sequence to raw data map
# Nr.     Sequence name     Start   End
# ---------------------------------------
# 00: 2011_10_03_drive_0027 000000 004540
# 01: 2011_10_03_drive_0042 000000 001100
# 02: 2011_10_03_drive_0034 000000 004660
# 03: 2011_09_26_drive_0067 000000 000800
# 04: 2011_09_30_drive_0016 000000 000270
# 05: 2011_09_30_drive_0018 000000 002760
# 06: 2011_09_30_drive_0020 000000 001100
# 07: 2011_09_30_drive_0027 000000 001100
# 08: 2011_09_30_drive_0028 001100 005170
# 09: 2011_09_30_drive_0033 000000 001590
# 10: 2011_09_30_drive_0034 000000 001200

mkdir -p $LOCATION_DIR

files=(
        2011_09_26_calib.zip
        2011_09_30_calib.zip
        2011_10_03_calib.zip
        2011_10_03_drive_0042
        2011_10_03_drive_0034
        2011_09_26_drive_0067
        2011_09_30_drive_0016
        2011_09_30_drive_0018
        2011_09_30_drive_0020
        2011_09_30_drive_0027
        2011_09_30_drive_0028
        2011_09_30_drive_0033
        2011_09_30_drive_0034
)

for i in ${files[@]}; do
        if [ ${i:(-3)} != "zip" ]
        then
                shortname=$LOCATION_DIR'/'$i'_sync.zip'
                fullname=$i'/'$i'_sync.zip'
        else
                shortname=$LOCATION_DIR'/'$i
                fullname=$i
        fi
	echo "Downloading: "$shortname
        wget 'https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/'$fullname -P $LOCATION_DIR
        unzip -o $shortname -d $LOCATION_DIR
        rm $shortname
done


GROUND_TRUTH_DIR=$LOCATION_DIR'/ground_truth'
filename=$GROUND_TRUTH_DIR'/data_odometry_poses.zip'
GROUND_TRUTH_SRC_DIR=$LOCATION_DIR'/ground_truth/dataset/poses/*'

mkdir -p $GROUND_TRUTH_DIR

# Downloading ground truth data
wget 'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_poses.zip' -P $GROUND_TRUTH_DIR
unzip -o $filename -d $GROUND_TRUTH_DIR
rm $filename
mv $GROUND_TRUTH_SRC_DIR $GROUND_TRUTH_DIR
rm -rf $GROUND_TRUTH_DIR'/dataset'


VO_CARIBRATION_DIR=$LOCATION_DIR'/vo_calibrations'
filename=$VO_CARIBRATION_DIR'/data_odometry_calib.zip'
VO_CARLIBRATION_SRC_DIR=$LOCATION_DIR'/vo_calibrations/dataset/sequences/*'

mkdir -p $VO_CARIBRATION_DIR

# Downloading camera calibration
wget 'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_calib.zip' -P $VO_CARIBRATION_DIR
unzip -o $filename -d $VO_CARIBRATION_DIR
rm $filename
mv $VO_CARLIBRATION_SRC_DIR $VO_CARIBRATION_DIR
rm -rf $VO_CARIBRATION_DIR'/dataset'

# Downloading visual odometry estimates
vo_files=(
        vo_pose_estimates.zip
        vo_pose_estimates_2d3d.zip
        vo_pose_estimates_hybrid.zip
        vo_pose_estimates_stereo.zip
)
for i in ${vo_files[@]}; do
        shortname=$LOCATION_DIR'/'$i
        fullname=$i
	echo "Downloading: "$shortname
        wget 'https://kitti-vo-estimates-bucket.s3.eu-north-1.amazonaws.com/'$fullname -P $LOCATION_DIR
        unzip -o $shortname -d $LOCATION_DIR
        rm -rf $shortname
done