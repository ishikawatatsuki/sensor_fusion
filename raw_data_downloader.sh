#!/bin/bash

LOCATION_DIR='./data'

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
2011_10_03_calib.zip
2011_10_03_drive_0027
# 2011_09_30_calib.zip
# 2011_09_30_drive_0016
# 2011_09_30_drive_0033
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
