#!/bin/sh

BASE_DIR=./outputs/EuRoC/visualization/1769424509480218

echo "Start creating video."

ffmpeg -r 30 -i $BASE_DIR/frames/%d.png -vcodec libx264 -crf 25 -pix_fmt yuv420p $BASE_DIR/demo_video.mp4

echo "Process finished!"

