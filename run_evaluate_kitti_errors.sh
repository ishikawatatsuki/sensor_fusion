#!/bin/bash

# Assign command-line arguments to variables
FOLDER_PATH="/app/outputs/errors_for_eval/filter_comparison_final/"

# Verify if the folder exists
if [ ! -d "$FOLDER_PATH" ]; then
  echo "Error: Folder '$FOLDER_PATH' does not exist."
  exit 1
fi


# Walk through the folder and process .txt files
find "$FOLDER_PATH" -type f -name "*.txt" -exec dirname {} \; | sort -u | while read -r DIR; do
  echo "Processing file: $DIR"
  
  # Run the Python script with the file as an argument
  python /app/libs/kitti-odom-eval/eval_odom.py --result "$DIR"
  
  # Check if the Python script executed successfully
  if [ $? -ne 0 ]; then
    echo "Error: Failed to process file '$DIR' with script '$PYTHON_SCRIPT'"
  fi
done
