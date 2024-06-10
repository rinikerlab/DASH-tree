#!/bin/bash

# Check if the directory is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <directory>"
  exit 1
fi

# Get the directory path
DIRECTORY=$1

# Check if the provided path is a directory
if [ ! -d "$DIRECTORY" ]; then
  echo "Error: $DIRECTORY is not a directory."
  exit 1
fi

# Loop through all .xz.part files and recombine them
for PART_FILE in "$DIRECTORY"/*.xz.part*;
do
  # Extract the base filename without the .part* suffix
  BASE_NAME=$(echo "$PART_FILE" | sed 's/.part[0-9]*$//')

  # Check if the recombined file already exists, if not, create it
  if [ ! -f "$BASE_NAME" ]; then
    cat "$BASE_NAME".part* > "$BASE_NAME"
  fi

  # Decompress the recombined file with lzma
  xz -d --format=lzma "$BASE_NAME"
done

echo "Recombination and decompression completed."
