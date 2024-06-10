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

# Loop through all files in the directory
for FILE in "$DIRECTORY"/*;
do
  # Check if it is a file
  if [ -f "$FILE" ]; then
    # Compress the file with lzma
    xz -z --format=lzma "$FILE"

    # Get the compressed file name
    COMPRESSED_FILE="${FILE}.lzma"

    # Check the size of the compressed file
    COMPRESSED_SIZE=$(stat -c%s "$COMPRESSED_FILE")

    # If compressed file size is greater than 99,000,000 bytes, split it
    if [ "$COMPRESSED_SIZE" -gt 99000000 ]; then
      split -b 80M "$COMPRESSED_FILE" "${COMPRESSED_FILE}.part"
      # Remove the original large compressed file after splitting
      rm "$COMPRESSED_FILE"
    fi
  fi
done

echo "Compression and splitting completed."
