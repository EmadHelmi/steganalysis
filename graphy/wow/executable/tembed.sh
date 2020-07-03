#!/bin/bash
# BOWS2OrigEp3 0.05
INPUT_FILES=/home/emadhelmi/myworks/stego/datasets/BOSSbase_1.01/total/PILresize/*
OUTPUT_DIR="/home/emadhelmi/myworks/stego/datasets/BOSSbase_1.01/stego/0.8/wow_PILresize"
for f in $INPUT_FILES
do
  echo "Embedding $f with 0.8 ..."
  ./WOW -i $f -O $OUTPUT_DIR -a 0.8
done
