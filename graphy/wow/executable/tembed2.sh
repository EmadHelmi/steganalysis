#!/bin/bash
# BOWS2OrigEp3 0.05
INPUT_FILES=/home/emadhelmi/myworks/stego/datasets/BOSSbase_1.01/total/PILresize/test/*
OUTPUT_DIR="/home/emadhelmi/myworks/stego/datasets/BOSSbase_1.01/stego/0.1/wow_PILresize/test"
for f in $INPUT_FILES
do
  echo "Embedding $f test with 0.1 ..."
  ./WOW -i $f -O $OUTPUT_DIR -a 0.1
done
