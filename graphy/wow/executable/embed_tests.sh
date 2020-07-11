#!/bin/bash

# BOSSbase_1.01/stego/0.1
INPUT_FILES=/home/emadhelmi/myworks/stego/datasets/BOSSbase_1.01/cover/PILresize/test/*
OUTPUT_DIR="/home/emadhelmi/myworks/stego/datasets/BOSSbase_1.01/stego/0.1/wow_PILresize/test"
mkdir -p $OUTPUT_DIR
for f in $INPUT_FILES
do
  echo "Embedding $f test with 0.1 ..."
  ./WOW -i $f -O $OUTPUT_DIR -a 0.1
done

# BOSSbase_1.01/stego/0.2
INPUT_FILES=/home/emadhelmi/myworks/stego/datasets/BOSSbase_1.01/cover/PILresize/test/*
OUTPUT_DIR="/home/emadhelmi/myworks/stego/datasets/BOSSbase_1.01/stego/0.2/wow_PILresize/test"
mkdir -p $OUTPUT_DIR
for f in $INPUT_FILES
do
  echo "Embedding $f test with 0.2 ..."
  ./WOW -i $f -O $OUTPUT_DIR -a 0.2
done


# BOSSbase_1.01/stego/0.4
INPUT_FILES=/home/emadhelmi/myworks/stego/datasets/BOSSbase_1.01/cover/PILresize/test/*
OUTPUT_DIR="/home/emadhelmi/myworks/stego/datasets/BOSSbase_1.01/stego/0.4/wow_PILresize/test"
mkdir -p $OUTPUT_DIR
for f in $INPUT_FILES
do
  echo "Embedding $f test with 0.4 ..."
  ./WOW -i $f -O $OUTPUT_DIR -a 0.4
done


# BOSSbase_1.01/stego/0.8
INPUT_FILES=/home/emadhelmi/myworks/stego/datasets/BOSSbase_1.01/cover/PILresize/test/*
OUTPUT_DIR="/home/emadhelmi/myworks/stego/datasets/BOSSbase_1.01/stego/0.8/wow_PILresize/test"
mkdir -p $OUTPUT_DIR
for f in $INPUT_FILES
do
  echo "Embedding $f test with 0.8 ..."
  ./WOW -i $f -O $OUTPUT_DIR -a 0.8
done

