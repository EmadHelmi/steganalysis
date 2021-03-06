#!/bin/bash

DATASET="BOWS2OrigEp3"

# $DATASET/stego/0.1
INPUT_FILES=/home/emadhelmi/myworks/stego/datasets/$DATASET/cover/PILresize/test/*.pgm
OUTPUT_DIR="/home/emadhelmi/myworks/stego/datasets/$DATASET/stego/0.1/s-uniward_PILresize/test"
mkdir -p $OUTPUT_DIR
for f in $INPUT_FILES
do
  echo "Embedding $f test with 0.1 ..."
  ./S-UNIWARD -i $f -O $OUTPUT_DIR -a 0.1
done

# $DATASET/stego/0.2
INPUT_FILES=/home/emadhelmi/myworks/stego/datasets/$DATASET/cover/PILresize/test/*.pgm
OUTPUT_DIR="/home/emadhelmi/myworks/stego/datasets/$DATASET/stego/0.2/s-uniward_PILresize/test"
mkdir -p $OUTPUT_DIR
for f in $INPUT_FILES
do
  echo "Embedding $f test with 0.2 ..."
  ./S-UNIWARD -i $f -O $OUTPUT_DIR -a 0.2
done


# $DATASET/stego/0.4
INPUT_FILES=/home/emadhelmi/myworks/stego/datasets/$DATASET/cover/PILresize/test/*.pgm
OUTPUT_DIR="/home/emadhelmi/myworks/stego/datasets/$DATASET/stego/0.4/s-uniward_PILresize/test"
mkdir -p $OUTPUT_DIR
for f in $INPUT_FILES
do
  echo "Embedding $f test with 0.4 ..."
  ./S-UNIWARD -i $f -O $OUTPUT_DIR -a 0.4
done


# $DATASET/stego/0.8
INPUT_FILES=/home/emadhelmi/myworks/stego/datasets/$DATASET/cover/PILresize/test/*.pgm
OUTPUT_DIR="/home/emadhelmi/myworks/stego/datasets/$DATASET/stego/0.8/s-uniward_PILresize/test"
mkdir -p $OUTPUT_DIR
for f in $INPUT_FILES
do
  echo "Embedding $f test with 0.8 ..."
  ./S-UNIWARD -i $f -O $OUTPUT_DIR -a 0.8
done

