#!/bin/bash

DATASET="BOWS2OrigEp3"

# $DATASET/stego/0.1
INPUT_FILES=/home/emadhelmi/myworks/stego/datasets/$DATASET/cover/PILresize/train/*
OUTPUT_DIR="/home/emadhelmi/myworks/stego/datasets/$DATASET/stego/0.1/wow_PILresize/train"
mkdir -p $OUTPUT_DIR
for f in $INPUT_FILES
do
  echo "Embedding $f train with 0.1 ..."
  ./WOW -i $f -O $OUTPUT_DIR -a 0.1
done


# $DATASET/stego/0.2
INPUT_FILES=/home/emadhelmi/myworks/stego/datasets/$DATASET/cover/PILresize/train/*
OUTPUT_DIR="/home/emadhelmi/myworks/stego/datasets/$DATASET/stego/0.2/wow_PILresize/train"
mkdir -p $OUTPUT_DIR
for f in $INPUT_FILES
do
  echo "Embedding $f train with 0.2 ..."
  ./WOW -i $f -O $OUTPUT_DIR -a 0.2
done


# $DATASET/stego/0.4
INPUT_FILES=/home/emadhelmi/myworks/stego/datasets/$DATASET/cover/PILresize/train/*
OUTPUT_DIR="/home/emadhelmi/myworks/stego/datasets/$DATASET/stego/0.4/wow_PILresize/train"
mkdir -p $OUTPUT_DIR
for f in $INPUT_FILES
do
  echo "Embedding $f train with 0.4 ..."
  ./WOW -i $f -O $OUTPUT_DIR -a 0.4
done


# $DATASET/stego/0.8
INPUT_FILES=/home/emadhelmi/myworks/stego/datasets/$DATASET/cover/PILresize/train/*
OUTPUT_DIR="/home/emadhelmi/myworks/stego/datasets/$DATASET/stego/0.8/wow_PILresize/train"
mkdir -p $OUTPUT_DIR
for f in $INPUT_FILES
do
  echo "Embedding $f train with 0.8 ..."
  ./WOW -i $f -O $OUTPUT_DIR -a 0.8
done