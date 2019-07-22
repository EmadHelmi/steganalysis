#!/bin/bash
# BOSSbase 0.05
INPUT_FILES=~/stego/datasets/BOSSbase_1.01/total/*.pgm
OUTPUT_DIR="/root/stego/datasets/BOSSbase_1.01/stego/s-uniward/0.05"
for f in $INPUT_FILES
do
  echo "Embedding $f with 0.05 ..."
  ./S-UNIWARD -i $f -O $OUTPUT_DIR -a 0.05
done

# BOSSbase 0.1
INPUT_FILES=~/stego/datasets/BOSSbase_1.01/total/*.pgm
OUTPUT_DIR="/root/stego/datasets/BOSSbase_1.01/stego/s-uniward/0.1"
for f in $INPUT_FILES
do
  echo "Embedding $f with 0.1 ..."
  ./S-UNIWARD -i $f -O $OUTPUT_DIR -a 0.1
done

# BOSSbase 0.2
INPUT_FILES=~/stego/datasets/BOSSbase_1.01/total/*.pgm
OUTPUT_DIR="/root/stego/datasets/BOSSbase_1.01/stego/s-uniward/0.2"
for f in $INPUT_FILES
do
  echo "Embedding $f with 0.2 ..."
  ./S-UNIWARD -i $f -O $OUTPUT_DIR -a 0.2
done

# BOSSbase 0.4
INPUT_FILES=~/stego/datasets/BOSSbase_1.01/total/*.pgm
OUTPUT_DIR="/root/stego/datasets/BOSSbase_1.01/stego/s-uniward/0.4"
for f in $INPUT_FILES
do
  echo "Embedding $f with 0.4 ..."
  ./S-UNIWARD -i $f -O $OUTPUT_DIR -a 0.4
done
