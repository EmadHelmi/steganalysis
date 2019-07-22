# BOWS2OrigEp3 0.1
INPUT_FILES=~/stego/datasets/BOWS2OrigEp3/total/*.pgm
OUTPUT_DIR="/root/stego/datasets/BOWS2OrigEp3/stego/wow/0.1"
for f in $INPUT_FILES
do
  echo "Embedding $f with 0.1 ..."
  ./WOW -i $f -O $OUTPUT_DIR -a 0.1
done

# BOWS2OrigEp3 0.2
INPUT_FILES=~/stego/datasets/BOWS2OrigEp3/total/*.pgm
OUTPUT_DIR="/root/stego/datasets/BOWS2OrigEp3/stego/wow/0.2"
for f in $INPUT_FILES
do
  echo "Embedding $f with 0.2 ..."
  ./WOW -i $f -O $OUTPUT_DIR -a 0.2
done

# BOWS2OrigEp3 0.4
INPUT_FILES=~/stego/datasets/BOWS2OrigEp3/total/*.pgm
OUTPUT_DIR="/root/stego/datasets/BOWS2OrigEp3/stego/wow/0.4"
for f in $INPUT_FILES
do
  echo "Embedding $f with 0.4 ..."
  ./WOW -i $f -O $OUTPUT_DIR -a 0.4
done
