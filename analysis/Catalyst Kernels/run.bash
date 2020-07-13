#!/bin/bash

DatasetPath="$HOME/myworks/stego/datasets"
DatasetName="BOSSbase_1.01"

ResizeMode="PILresize"
EmbeddingAlgorithm="wow"
EmbeddingRatio="0.8"

TrainTest="80.20"

OutputPath="results"
ModelName=${DatasetName}_${TrainTest}_${EmbeddingRatio}

python main.py \
	--ctrp $DatasetPath/$DatasetName/cover/$ResizeMode/train \
	--ctep $DatasetPath/$DatasetName/cover/$ResizeMode/test \
	--strp $DatasetPath/$DatasetName/stego/$EmbeddingRatio/${EmbeddingAlgorithm}_${ResizeMode}/train \
	--step $DatasetPath/$DatasetName/stego/$EmbeddingRatio/${EmbeddingAlgorithm}_${ResizeMode}/test \
	--op $OutputPath/$ModelName \
	--shuffle -v
