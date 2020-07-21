#!/bin/bash

DatasetPath="$HOME/myworks/stego/datasets"
DatasetName="Mix"

ResizeMode="PILresize"
EmbeddingAlgorithm="s-uniward"
EmbeddingRatio="0.8"

TrainTest="80.20"

OutputPath="results"
ModelName=${DatasetName}_${EmbeddingAlgorithm}_${TrainTest}_${EmbeddingRatio}

python main.py \
	--ctrp $DatasetPath/$DatasetName/cover/$ResizeMode/train \
	--ctep $DatasetPath/$DatasetName/cover/$ResizeMode/test \
	--strp $DatasetPath/$DatasetName/stego/$EmbeddingRatio/${EmbeddingAlgorithm}_${ResizeMode}/train \
	--step $DatasetPath/$DatasetName/stego/$EmbeddingRatio/${EmbeddingAlgorithm}_${ResizeMode}/test \
	--op $OutputPath/$ModelName \
	--shuffle \
	-v 1
