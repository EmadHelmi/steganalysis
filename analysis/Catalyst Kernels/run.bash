#!/bin/bash

DatasetPath="$HOME/myworks/stego/datasets"
DatasetName="BOSSbase_1.01"

ResizeMode="PILresize"
EmbeddingAlgorithm="s-uniward"
EmbeddingRatio="0.8"

TrainTest="80.20"

OutputPath="results"
ModelVersion="1.1.0"
ModelName=${DatasetName}_${EmbeddingAlgorithm}_${TrainTest}_${EmbeddingRatio}_V${ModelVersion}

python main.py \
	--ctrp $DatasetPath/$DatasetName/cover/$ResizeMode/train \
	--ctep $DatasetPath/$DatasetName/cover/$ResizeMode/test \
	--strp $DatasetPath/$DatasetName/stego/$EmbeddingRatio/${EmbeddingAlgorithm}_${ResizeMode}/train \
	--step $DatasetPath/$DatasetName/stego/$EmbeddingRatio/${EmbeddingAlgorithm}_${ResizeMode}/test \
	--model_version $ModelVersion \
	--op $OutputPath/$ModelName \
	--shuffle \
	-v 1
