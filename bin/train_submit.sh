#!/bin/bash
if [ "$#" -ne 3 && $4 -ne "--recover" ]; then
    echo ""
    echo "---------------------------------------------"
	echo "Please provide the following three arguments:"
	echo "[cuda device number]"
	echo "[path to .json config]"
	echo "[path to save checkpoint to]"
    echo "---------------------------------------------"
    echo ""
	exit 3
fi

CUDA_VISIBLE_DEVICES=$1 allennlp train $2 -s $3 --include-package kglm $4
