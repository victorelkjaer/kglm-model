#!/bin/bash
if [ "$#" -ne 3 && $4 -ne "--recover" ]; then
    echo ""
    echo "---------------------------------------------"
	echo "Please provide the following three arguments:"
	echo "[cuda device number]"
	echo "[path to model archive [.tar.gz] file]"
    echo "[path to test dataset (.jsonl)]"
    echo "---------------------------------------------"
    echo ""
	exit 3
fi

CUDA_VISIBLE_DEVICES=$1 python -m kglm.run complete-the-sentence $2 $3 --include-package kglm
