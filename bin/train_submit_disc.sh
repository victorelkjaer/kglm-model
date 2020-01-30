#!/bin/bash
### General options 
#BSUB -q gpuv100
#BSUB -J kglm-disc-no-grad-clip
#BSUB -n 1 
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=16GB]"
#BSUB -M 16GB
#BSUB -W 23:59 
##BSUB -u s144086@student.dtu.dk
#BSUB -o Output_no_grad_clip.out 
#BSUB -e Error_no_grad_clip.err 

# required modules
module load cuda/9.0
module load cudnn/v7.4.2.24-prod-cuda-9.0

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


source activate kglm

export LD_LIBRARY_PATH=/appl/python/3.6.2/lib/

# run on a single GPU
DEVICE=0

# set serialisation directory
SAVE_PATH=/work1/s144086/kglm/discriminative_models/disco_diktatoren

if [ -d $SAVE_PATH ]
then
    echo "Serialisation directory already axists.. Recovering from previous training session"
    echo "From: $SAVE_PATH"
    RECOVER=--recover
    EXPERIMENT=$SAVE_PATH/config.json
else
    echo "Starting Training from scratch"
    echo "Saving output to $SAVE_PATH"
    RECOVER=
    EXPERIMENT=/zhome/9e/7/97809/thesis/kglm-model/experiments/kglm-disc.jsonnet
fi

CUDA_VISIBLE_DEVICES=$DEVICE allennlp train $EXPERIMENT -s $SAVE_PATH --include-package kglm $RECOVER
