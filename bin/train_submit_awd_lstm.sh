#!/bin/bash
### General options 
#BSUB -q gpuv100
#BSUB -J awd-lstm
#BSUB -n 1 
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=16GB]"
#BSUB -M 16GB
#BSUB -W 23:59 
##BSUB -u s144086@student.dtu.dk
#BSUB -o Output_awd-lstm.out 
#BSUB -e Error_awd-lstm.err 

# required modules
module load cuda/9.0
module load cudnn/v7.4.2.24-prod-cuda-9.0
module load gcc/9.2.0


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

#export PATH=$PATH:/zhome/9e/7/97809/thesis/context-dependent-lanaguage-models/kglm-model/
#export PYTHONPATH=$PYTHONPATH:/zhome/9e/7/97809/thesis/context-dependent-lanaguage-models/kglm-model/
#export PYTHONPATH=${PYTHONPATH}:/zhome/9e/7/97809/apex/

export LD_LIBRARY_PATH=/appl/python/3.6.2/lib/

DEVICE=0
EXPERIMENT=/zhome/9e/7/97809/thesis/kglm-model/experiments/awd-lstm-lm.jsonnet
SAVE_PATH=/work1/s144086/kglm/awd-lstm

if [ -d $SAVE_PATH ]
then
    echo "Serialisation Path Already Exists.. Recovering from previous training session"
    RECOVER=--recover
else
    echo "Starting Training from scratch"
    RECOVER=
fi


CUDA_VISIBLE_DEVICES=$DEVICE allennlp train $EXPERIMENT -s $SAVE_PATH --include-package kglm $RECOVER
