


# general paths
GENE_MODEL_ARCHIVE=/work1/s144086/kglm/generative_models/
DISCO_MODEL_ARCHIVE=/work1/s144086/kglm/discriminative_models/

# set run names
GENE_RUN_NAME=generative_standard
DISCO_RUN_NAME=disco

# don't change
GENE_MODEL_ARCHIVE=${GENE_MODEL_ARCHIVE}${GENE_RUN_NAME}/model.tar.gz
DISCO_MODEL_ARCHIVE=${DISCO_MODEL_ARCHIVE}${DISCO_RUN_NAME}/model.tar.gz

# data to evaluate perplexity upon
HELD_OUT_DATA_PATH=/work1/s144086/kglm/linked-wikitext-2/valid.jsonl


source activate kglm

python -m kglm.run evaluate-perplexity \
                   $GENE_MODEL_ARCHIVE \
                   $DISCO_MODEL_ARCHIVE \
                   $HELD_OUT_DATA_PATH \
                   --include-package kglm
