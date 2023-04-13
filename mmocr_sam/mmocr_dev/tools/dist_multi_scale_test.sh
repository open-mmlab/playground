#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
OUT_FOLDER=$4
NNODES=-1
NODE_RANK=-0
PORT=-29500
MASTER_ADDR="127.0.0.1"

python -m torch.distributed.launch \
    --nproc_per_node=$GPUS \
    tools/test.py \
    $CONFIG \
    $CHECKPOINT \
    --work-dir $OUT_FOLDER \
    --show-dir $OUT_FOLDER \
    --save-preds \
    --launcher pytorch \