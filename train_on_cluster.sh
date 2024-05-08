#!/bin/bash

PROC_N=$1
if [ -z $PROC_N ]; then
    PROC_N=1
fi
JULIA=$(which julia)
PROJECT_DIR=/home/eugene/Programming/dnu/swm
EXEC_COMMAND="$JULIA --project=$PROJECT_DIR src/train_distributed.jl src/notebooks/hnode_vocoder_params.bson"

mpiexec \
    --mca btl_tcp_if_include 192.168.0.0/24 \
    --hostfile hostfile \
    --wdir $PROJECT_DIR \
    -np $PROC_N \
    $EXEC_COMMAND
