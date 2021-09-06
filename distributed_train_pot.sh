#!/bin/bash
NUM_PROC=$1
PORT=${PORT:-29501}
shift
python3 -m torch.distributed.launch --nproc_per_node=$NUM_PROC --master_port=$PORT main.py "$@"

