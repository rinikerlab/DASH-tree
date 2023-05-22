#!/bin/bash

bsub -n 1 -o logfiles/extraction.out -e logfiles/extraction.err -W 120:00 -J "ext[1-9966]" "./worker.sh  ./GNN_lr_0.00010000_batch_64_seed_1_model_sd.pt /cluster/work/igc/mlehner/test156_explain/sdf_data" > id.txt
