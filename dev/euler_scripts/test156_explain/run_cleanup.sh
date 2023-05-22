#!/bin/bash

bsub -n 1 -J 'clean_up' -o logfiles/cleanup.out -e logfiles/cleanup.err -w 'done(256439411)' './cleaner.sh 9966 108 /cluster/work/igc/mlehner/test146_combine/combined_multi.sdf'
