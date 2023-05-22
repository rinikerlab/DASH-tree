#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --job-name="att"
#SBATCH --mem-per-cpu=16384
#SBATCH --tmp=50G
#SBATCH --array=1-100

mkdir ${TMPDIR}/tree
cp ../test169_tree/test_009_out/tree/tree*.pkl ${TMPDIR}/tree/
cp test.sdf ${TMPDIR}
cp test_092_attThresh.py ${TMPDIR}

cd ${TMPDIR}
python test_092_attThresh.py $SLURM_ARRAY_TASK_ID

cp *.csv /cluster/work/igc/mlehner/test171_attThresh
