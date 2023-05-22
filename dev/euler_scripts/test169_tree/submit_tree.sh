#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=128
#SBATCH --time=120:00:00
#SBATCH --job-name="treP2"
#SBATCH --mem-per-cpu=8000
#SBATCH --tmp=50000
#SBATCH --output="tree.out"
#SBATCH --error="tree.err"
#SBATCH --open-mode=append

#cp ../test154_explain/combined.csv ${TMPDIR}
cp /cluster/work/igc/mlehner/test159_tree/cleaned_df.csv ${TMPDIR}/combined.csv
#cp ../test154_explain/sdf_explain.sdf ${TMPDIR}/combined_multi.sdf
cp /cluster/work/igc/mlehner/test146_combine/combined_multi.sdf ${TMPDIR}/combined_multi.sdf
cd ${TMPDIR}
mkdir test_009_out
cd test_009_out
mkdir tree
mkdir tree_pruned
cd ${TMPDIR}
echo " all copies done. start tree build script ... "

python /cluster/work/igc/mlehner/test169_tree/build_tree.py

echo " all tree building script done, start cleanup ..."
cp -R test_009_out /cluster/work/igc/mlehner/test169_tree/
cp ${TMPDIR}/* /cluster/work/igc/mlehner/test169_tree/test_009_out/
