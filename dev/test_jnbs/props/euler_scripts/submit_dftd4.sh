#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --job-name="dftd4"
#SBATCH --mem-per-cpu=8000
#SBATCH --tmp=50000
#SBATCH --output="dftd4.out"
#SBATCH --error="dftd4.err"
#SBATCH --open-mode=append

cp /cluster/work/igc/mlehner/test146_combine/combined_multi.sdf ${TMPDIR}/combined_multi.sdf
cd ${TMPDIR}
echo " all copies done. start script ... "

python /cluster/work/igc/mlehner/test170_dftd4/run_dft_d4.py

echo " all script done, start cleanup ..."
cp mols_comb_dftd4.sdf /cluster/work/igc/mlehner/test170_dftd4/
