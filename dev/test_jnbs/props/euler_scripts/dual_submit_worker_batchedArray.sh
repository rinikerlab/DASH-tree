#!/bin/bash
#SBATCH -n 1
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --job-name='dual'
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=8192
#SBATCH --tmp=20000
#SBATCH --output="dual/dual.%A.%a.out"
#SBATCH --error="dual/dual.%A.%a.err"
#SBATCH --array=1-10000

max_molJobIndex=180000
min_molJobIndex=1

cd ${TMPDIR}
mkdir scratch
export PSI_SCRATCH=${TMPDIR}/scratch

molJobIndicesPerArrayJob=18
start_molJobIndex=$(($min_molJobIndex + ($SLURM_ARRAY_TASK_ID-1)*$molJobIndicesPerArrayJob))
end_molJobIndex=$(($start_molJobIndex + $molJobIndicesPerArrayJob - 1))

for ((molJobIndex=$start_molJobIndex; molJobIndex<=$end_molJobIndex; molJobIndex++))
do
    echo "molJobIndex: $molJobIndex"
    if [ ! -f "${SLURM_SUBMIT_DIR}/dual/dual_${molJobIndex}.pkl" ]; then
        python ${SLURM_SUBMIT_DIR}/run_psi4_dual.py $molJobIndex
        # clear scratch
        rm -rf ${TMPDIR}/scratch/*
    else
        echo "dual_${molJobIndex}.pkl already exists"
    fi
done

echo "Finished with array job: $SLURM_ARRAY_JOB_ID, task: $SLURM_ARRAY_TASK_ID"
