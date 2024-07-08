echo "Started worker"
jobIDX=$(($1 + 0))
echo "bash setup: ${jobIDX}"

cd ${TMPDIR}
mkdir scratch
export PSI_SCRATCH=${TMPDIR}/scratch
python ${SLURM_SUBMIT_DIR}/run_psi4_dual.py ${jobIDX}
echo "all done dual worker"
