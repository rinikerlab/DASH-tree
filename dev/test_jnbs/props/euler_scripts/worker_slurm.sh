echo "Started worker"
jobIDX=$1
echo "bash setup: ${jobIDX} " #, ${fileIdx}, ${lineIdx}, ${smiles}"

cd ${TMPDIR}
mkdir scratch
export PSI_SCRATCH=${TMPDIR}/scratch
mkdir sdfs
#mkdir outs
cp ${SLURM_SUBMIT_DIR}/sdfs/${jobIDX}.sdf sdfs/
python ${SLURM_SUBMIT_DIR}/run_psi4.py ${jobIDX}
cp psi4_out_${jobIDX}.dat ${SLURM_SUBMIT_DIR}/out_dat/
cp sdf_mbis_${jobIDX}.sdf ${SLURM_SUBMIT_DIR}/out_sdf/
cp wfn_${jobIDX}_* ${SLURM_SUBMIT_DIR}/wfn/
echo "all files copied"
