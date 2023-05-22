echo "Started worker"
jobIDX=$1
fileIdx=$((${jobIDX} % 2000))
lineIdx=$((${jobIDX} / 2000 + 1))
smiles="$(sed "${lineIdx}q;d" smiles/sorted_smiles_${fileIdx}.csv)"
echo "bash setup: ${jobIDX}, ${fileIdx}, ${lineIdx}, ${smiles}"

cd ${TMPDIR}
mkdir scratch
export PSI_SCRATCH=${TMPDIR}/scratch
#mkdir sdfs
mkdir outs
#cp ${SLURM_SUBMIT_DIR}/sdfs/${jobIDX}.sdf sdfs/
python ${SLURM_SUBMIT_DIR}/run_psi4.py ${jobIDX} ${smiles}
cp psi4_out_${jobIDX}.dat ${SLURM_SUBMIT_DIR}/outs/
cp sdf_mbis_${jobIDX}.sdf ${SLURM_SUBMIT_DIR}/outs/
#cp outs/psi4_out_${jobIDX}.dat ${SLURM_SUBMIT_DIR}/outs/
#cp outs/sdf_qmugs500_mbis_${jobIDX}.sdf ${SLURM_SUBMIT_DIR}/outs/
echo "all files copied"
