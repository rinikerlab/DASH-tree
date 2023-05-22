# 36112
# 100171 // 100 = 1002
sbatch -n 1 --cpus-per-task=4 --time=24:00:00 --job-name="4chg" --array=1-1002 --mem-per-cpu=16384 --wrap="python test_068_charge4_worker.py \$SLURM_ARRAY_TASK_ID"
