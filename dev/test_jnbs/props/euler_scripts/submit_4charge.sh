# 36112
# 100171 // 100 = 1002
sbatch -n 1 --cpus-per-task=4 --time=120:00:00 --job-name="am1bcc" --array=1-1012 --mem-per-cpu=6000 --wrap="python run_am1bcc.py \$SLURM_ARRAY_TASK_ID"
