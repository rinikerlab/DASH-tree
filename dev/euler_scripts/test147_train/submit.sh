sbatch -n 1 --cpus-per-task=48 --time=120:00:00 --job-name="train" --mem-per-cpu=2048 --wrap="python test_058_train.py"
