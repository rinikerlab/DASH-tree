#sbatch -n 1 --cpus-per-task=4 --time=4:00:00 --mem-per-cpu=4096 --wrap="python extractor.py"
bsub -n 4 -W 4:00 -R "rusage[mem=4096]" 'python extractor.py'
