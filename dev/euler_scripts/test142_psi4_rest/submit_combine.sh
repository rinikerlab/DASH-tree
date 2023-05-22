bsub -n 2 -W 24:00 -R "rusage[mem=2048]" -J "comb" "python combine.py"
