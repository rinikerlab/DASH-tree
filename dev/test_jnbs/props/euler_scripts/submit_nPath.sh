#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=6
#SBATCH --time=120:00:00
#SBATCH --job-name="nPath2"
#SBATCH --mem-per-cpu=16384
#SBATCH --output="nPath.out"
#SBATCH --error="nPath.err"
#SBATCH --open-mode=append

#CH --dependency=after:29788053
#8192
#2
#python combine_props.py
python node_path_for_atoms.py
