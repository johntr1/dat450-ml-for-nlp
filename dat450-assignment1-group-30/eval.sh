#! /bin/bash

#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu

# shared environment not working on cluster, activating local venv
source ~/.envs/a1/bin/activate
#source /data/courses/2025_dat450_dit247/venvs/dat450_venv/bin/activate

python3 -u eval.py