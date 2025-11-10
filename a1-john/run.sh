#! /bin/bash

#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu

# shared environment not working on cluster, activating local venv
source ~/.envs/a1/bin/activate

python3 main.py