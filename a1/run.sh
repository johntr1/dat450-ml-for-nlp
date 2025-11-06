#! /bin/bash

#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu
source source /data/courses/2025_dat450_dit247/venvs/dat450_venv/bin/activate

python3 test.py