#!/bin/bash
#SBATCH --job-name=groupD5_full_supervised
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --output=slurm/d5_full_supervised_%j.out

module load anaconda3/2024.06
conda activate MLenv
cd ~/Machine-Learning/finalproject-stepbystep/phase2

python run_experiment.py --epochs 5 --samples_per_pair 10000 --backbones mbert --focal_alpha 0.8 --focal_gamma 2.0
