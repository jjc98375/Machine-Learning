#!/bin/bash
#SBATCH --job-name=groupD4_hindi_korean
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --output=slurm/d4_hindi_korean_%j.out

module load anaconda3/2024.06
conda activate MLenv
cd ~/Machine-Learning/finalproject-stepbystep/phase2

python run_experiment.py --epochs 5 --samples_per_pair 8000 --backbones mbert --zero_shot_pairs Hindi-English Korean-English --focal_alpha 0.8 --focal_gamma 2.0
