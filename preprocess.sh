#!/bin/bash
#SBATCH -p IAI_SLURM_3090
#SBATCH -o logs/%j-preprocess.out
#SBATCH -e logs/%j-preprocess.err
#SBATCH -c 8
#SBATCH --gres=gpu:0
#SBATCH --qos=8gpu
#SBATCH -J LooGLE-preprocess
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00


python lift/gated_memory/utils.py

