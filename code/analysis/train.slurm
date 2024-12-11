#!/bin/bash
#SBATCH -J load-ddfuser-grid
#SBATCH -o res/balance_loss.txt
#SBATCH -c 32
#SBATCH --gres=gpu:V100:1
#SBATCH --exclude dell-gpu-23
source /home/LAB/anaconda3/bin/activate /home/LAB/terigen/.conda/envs/grid_ddff
export PYTHONPATH=/home/LAB/terigen/mod-decision-diffuser/code/:/home/LAB/terigen/grid2op_mod/
python train.py
