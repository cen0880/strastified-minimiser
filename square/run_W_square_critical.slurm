#!/bin/bash
#SBATCH -J critical_J-I_index_3
#SBATCH -p cpu 
#SBATCH --cpus-per-task=32
#SBATCH -N 1
#SBATCH -t 168:00:00
#SBATCH -o slurm_J-I_critical_index_3.out
#SBATCH -e slurm_J-I_critical_index_3.err
#module add anaconda/3
. /public/software/anaconda/3/etc/profile.d/conda.sh
conda activate fenics

hostname
python critical_W_square_without_delta_mess_second.py
