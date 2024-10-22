#!/bin/bash
#SBATCH -J critical_Tt_index_1
#SBATCH -p cpu 
#SBATCH --cpus-per-task=16
#SBATCH -N 1
#SBATCH -t 168:00:00
#SBATCH -o slurm_Tt_critical_index_1.out
#SBATCH -e slurm_Tt_critical_index_1.err
#module add anaconda/3
. /public/software/anaconda/3/etc/profile.d/conda.sh
conda activate fenics

hostname
python critical_W_hexagon_without_delta_mess_second.py
