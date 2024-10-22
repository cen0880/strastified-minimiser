#!/bin/bash
#SBATCH -J eigen_J-I
#SBATCH -p cpu 
#SBATCH --cpus-per-task=32
#SBATCH -N 1
#SBATCH -t 168:00:00
#SBATCH -o slurm_J-I_eigen.out
#SBATCH -e slurm_J-I_eigen.err
#module add anaconda/3
. /public/software/anaconda/3/etc/profile.d/conda.sh
conda activate fenics

hostname

export OPENBLAS_NUM_THREADS=64
export OMP_NUM_THREADS=64

ulimit -a
ulimit -u 1024
python eigen_W_square_without_delta_mess_second.py
