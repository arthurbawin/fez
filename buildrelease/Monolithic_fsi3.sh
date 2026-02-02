#!/bin/bash
#SBATCH --job-name=monolithic_fsi3_3D
#SBATCH --account=def-etienne1
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=3
#SBATCH --mem-per-cpu=4000M
#SBATCH --time=6-00:00
#SBATCH --output=/home/joan2810/scratch/monolithic_fsi3/log/output.txt
#SBATCH --error=/home/joan2810/scratch/monolithic_fsi3/log/error.txt

export OMP_NUM_THREADS=3
srun --cpu-bind=cores ./monolithic_fsi ../data/parameters_file/monolithic_fsi3_3D.prm
