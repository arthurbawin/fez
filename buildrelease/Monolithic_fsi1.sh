#!/bin/bash
#SBATCH --job-name=monolithic_fsi1_3D
#SBATCH --account=def-etienne1
#SBATCH --nodes=9
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=3
#SBATCH --mem-per-cpu=4000M
#SBATCH --time=6-00:00
#SBATCH --output=/home/joan2810/scratch/monolithic_fsi1/log/output.txt
#SBATCH --error=/home/joan2810/scratch/monolithic_fsi1/log/error.txt

srun --cpu-bind=cores ./monolithic_fsi ../data/parameters_file/monolithic_fsi1_3D.prm