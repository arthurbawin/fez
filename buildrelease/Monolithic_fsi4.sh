#!/bin/bash
#SBATCH --job-name=monolithic_fsi4_3D
#SBATCH --account=def-etienne1
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=3
#SBATCH --mem-per-cpu=4000M
#SBATCH --time=4-00:00
#SBATCH --output=/home/joan2810/scratch/monolithic_fsi4/log/log.%t.txt
#SBATCH --error=/home/joan2810/scratch/monolithic_fsi4/log/log.%t.txt

srun --cpu-bind=cores ./monolithic_fsi ../data/parameters_file/monolithic_fsi4_3D.prm