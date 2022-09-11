#!/bin/bash
#SBATCH --job-name mpi-go-neural-network
#SBATCH -N 1
#SBATCH -p fat
#SBATCH -n 20
#SBATCH --time=01:30:00

module purge
module load openmpi

mpirun -n 20 ./goai




