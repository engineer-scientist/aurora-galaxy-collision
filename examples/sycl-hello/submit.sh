#!/bin/bash
#PBS -N sycl-vecadd
#PBS -A DD_plasma_physics
#PBS -q debug
#PBS -l select=1
#PBS -l walltime=00:10:00
#PBS -l filesystems=home:flare
#PBS -j oe

cd "$PBS_O_WORKDIR" || exit 1

./vecadd

# To put one rank on each of the 12 tiles instead:
#   mpiexec -n 12 -ppn 12 gpu_tile_compact.sh ./vecadd
