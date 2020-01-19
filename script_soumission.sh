#!/bin/bash
#PBS -N python3 -m src.cars.main
#PBS -A stcar123
#PBS -l walltime=300
#PBS -l nodes=1:gpus=2
cd "${PBS_O_WORKDIR}"
