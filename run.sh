#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=python_example
module load Python/Anaconda_v10.2019
python main.py models.simplest -steps=10