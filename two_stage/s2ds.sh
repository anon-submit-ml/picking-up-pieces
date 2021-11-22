#!/bin/bash 

#SBATCH --nodes=1
#SBATCH --mem=48G
#SBATCH --partition=dggpu
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00

cd DARTS-space
python search2.py --data '../../data' --method $method --seed $seed --dataset $dset --s_time $t

