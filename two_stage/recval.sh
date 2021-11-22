#!/bin/bash 

#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --partition=dggpu
#SBATCH --gres=gpu:1
#SBATCH --time=36:00:00

cd 201-space
python record_val.py --data '../../data' --method $method --seed $seed --dataset $dset --s_time $t

