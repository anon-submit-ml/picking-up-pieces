#!/bin/bash 

#SBATCH --nodes=1
#SBATCH --mem=48G
#SBATCH --partition=dggpu
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00

cd DARTS-space
python train_search.py --data '../../data' --method 'gdas' --seed 10 --dataset 'cifar100' --k 2

