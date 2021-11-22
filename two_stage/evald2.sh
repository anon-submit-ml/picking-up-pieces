#!/bin/bash 

#SBATCH --nodes=1
#SBATCH --mem=48G
#SBATCH --partition=dggpu
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00

cd DARTS-space
python train.py --data '../../data' --dataset $dset --arch $arch --resume_epoch $rep --s_time $t --resume_lr $rlr --cutout --auxiliary

