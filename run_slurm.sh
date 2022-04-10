#!/bin/bash

#SBATCH --partition=gpu           # Partition (job queue)
#SBATCH --requeue                 # Return job to the queue if preempted
#SBATCH --job-name=riboseq        # Assign an short name to your job
#SBATCH --cpus-per-task=1         # Cores per task (>1 if multithread tasks)
#SBATCH --mem-per-cpu=16000M      # Real memory (RAM) required
#SBATCH --gres=gpu:2              # Generic resources
#SBATCH --time=02:00:00           # Total run time limit (HH:MM:SS)
#SBATCH --error=/scratch/vw120/slurm.%N.%j.err
#SBATCH --out=/scratch/vw120/slurm.%N.%j.out
#SBATCH --mail-type=all           # when something happens
#SBATCH --mail-user=vw120@scarletmail.rutgers.edu # send me mail

source /home/vw120/miniconda3/bin/activate
conda activate riboseq

python /home/vw120/nnPUlearning/train.py -g 0 -p exp-gene -D