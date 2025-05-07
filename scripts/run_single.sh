#!/bin/bash
#SBATCH -c 8  # Number of Cores
#SBATCH --mem=50G  # Requested Memory
#SBATCH -p gpu-preempt  # Partition
#SBATCH -G 1  # Number of GPUs and type
#SBATCH -C vram23  
#SBATCH -t 24:00:00  # Job time limit
#SBATCH -o sbatch_output/slurm-%j.out  # Output file (%j = job ID)
#SBATCH -e sbatch_output/slurm-%j.err

# Activate conda environment
source ~/anaconda3/bin/activate ai4sci

python3 -m ec_blast.main training.alpha=1.0 model.layer_idx=-1 model.fine_tuning_mode=full\
    model.architecture=original training.batch_size=32 training.num_epochs=200 

wait