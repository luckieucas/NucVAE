#!/bin/bash
#SBATCH --job-name=train_VAE   # Job name
#SBATCH --time=96:00:00           # Maximum runtime
#SBATCH --nodes=1                 # Number of nodes
#SBATCH --ntasks=1                # Number of tasks
#SBATCH --cpus-per-task=6         # Number of CPU cores per task
#SBATCH --gpus=1                  # Number of GPUs per task
#SBATCH --mem=32gb                # Memory allocation
#SBATCH --partition=gpua100,weidf,gpuv100       # Partition to submit the job
#SBATCH --mail-type=BEGIN,END,FAIL  # Email notifications (on job start, end, or fail)
#SBATCH --mail-user=liupen@bc.edu  # Email address for notifications
#SBATCH --output=/mmfs1/data/liupen/project/NucleiVae/log/train_vae_%j.out     # Output log file (%j represents the Job ID)

# Activate the Conda environment
source ~/anaconda3/bin/activate
conda activate sam2_mito

# Print debug information
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "Start time: $(date)"

# Run the nnUNet training command
python -u /mmfs1/data/liupen/project/NucleiVae/train.py -c $config -m $model

# Print end time
echo "Training completed at $(date)"
