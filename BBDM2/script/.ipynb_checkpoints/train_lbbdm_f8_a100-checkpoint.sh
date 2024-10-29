#!/bin/bash
# SBATCH --job-name=train_nmduong        # Job name
#SBATCH --output=result_nmduong.txt      # Output file
#SBATCH --error=error_nmduong.txt        # Error file

#SBATCH --ntasks=1               # Number of tasks (processes)
#SBATCH --cpus-per-task=8        # Number of CPU cores per task
#SBATCH --mem=16G                 # Memory per node (4 GB)
#SBATCH --gpus=1                 # Number of GPUs per node

# Set the number of threads
export OMP_NUM_THREADS=4

# Set the GPU index
# export CUDA_VISIBLE_DEVICES=1

# Load any necessary modules or set environment variables here
# For example:
module load cuda/11.4  # Load CUDA if necessary
# module load your_gpu_module

# Activate your conda environment if needed
# source activate my_env

# Run your application or command
# conda env list
# pip install -r requirements.txt
source /home/user01/.bashrc
conda init
echo "Start to activate nmduong"
conda activate BBDM

python3 main.py \
    --config configs/Template-LBBDM-f8-a100.yaml \
    --train \
    --sample_at_start \
    --save_top \
    --gpu_ids 0