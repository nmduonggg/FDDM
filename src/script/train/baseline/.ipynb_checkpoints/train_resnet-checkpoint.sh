#!/bin/bash
#!/bin/bash
#!/bin/bash
# SBATCH --job-name=nmduong_train_resnet_baseline        # Job name
#SBATCH --output=result_nmduong_train_resnet_baseline.txt      # Output file
#SBATCH --error=error_nmduong_train_vit_baselin.txt        # Error file
#SBATCH --ntasks=1               # Number of tasks (processes)
#SBATCH --cpus-per-task=8        # Number of CPU cores per task
#SBATCH --mem=16G                 # Memory per node (4 GB)
#SBATCH --gpus=1                 # Number of GPUs per node

# Set the number of threads
export OMP_NUM_THREADS=4
export LD_LIBRARY_PATH=/home/user01/miniconda3/envs/glamm/lib:$LD_LIBRARY_PATH
export PYTHONPATH="mccv:$PYTHONPATH"
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
echo "Start to activate glamm"
conda activate glamm_2
# conda list
# conda env list
# # pip install -r requirements.txt
# # Check GPU status
nvidia-smi


python train_classifier.py \
    -opt options/ResNet_baseline.yml \