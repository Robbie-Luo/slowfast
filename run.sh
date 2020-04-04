#!/bin/bash
#SBATCH --job-name=example
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --ntasks-per-node=1
#SBATCH --time=1:00:00
#SBATCH --mem=60000M
#SBATCH --partition=gpu_shared
#SBATCH --gres=gpu:1
module purge
module load 2019
module load eb
module load CUDA/10.0.130
source activate slowfast
export LD_LIBRARY_PATH=/hpc/eb/Debian/cuDNN/7.4.2-CUDA-10.0.130/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=/home/shuogpu/wluo/slowfast:$PYTHONPATH
srun python3 tools/run_net.py