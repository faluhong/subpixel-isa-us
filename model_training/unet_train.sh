#!/bin/sh
#SBATCH --partition=priority-gpu
#SBATCH --account=zhz18039
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --constraint="a100"


source /home/fah20002/miniconda3/etc/profile.d/conda.sh
conda activate py38

cd /scratch/zhz18039/fah20002/CSM_project/pythoncode/deep_learning_isp/

python3 unet_train.py  