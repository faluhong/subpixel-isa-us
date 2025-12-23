#!/bin/sh
#SBATCH --partition=general  
#SBATCH --account=zhz18039
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=30G
#SBATCH --array=1-25
#SBATCH --no-requeue 
#SBATCH --exclude=cn538,cn523,cn551,cn553

source /home/fah20002/miniconda3/etc/profile.d/conda.sh
conda activate py38

cd /scratch/zhz18039/fah20002/CSM_project/pythoncode/deep_learning_isp/

python3 unet_prediction_parallel.py   --rank=$SLURM_ARRAY_TASK_ID  