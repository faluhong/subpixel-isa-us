#!/bin/sh
#SBATCH --partition=general  
#SBATCH --account=zhz18039
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=30G
#SBATCH --array=1-100
#SBATCH --exclude=cn538,cn523,cn551

source /home/fah20002/miniconda3/etc/profile.d/conda.sh
conda activate py38

cd /scratch/zhz18039/fah20002/CSM_project/pythoncode/deep_learning_isp/

python3 conus_unet_prediction_parallel.py  --n_cores=$SLURM_ARRAY_TASK_MAX  --rank=$SLURM_ARRAY_TASK_ID  

# python3 conus_unet_prediction_parallel.py