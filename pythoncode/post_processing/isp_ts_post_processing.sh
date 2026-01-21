#!/bin/sh
#SBATCH --partition=priority  
#SBATCH --account=zhz18039
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=6G
#SBATCH --array=1-110
#SBATCH --exclude=cn538,cn523,cn551
#SBATCH --time=02-00:00:00  # Job should run for up to 2 days       

source /home/fah20002/miniconda3/etc/profile.d/conda.sh
conda activate py38

cd /shared/zhulab/Falu/CSM_project/pythoncode/post_processing/

python3 isp_ts_post_processing.py  --n_cores=$SLURM_ARRAY_TASK_MAX  --rank=$SLURM_ARRAY_TASK_ID  