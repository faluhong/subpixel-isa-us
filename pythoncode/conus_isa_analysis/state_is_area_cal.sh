#!/bin/sh
#SBATCH --partition=priority  
#SBATCH --account=zhz18039
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=96G
#SBATCH --array=1-20
#SBATCH --exclude=cn538,cn523,cn551,cn584,cn585

source /home/fah20002/miniconda3/etc/profile.d/conda.sh
conda activate py38

cd /shared/zhulab/Falu/CSM_project/pythoncode/conus_isp_analysis/

python3 state_is_area_cal.py  --n_cores=$SLURM_ARRAY_TASK_MAX  --rank=$SLURM_ARRAY_TASK_ID  











