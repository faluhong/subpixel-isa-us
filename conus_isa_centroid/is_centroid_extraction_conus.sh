#!/bin/sh
#SBATCH --partition=general  
#SBATCH --account=zhz18039
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=256G
#SBATCH --exclude=cn538,cn523,cn551,cn584,cn585

source /home/fah20002/miniconda3/etc/profile.d/conda.sh
conda activate py38

cd /shared/zhulab/Falu/CSM_project/pythoncode/conus_isp_trajectory/

python3 is_centroid_extraction_conus.py 


