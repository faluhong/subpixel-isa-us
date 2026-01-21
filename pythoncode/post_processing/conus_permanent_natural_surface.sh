#!/bin/sh
#SBATCH --partition=priority  
#SBATCH --account=zhz18039
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --no-requeue 

source /home/fah20002/miniconda3/etc/profile.d/conda.sh
conda activate py38

cd /scratch/zhz18039/fah20002/CSM_project/pythoncode/post_processing/

python3 conus_permanent_natural_surface.py   