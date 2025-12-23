#!/bin/sh
#SBATCH --partition=priority  
#SBATCH --account=zhz18039
#SBATCH --ntasks=12
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --no-requeue 

source /home/fah20002/miniconda3/etc/profile.d/conda.sh
conda activate py38

cd /scratch/zhz18039/fah20002/CSM_project/pythoncode/deep_learning_isp/

python3 add_topography_in_training.py   