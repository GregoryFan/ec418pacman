#!/bin/bash -l

#$ -P ec523bn    #Project Name
#$ -N pacDQNTrain   #Job Name
#$ -l h_rt=48:00:00     #Job Cut-Off Time
#$ -m ea
#$ -j y #Merge Error and Output
#$ -o results 
#$ -l gpus=1 
#$ -l gpu_c=7.0
#$ -pe omp 4

module load python3/3.8.10
module load cuda/11.8
module load pytorch/1.13.1

source ~/pacman/pac_env/bin/activate

echo "Using Python at: $(which python)"
python -c "import torch; print('Torch version:', torch.__version__)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"


python3 train.py
