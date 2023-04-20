#!/bin/sh
#
# Script for training PPC models to do the firefly task.
#
#SBATCH --account=theory         # Replace ACCOUNT with your group account name
#SBATCH --job-name=firefly       # The job name.
#SBATCH -c 4                     # The number of cpu cores to use
#SBATCH -N 1                     # The number of nodes to use
#SBATCH -t 0-02:00               # Runtime in D-HH:MM
#SBATCH --mem-per-cpu=5gb        # The memory the job will use per cpu core

module load anaconda
# pip install torchvision
# conda install pytorch torchvision torchaudio -c pytorch
# conda install -c conda-forge numpy
# conda install -c conda-forge scipy
# conda install -c conda-forge scikit-learn
# conda install -c conda-forge matplotlib

#Command to execute Python program
#python main_batch.py 'firefly-baseline' $SLURM_ARRAY_TASK_ID
#python main_batch.py 'firefly-feedpos' $SLURM_ARRAY_TASK_ID
#python main_batch.py 'firefly-feedbelief' $SLURM_ARRAY_TASK_ID

#python main_batch.py 'firefly-baseline-pn' $SLURM_ARRAY_TASK_ID
#python main_batch.py 'firefly-feedpos-pn' $SLURM_ARRAY_TASK_ID
#python main_batch.py 'firefly-feedbelief-pn' $SLURM_ARRAY_TASK_ID

#python main_batch.py 'firefly-baseline-sn' $SLURM_ARRAY_TASK_ID
#python main_batch.py 'firefly-feedpos-sn' $SLURM_ARRAY_TASK_ID
#python main_batch.py 'firefly-feedbelief-sn' $SLURM_ARRAY_TASK_ID

#python main_batch.py 'firefly-baseline-db' $SLURM_ARRAY_TASK_ID
#python main_batch.py 'firefly-feedpos-db' $SLURM_ARRAY_TASK_ID
#python main_batch.py 'firefly-feedbelief-db' $SLURM_ARRAY_TASK_ID

#python main_batch.py 'firefly-ppc' $SLURM_ARRAY_TASK_ID
#python main_batch.py 'firefly-ppc-tuned' $SLURM_ARRAY_TASK_ID
#python main_batch.py 'firefly-ppc-oc' $SLURM_ARRAY_TASK_ID
#python main_batch.py 'firefly-ppc-oc-tuned' $SLURM_ARRAY_TASK_ID

python main_batch.py 'model-1' $SLURM_ARRAY_TASK_ID

#End of script