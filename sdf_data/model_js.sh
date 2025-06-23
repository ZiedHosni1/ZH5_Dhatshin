#!/bin/bash
#SBATCH --job-name=wgan_flp
#SBATCH --output=wgan_output_%j.log
#SBATCH --error=wgan_error_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8               # Better CPU parallelism
#SBATCH --gres=gpu:1                    # Single GPU
#SBATCH --mem=32G                       # Adjust only if needed
#SBATCH --time=24:00:00
#SBATCH --partition=gpu                # Or 'small-gpu' on Kathleen

module purge
module load anaconda3
source activate wgan_env

cd $SLURM_SUBMIT_DIR
python batch_wgan_restartable.py