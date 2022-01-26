#!/bin/bash
#SBATCH -p cpu
#SBATCH -c2
#SBATCH --time=1:00:00
#SBATCH --output=/media/compute/homes/dmindlin/inception_score_fashion_mnist/slurm_logs/upload_tensorboard_inception.out
#SBATCH --mail-user=dmindlin@techfak.uni-bielefeld.de
#SBATCH --mail-type=END
#SBATCH --ntasks=1
#SBATCH --ntasks-per-core=1

srun upload_tensorboard_inception.job.sh