#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu:4
#SBATCH --job-name=main-objects-3
#SBATCH --error=%x.err
#SBATCH --output=%x.out
#SBATCH --mail-type=all            # when something happens
#SBATCH --mail-user=cg4177@nyu.edu # send me mail

module purge

singularity exec --nv \
	    --overlay /scratch/cg4177/video-frame-prediction-env/overlay-15GB-500K.ext3:ro \
	    /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif\
	    /bin/bash -c "source /ext3/env.sh; python main_hpc.py --cfg=config_hpc.json"
