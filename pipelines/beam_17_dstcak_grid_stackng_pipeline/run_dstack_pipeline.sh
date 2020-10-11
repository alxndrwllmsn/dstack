#!/bin/bash

#SBATCH --time=30:00:00
#SBATCH --mem=32000
#SBATCH -c 16 #Run on 16 CPUs
#SBATCH --mail-user=rstofi@gmail.com
#SBATCH --mail-type=FAIL
#SBATCH --job-name=dsack_pipeline_test
#SBATCH --output=./slurm_dsack_pipeline_test.out

source /home/krozgonyi/.bashrc
dstack_env_setup

snakemake --cores 16
