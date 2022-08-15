#!/bin/bash
#SBATCH --job-name=mod_meta_create_ds
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=40         # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time 20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=/gpfsssd/scratch/rech/six/uue59kq/mod-meta/logs/create_ds/%x-%j.out           # output file name
#SBATCH --account=six@cpu

set -x -e

source $HOME/.bashrc_six

export HF_DATASETS_OFFLINE=1
export HF_DATASETS_CACHE=$SCRATCH/mod-meta/hf_datasets_cache

DATASET_DIR=$SCRATCH/dataset/c4-en-html-with-metadata

WORKING_DIR=$WORK/repos/bs-meta-stats
pushd $WORKING_DIR

python first_time_load.py \
    --dataset_path $DATASET_DIR
