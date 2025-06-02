#!/bin/bash
#SBATCH --job-name=emb_gen
#SBATCH --output=logs/%x-%j.out         # stdout
#SBATCH --error=logs/%x-%j.err          # stderr
#SBATCH --partition=P2
#SBATCH --nodes=1                  # single node
#SBATCH --cpus-per-task=8         # CPU cores per task
#SBATCH --gres=gpu:1               # 1 GPUs
#SBATCH --mem=100G                 # memory
#SBATCH --exclude=b[19,31]

DATASET=McAuley-Lab/Amazon-C4
CACHE_DIR=data/amazon_c4/raw/cache
PLM_NAME=hyp1231/blair-roberta-large
FEAT_NAME=blair-large
GPU_ID=0

python src/blair/generate_emb.py \
    --dataset $DATASET \
    --cache_path $CACHE_DIR \
    --plm_name $PLM_NAME \
    --feat_name $FEAT_NAME \
    --gpu_id $GPU_ID