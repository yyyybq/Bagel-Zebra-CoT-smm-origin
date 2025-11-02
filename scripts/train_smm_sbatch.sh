#!/bin/bash
#SBATCH --job-name=bagel-zebra-cot-smm
#SBATCH --partition=h200_tandon
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h200:8
#SBATCH --mem=1600G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_logs/train_smm_%j.out
#SBATCH --error=slurm_logs/train_smm_%j.err

# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

# Load any necessary modules (adjust as needed for your cluster)
# module load cuda/12.1
# module load conda

# Activate conda environment
source /scratch/by2593/miniconda3/etc/profile.d/conda.sh
conda activate bagel

# Change to the project directory
cd /scratch/by2593/Bagel-Zebra-CoT-origin

# Set environment variables
export HF_HOME=/dev/shm/
export PYTHONPATH=/scratch/by2593/Bagel-Zebra-CoT-origin:$PYTHONPATH
export WANDB_MODE=offline
export WANDB_ANONYMOUS=must

# SLURM variables
NUM_NODES=1
NODE_RANK=0
MASTER_ADDR=$(hostname)
MASTER_PORT=29500
NPROC_PER_NODE=8
MODEL_PATH=/scratch/by2593/hf_cache/hub/models--multimodal-reasoning-lab--Bagel-Zebra-CoT/snapshots/ebce32410ee2062d073feae484ea2c6c1515fba8

echo "Starting SMM training on node: $SLURM_JOB_NODELIST"
echo "Job ID: $SLURM_JOB_ID"
echo "Number of GPUs: $NPROC_PER_NODE"

# Run training
torchrun \
  --nnodes=$NUM_NODES \
  --node_rank=$NODE_RANK \
  --nproc_per_node=$NPROC_PER_NODE \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  train/pretrain_unified_navit.py \
  --dataset_config_file ./data/configs/example_smm_random.yaml \
  --model_path $MODEL_PATH \
  --layer_module Qwen2MoTDecoderLayer \
  --max_latent_size 64 \
  --visual_und True \
  --finetune_from_hf True \
  --auto_resume True \
  --resume-model-only False \
  --finetune-from-ema False \
  --log_every 1 \
  --lr 2e-5 \
  --lr_scheduler cosine \
  --min_lr 1e-6 \
  --num_worker 1 \
  --expected_num_tokens 50000 \
  --max_num_tokens 50000 \
  --max_num_tokens_per_sample 50000 \
  --prefer_buffer_before 10000 \
  --num_shard=$NPROC_PER_NODE \
  --sharding_strategy="HYBRID_SHARD" \
  --wandb_project "smm" \
  --wandb_name "h200-zebra-cot-smm-sbatch-$(date +%Y%m%d_%H%M%S)" \
  --save_every 100 \
  --warmup_steps 50 \
  --total_steps 5000 \
  --results_dir results/ \
  --checkpoint_dir /scratch/by2593/Bagel-Zebra-CoT-origin/results/checkpoints_smm_random_20251026_033448/ \
  --cpu_offload True \
  --max_checkpoints 2

echo "SMM training completed on $(date)"

# sbatch scripts/train_smm_sbatch.sh