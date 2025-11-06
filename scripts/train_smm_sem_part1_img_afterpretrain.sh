#!/bin/bash
# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

# Change to the project directory
cd /lustre/fsw/portfolios/nvr/users/ymingli/projects/ybq/Bagel-Zebra-CoT-smm-origin
source /lustre/fsw/portfolios/nvr/users/ymingli/miniconda3/etc/profile.d/conda.sh
conda activate bagel
export PYTHONPATH=/lustre/fsw/portfolios/nvr/users/ymingli/projects/ybq/Bagel-Zebra-CoT-smm-origin:$PYTHONPATH
export WANDB_MODE=offline
export WANDB_ANONYMOUS=must


NUM_NODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=29500
NPROC_PER_NODE=8
MODEL_PATH=/lustre/fsw/portfolios/nvr/users/ymingli/projects/ybq/results/checkpoints_random_views1357_imgfirst/0002400

# replace the variables with your own
torchrun \
  --nnodes=$NUM_NODES \
  --node_rank=$NODE_RANK \
  --nproc_per_node=$NPROC_PER_NODE \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  train/pretrain_unified_navit.py \
  --dataset_config_file ./data/configs/example_smm_semantic_part1.yaml \
  --model_path $MODEL_PATH \
  --layer_module Qwen2MoTDecoderLayer \
  --llm_path Qwen/Qwen2.5-0.5B-Instruct \
  --vit_path /lustre/fsw/portfolios/nvr/users/ymingli/hf_cache/models--multimodal-reasoning-lab--Bagel-Zebra-CoT/snapshots/ebce32410ee2062d073feae484ea2c6c1515fba8 \
  --max_latent_size 64 \
  --visual_und True \
  --finetune_from_hf False \
  --auto_resume True \
  --resume-model-only False \
  --finetune-from-ema False \
  --log_every 1 \
  --lr 2e-5 \
  --lr_scheduler cosine \
  --min_lr 1e-6 \
  --num_worker 1 \
  --expected_num_tokens 42000 \
  --max_num_tokens 42000 \
  --max_num_tokens_per_sample 42000 \
  --prefer_buffer_before 10000 \
  --num_shard=$NPROC_PER_NODE \
  --sharding_strategy="HYBRID_SHARD" \
  --wandb_project "smm" \
  --wandb_name "h200-zebra-cot-smm-sbatch-$(date +%Y%m%d_%H%M%S)" \
  --save_every 50 \
  --warmup_steps 50 \
  --total_steps 500 \
  --results_dir /lustre/fsw/portfolios/nvr/users/ymingli/projects/ybq/results/ \
  --checkpoint_dir /lustre/fsw/portfolios/nvr/users/ymingli/projects/ybq/results/checkpoints_smm_sem1_from_2400/ \
  --cpu_offload True \
  --max_checkpoints 2 \


echo "SMM training completed on $(date)"

# sbatch scripts/train_smm_sbatch.sh
#   > run_sem1.out 2> run_sem1.err