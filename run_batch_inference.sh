#!/bin/bash

# Batch Inference Example Script
# This script demonstrates how to run batch inference on a JSONL dataset

# Configuration
INPUT_JSONL="path/to/your/input_data.jsonl"
OUTPUT_DIR="./output"
CHECKPOINT_DIR="/scratch/by2593/hf_cache/hub/models--multimodal-reasoning-lab--Bagel-Zebra-CoT/snapshots/ebce32410ee2062d073feae484ea2c6c1515fba8"
CHECKPOINT_FILE="model.safetensors"

# Inference parameters
DO_SAMPLE=false
TEXT_TEMPERATURE=0.0
CFG_TEXT_SCALE=4.0
CFG_IMG_SCALE=2.0
NUM_TIMESTEPS=50
SEED=42

# Processing range (optional)
START_IDX=0
END_IDX=100  # Process first 100 samples, remove this argument to process all

# Run batch inference
python batch_inference.py \
    --input_jsonl "$INPUT_JSONL" \
    --output_dir "$OUTPUT_DIR" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --checkpoint_file "$CHECKPOINT_FILE" \
    --text_temperature $TEXT_TEMPERATURE \
    --cfg_text_scale $CFG_TEXT_SCALE \
    --cfg_img_scale $CFG_IMG_SCALE \
    --num_timesteps $NUM_TIMESTEPS \
    --start_idx $START_IDX \
    --end_idx $END_IDX \
    --seed $SEED

# If you want to enable sampling, add: --do_sample

echo "Batch inference completed!"
echo "Check the output directory for results: $OUTPUT_DIR"
