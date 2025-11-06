#!/bin/bash

# Evaluation Example Script

# Path to the results JSONL file from batch inference
RESULTS_JSONL="./output/batch_inference_20250104_120000/results.jsonl"

# Optional: specify output evaluation JSON file
# If not specified, it will be saved in the same directory as results.jsonl
OUTPUT_EVAL_JSON="./output/batch_inference_20250104_120000/evaluation_data.json"

# Run evaluation
python evaluate_results.py \
    --results_jsonl "$RESULTS_JSONL" \
    --output_eval_json "$OUTPUT_EVAL_JSON"

echo "Evaluation completed!"
echo "Check the evaluation data and statistics in the output directory"
