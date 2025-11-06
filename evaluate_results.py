import os
import json
import argparse
from pathlib import Path
from collections import defaultdict


def load_jsonl(file_path):
    """Load data from JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def evaluate_results(results_jsonl):
    """Evaluate the batch inference results"""
    
    print("Loading results...")
    results = load_jsonl(results_jsonl)
    
    # Statistics
    stats = {
        "total_samples": len(results),
        "successful": 0,
        "failed": 0,
        "avg_reasoning_steps": 0,
        "avg_generated_images": 0,
        "reasoning_steps_distribution": defaultdict(int),
        "image_count_distribution": defaultdict(int)
    }
    
    total_steps = 0
    total_images = 0
    
    for item in results:
        if "error" in item:
            stats["failed"] += 1
            print(f"Sample {item.get('sample_id', '?')} failed: {item['error']}")
        else:
            stats["successful"] += 1
            
            if "num_reasoning_steps" in item:
                steps = item["num_reasoning_steps"]
                total_steps += steps
                stats["reasoning_steps_distribution"][steps] += 1
            
            if "num_generated_images" in item:
                images = item["num_generated_images"]
                total_images += images
                stats["image_count_distribution"][images] += 1
    
    if stats["successful"] > 0:
        stats["avg_reasoning_steps"] = total_steps / stats["successful"]
        stats["avg_generated_images"] = total_images / stats["successful"]
    
    # Print statistics
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(f"Total samples: {stats['total_samples']}")
    print(f"Successful: {stats['successful']}")
    print(f"Failed: {stats['failed']}")
    print(f"Success rate: {stats['successful']/stats['total_samples']*100:.2f}%")
    print(f"\nAverage reasoning steps: {stats['avg_reasoning_steps']:.2f}")
    print(f"Average generated images: {stats['avg_generated_images']:.2f}")
    
    print("\nReasoning steps distribution:")
    for steps in sorted(stats["reasoning_steps_distribution"].keys()):
        count = stats["reasoning_steps_distribution"][steps]
        print(f"  {steps} steps: {count} samples")
    
    print("\nGenerated images distribution:")
    for images in sorted(stats["image_count_distribution"].keys()):
        count = stats["image_count_distribution"][images]
        print(f"  {images} images: {count} samples")
    
    print("="*80)
    
    return stats


def extract_for_evaluation(results_jsonl, output_file):
    """Extract data in a format suitable for evaluation"""
    
    results = load_jsonl(results_jsonl)
    
    eval_data = []
    for item in results:
        eval_item = {
            "sample_id": item.get("sample_id"),
            "question": item.get("Question"),
            "ground_truth_reasoning": item.get("Text Reasoning Trace"),
            "ground_truth_answer": item.get("Final Answer"),
            "output_reasoning": item.get("output_reasoning"),
            "output_images": item.get("output_images", []),
            "num_reasoning_steps": item.get("num_reasoning_steps"),
            "num_generated_images": item.get("num_generated_images"),
            "error": item.get("error")
        }
        
        # Extract ground truth image paths
        gt_reasoning_images = []
        for key in sorted(item.keys()):
            if key.startswith("reasoning_image_"):
                gt_reasoning_images.append(item[key])
        eval_item["ground_truth_images"] = gt_reasoning_images
        
        eval_data.append(eval_item)
    
    # Save evaluation data
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(eval_data, f, indent=2, ensure_ascii=False)
    
    print(f"Evaluation data saved to: {output_file}")
    return eval_data


def main():
    parser = argparse.ArgumentParser(description='Evaluate batch inference results')
    parser.add_argument('--results_jsonl', type=str, required=True, help='Path to results JSONL file')
    parser.add_argument('--output_eval_json', type=str, default=None, help='Output evaluation JSON file')
    
    args = parser.parse_args()
    
    # Evaluate results
    stats = evaluate_results(args.results_jsonl)
    
    # Extract for evaluation
    if args.output_eval_json is None:
        results_dir = os.path.dirname(args.results_jsonl)
        args.output_eval_json = os.path.join(results_dir, "evaluation_data.json")
    
    eval_data = extract_for_evaluation(args.results_jsonl, args.output_eval_json)
    
    # Save statistics
    stats_file = os.path.join(os.path.dirname(args.results_jsonl), "statistics.json")
    with open(stats_file, 'w', encoding='utf-8') as f:
        # Convert defaultdict to dict for JSON serialization
        stats_json = dict(stats)
        stats_json["reasoning_steps_distribution"] = dict(stats_json["reasoning_steps_distribution"])
        stats_json["image_count_distribution"] = dict(stats_json["image_count_distribution"])
        json.dump(stats_json, f, indent=2, ensure_ascii=False)
    
    print(f"Statistics saved to: {stats_file}")


if __name__ == "__main__":
    main()
