import os
import json
import argparse
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import torch
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights

from data.transforms import ImageTransform
from data.data_utils import pil_img2rgb, add_special_tokens
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer
from modeling.bagel.qwen2_navit import NaiveCache
from modeling.autoencoder import load_ae
from inferencer import InterleaveInferencer
from modeling.bagel.modeling_utils import get_2d_sincos_pos_embed


def load_model(checkpoint_dir, checkpoint_file="model.safetensors"):
    """Load the Bagel model from checkpoint"""
    
    print(f"Available GPUs: {torch.cuda.device_count()}")
    print(f"GPU memory per device:")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name}, {props.total_memory / 1e9:.1f} GB")

    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
    
    # LLM config preparing
    llm_config = Qwen2Config.from_json_file(os.path.join(checkpoint_dir, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    # ViT config preparing
    vit_config = SiglipVisionConfig.from_json_file(os.path.join(checkpoint_dir, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

    # VAE loading
    vae_model, vae_config = load_ae(local_path=os.path.join(checkpoint_dir, "ae.safetensors"))

    # Bagel config preparing
    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config, 
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act='gelu_pytorch_tanh',
        latent_patch_size=2,
        max_latent_size=64,
    )

    # Create model with empty weights
    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model      = SiglipVisionModel(vit_config)
        model          = Bagel(language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

    # Initialize position embeddings before loading
    print("Initializing position embeddings before loading...")
    
    if hasattr(model, 'latent_pos_embed'):
        print("Initializing latent_pos_embed...")
        pos_embed = get_2d_sincos_pos_embed(model.latent_pos_embed.hidden_size, model.latent_pos_embed.max_num_patch_per_side)
        model.latent_pos_embed.pos_embed = torch.nn.Parameter(
            torch.from_numpy(pos_embed).float(), requires_grad=False
        )
        print(f"latent_pos_embed initialized with shape {model.latent_pos_embed.pos_embed.shape}")

    if hasattr(model, 'vit_pos_embed'):
        print("Initializing vit_pos_embed...")
        pos_embed = get_2d_sincos_pos_embed(model.vit_pos_embed.hidden_size, model.vit_pos_embed.max_num_patch_per_side)
        model.vit_pos_embed.pos_embed = torch.nn.Parameter(
            torch.from_numpy(pos_embed).float(), requires_grad=False
        )
        print(f"vit_pos_embed initialized with shape {model.vit_pos_embed.pos_embed.shape}")

    # Tokenizer Preparing
    tokenizer = Qwen2Tokenizer.from_pretrained(checkpoint_dir)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    # Image Transform Preparing
    vae_transform = ImageTransform(1024, 512, 16)
    vit_transform = ImageTransform(980, 512, 14)

    # Device mapping
    max_mem_per_gpu = "80GiB"
    
    print("Setting up device mapping...")
    device_map = infer_auto_device_map(
        model,
        max_memory={i: max_mem_per_gpu for i in range(torch.cuda.device_count())},
        no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
        dtype=torch.bfloat16,
    )

    # Handle same-device modules
    same_device_modules = [
        'language_model.model.embed_tokens',
        'time_embedder',
        'latent_pos_embed',
        'vae2llm',
        'llm2vae',
        'connector',
        'vit_pos_embed'
    ]

    if torch.cuda.device_count() == 1:
        first_device = device_map.get(same_device_modules[0], "cuda:0")
        for k in same_device_modules:
            if k in device_map:
                device_map[k] = first_device
            else:
                device_map[k] = "cuda:0"
    else:
        first_device = device_map.get(same_device_modules[0])
        if first_device is not None:
            for k in same_device_modules:
                if k in device_map:
                    device_map[k] = first_device

    print("Loading checkpoint...")
    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=checkpoint_path,
        device_map=device_map,
        offload_buffers=False,
        dtype=torch.bfloat16,
        force_hooks=True,
    )

    model = model.eval()
    print('Model loaded successfully!')
    
    return model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids


def load_jsonl(file_path):
    """Load data from JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_jsonl(data, file_path):
    """Save data to JSONL file"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def inference_single_sample(sample, inferencer, output_base_dir, sample_id, inference_hyper, system_prompt=""):
    """Run inference on a single sample"""
    
    # Extract prompt and images
    prompt = sample["Question"]
    
    # Load problem images
    problem_images = []
    problem_image_keys = sorted([k for k in sample.keys() if k.startswith("problem_image_")])
    
    for key in problem_image_keys:
        img_path = sample[key]
        try:
            img = Image.open(img_path).convert('RGB')
            problem_images.append(img)
        except Exception as e:
            print(f"Warning: Failed to load image {img_path}: {e}")
            # Create placeholder
            img = Image.new('RGB', (512, 512), color='gray')
            problem_images.append(img)
    
    # Create output directory for this sample - images saved in output_images/{data_id}/
    images_dir = os.path.join(output_base_dir, "output_images", str(sample_id))
    os.makedirs(images_dir, exist_ok=True)
    
    # Save problem images to output folder
    saved_problem_images = []
    for i, img in enumerate(problem_images):
        img_filename = f"problem_image_{i+1}.png"
        img_path = os.path.join(images_dir, img_filename)
        img.save(img_path)
        # Store relative path from output_base_dir
        saved_problem_images.append(os.path.join("output_images", str(sample_id), img_filename))
    
    # Prepare input
    if problem_images:
        current_input = [prompt] + problem_images
    else:
        current_input = [prompt]
    
    # Run iterative inference
    reasoning_text_list = []
    generated_images = []
    generated_image_paths = []
    
    iteration = 0
    max_iterations = 50  # Safety limit
    
    while iteration < max_iterations:
        # Get understanding output
        output = inferencer.interleave_inference(
            current_input, 
            understanding_output=True, 
            system_prompt=system_prompt, 
            **inference_hyper
        )
        
        # Check for stopping conditions
        has_final_answer = 'Final Answer:' in output[0] or '<answer>' in output[0] or '<eoc>' in output[0]
        
        if has_final_answer:
            if output[0].strip():
                try:
                    extracted_text = output[0].split('<|im_end|>')[0].split('<|im_start|>')[1]
                except:
                    extracted_text = output[0]
                reasoning_text_list.append(extracted_text)
                current_input = current_input + [extracted_text]
            break
        
        # Extract reasoning text
        try:
            extracted_text = output[0].split('<|im_end|>')[0].split('<|im_start|>')[1]
        except:
            extracted_text = output[0]
        
        reasoning_text_list.append(extracted_text)
        
        # Generate image based on current reasoning
        current_input_with_reasoning = current_input + [extracted_text]
        output = inferencer.interleave_inference(
            current_input_with_reasoning, 
            system_prompt=system_prompt, 
            **inference_hyper
        )
        image_output = output[0]
        
        # Save generated image
        generated_images.append(image_output)
        image_filename = f'output_image_{iteration + 1}.png'
        image_path = os.path.join(images_dir, image_filename)
        
        image_output.save(image_path)
        generated_image_paths.append(image_path)  # Full path for evaluation
        
        # Update input for next iteration
        current_input = current_input_with_reasoning + [image_output]
        
        iteration += 1
    
    # Format output reasoning
    output_reasoning = " ".join(reasoning_text_list)
    
    return {
        "output_reasoning": output_reasoning,
        "output_images": generated_image_paths,  # List of full paths
        "num_reasoning_steps": len(reasoning_text_list),
        "num_generated_images": len(generated_image_paths)
    }


def main():
    parser = argparse.ArgumentParser(description='Batch inference for Bagel model')
    parser.add_argument('--input_jsonl', type=str, required=True, help='Path to input JSONL file')
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory')
    parser.add_argument('--checkpoint_dir', type=str, required=True, help='Model checkpoint directory')
    parser.add_argument('--checkpoint_file', type=str, default='model.safetensors', help='Checkpoint filename')
    parser.add_argument('--do_sample', action='store_true', help='Whether to use sampling')
    parser.add_argument('--text_temperature', type=float, default=0.0, help='Text generation temperature')
    parser.add_argument('--cfg_text_scale', type=float, default=4.0, help='CFG text scale')
    parser.add_argument('--cfg_img_scale', type=float, default=2.0, help='CFG image scale')
    parser.add_argument('--num_timesteps', type=int, default=50, help='Number of diffusion timesteps')
    parser.add_argument('--start_idx', type=int, default=0, help='Start index for processing')
    parser.add_argument('--end_idx', type=int, default=None, help='End index for processing')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base_dir = os.path.join(args.output_dir, f"batch_inference_{timestamp}")
    os.makedirs(output_base_dir, exist_ok=True)
    
    print(f"Output directory: {output_base_dir}")
    
    # Load model
    print("Loading model...")
    model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids = load_model(
        args.checkpoint_dir, 
        args.checkpoint_file
    )
    
    # Create inferencer
    inferencer = InterleaveInferencer(
        model=model, 
        vae_model=vae_model, 
        tokenizer=tokenizer, 
        vae_transform=vae_transform, 
        vit_transform=vit_transform, 
        new_token_ids=new_token_ids
    )
    
    # Inference hyperparameters
    inference_hyper = dict(
        do_sample=args.do_sample,
        text_temperature=args.text_temperature,
        cfg_text_scale=args.cfg_text_scale,
        cfg_img_scale=args.cfg_img_scale,
        cfg_interval=[0.0, 1.0],
        timestep_shift=3.0,
        num_timesteps=args.num_timesteps,
        cfg_renorm_min=0.0,
        cfg_renorm_type="text_channel",
    )
    
    system_prompt = ''
    
    # Load input data
    print(f"Loading data from {args.input_jsonl}...")
    input_data = load_jsonl(args.input_jsonl)
    
    # Determine processing range
    start_idx = args.start_idx
    end_idx = args.end_idx if args.end_idx is not None else len(input_data)
    end_idx = min(end_idx, len(input_data))
    
    print(f"Processing samples {start_idx} to {end_idx-1} (total: {end_idx - start_idx})")
    
    # Process samples
    output_data = []
    
    for idx in tqdm(range(start_idx, end_idx), desc="Processing samples"):
        sample = input_data[idx]
        
        print(f"\n{'='*80}")
        print(f"Processing sample {idx}")
        print(f"{'='*80}")
        
        try:
            # Run inference
            result = inference_single_sample(
                sample=sample,
                inferencer=inferencer,
                output_base_dir=output_base_dir,
                sample_id=idx,
                inference_hyper=inference_hyper,
                system_prompt=system_prompt
            )
            
            # Combine original data with results
            output_sample = sample.copy()
            output_sample["output_reasoning"] = result["output_reasoning"]
            output_sample["output_images"] = result["output_images"]
            output_sample["num_reasoning_steps"] = result["num_reasoning_steps"]
            output_sample["num_generated_images"] = result["num_generated_images"]
            output_sample["sample_id"] = idx
            
            output_data.append(output_sample)
            
            print(f"✓ Sample {idx} completed:")
            print(f"  - Reasoning steps: {result['num_reasoning_steps']}")
            print(f"  - Generated images: {result['num_generated_images']}")
            
        except Exception as e:
            print(f"✗ Error processing sample {idx}: {e}")
            import traceback
            traceback.print_exc()
            
            # Save error info
            output_sample = sample.copy()
            output_sample["error"] = str(e)
            output_sample["sample_id"] = idx
            output_data.append(output_sample)
    
    # Save results
    output_jsonl_path = os.path.join(output_base_dir, "results.jsonl")
    save_jsonl(output_data, output_jsonl_path)
    
    print(f"\n{'='*80}")
    print(f"Batch inference completed!")
    print(f"Results saved to: {output_jsonl_path}")
    print(f"Total samples processed: {len(output_data)}")
    print(f"Output directory: {output_base_dir}")
    print(f"{'='*80}")
    
    # Save summary
    summary = {
        "timestamp": timestamp,
        "input_file": args.input_jsonl,
        "checkpoint_dir": args.checkpoint_dir,
        "total_samples": len(output_data),
        "start_idx": start_idx,
        "end_idx": end_idx,
        "inference_hyper": inference_hyper,
        "errors": sum(1 for item in output_data if "error" in item)
    }
    
    summary_path = os.path.join(output_base_dir, "summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
