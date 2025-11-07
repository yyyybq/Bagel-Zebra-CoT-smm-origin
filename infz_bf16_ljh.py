import os
import json
import numpy as np
from datetime import datetime
from copy import deepcopy
from typing import (
    Any,
    AsyncIterable,
    Callable,
    Dict,
    Generator,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)
import requests
from io import BytesIO

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

# Set paths for your trained checkpoint
# checkpoint_dir = "/scratch/by2593/merged_checkpoint_final"
origin_checkpoint_dir = "/lustre/fsw/portfolios/nvr/users/ymingli/hf_cache/models--multimodal-reasoning-lab--Bagel-Zebra-CoT/snapshots/ebce32410ee2062d073feae484ea2c6c1515fba8"


checkpoint_path = "results/checkpoints_smm_sem1_1106/0000300/model.safetensors"

print(f"Available GPUs: {torch.cuda.device_count()}")
print(f"GPU memory per device:")
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {props.name}, {props.total_memory / 1e9:.1f} GB")

# LLM config preparing (use base model configs)
llm_config = Qwen2Config.from_json_file(os.path.join(origin_checkpoint_dir, "llm_config.json"))
llm_config.qk_norm = True
llm_config.tie_word_embeddings = False
llm_config.layer_module = "Qwen2MoTDecoderLayer"

# ViT config preparing (use base model configs)
vit_config = SiglipVisionConfig.from_json_file(os.path.join(origin_checkpoint_dir, "vit_config.json"))
vit_config.rope = False
vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

# VAE loading (use base model VAE)
vae_model, vae_config = load_ae(local_path=os.path.join(origin_checkpoint_dir, "ae.safetensors"))

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
    max_latent_size=64,## 默认64，改为实际的latent尺寸
)

# Import the position embedding function first
from modeling.bagel.modeling_utils import get_2d_sincos_pos_embed

# Create model with empty weights
with init_empty_weights():
    language_model = Qwen2ForCausalLM(llm_config)
    vit_model      = SiglipVisionModel(vit_config)
    model          = Bagel(language_model, vit_model, config)
    model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

# Initialize position embeddings with proper values BEFORE loading checkpoint
print("Initializing position embeddings before loading...")

# Initialize latent_pos_embed if it exists
if hasattr(model, 'latent_pos_embed'):
    print("Initializing latent_pos_embed...")
    pos_embed = get_2d_sincos_pos_embed(model.latent_pos_embed.hidden_size, model.latent_pos_embed.max_num_patch_per_side)
    # Create parameter with actual values, not meta
    model.latent_pos_embed.pos_embed = torch.nn.Parameter(
        torch.from_numpy(pos_embed).float(), requires_grad=False
    )
    print(f"latent_pos_embed initialized with shape {model.latent_pos_embed.pos_embed.shape}")

# Initialize vit_pos_embed if it exists  
if hasattr(model, 'vit_pos_embed'):
    print("Initializing vit_pos_embed...")
    pos_embed = get_2d_sincos_pos_embed(model.vit_pos_embed.hidden_size, model.vit_pos_embed.max_num_patch_per_side)
    # Create parameter with actual values, not meta
    model.vit_pos_embed.pos_embed = torch.nn.Parameter(
        torch.from_numpy(pos_embed).float(), requires_grad=False
    )
    print(f"vit_pos_embed initialized with shape {model.vit_pos_embed.pos_embed.shape}")

print("Position embeddings initialized successfully")

# Tokenizer Preparing (use base model tokenizer)
tokenizer = Qwen2Tokenizer.from_pretrained(origin_checkpoint_dir)
tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

# Image Transform Preparing
vae_transform = ImageTransform(1024, 512, 16)
vit_transform = ImageTransform(980, 512, 14)

# Device mapping for 8x80GB GPUs - use bf16 directly
max_mem_per_gpu = "80GiB"

print("Setting up device mapping...")
device_map = infer_auto_device_map(
    model,
    max_memory={i: max_mem_per_gpu for i in range(torch.cuda.device_count())},
    no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
    dtype=torch.bfloat16,  # Use bf16 for device mapping
)

print("Device map:", device_map)

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

print("Final device map:", device_map)

# Load checkpoint directly in bf16
print(f"Loading checkpoint directly in bfloat16: {checkpoint_path}")
print("Loading model from safetensors file...")

# Load model directly in bf16
model = load_checkpoint_and_dispatch(
    model,
    checkpoint=checkpoint_path,
    device_map=device_map,
    offload_buffers=False,
    dtype=torch.bfloat16,   # Load directly as bf16
    force_hooks=True,
)

model = model.eval()

print('Model loaded directly in bfloat16!')
print(f"Model dtype: {next(model.parameters()).dtype}")

# Position embeddings were already initialized before model loading
print("Position embeddings were pre-initialized before loading checkpoint")

print("Model loading completed successfully!")

# Check memory usage
print("GPU memory usage after loading:")
for i in range(torch.cuda.device_count()):
    if torch.cuda.memory_allocated(i) > 0:
        allocated = torch.cuda.memory_allocated(i) / 1e9
        cached = torch.cuda.memory_reserved(i) / 1e9
        print(f"  GPU {i}: {allocated:.1f}GB allocated, {cached:.1f}GB cached")

# Rest of inference code
from inferencer import InterleaveInferencer

inferencer = InterleaveInferencer(
    model=model, 
    vae_model=vae_model, 
    tokenizer=tokenizer, 
    vae_transform=vae_transform, 
    vit_transform=vit_transform, 
    new_token_ids=new_token_ids
)

import random
import numpy as np

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

inference_hyper=dict(
    do_sample=False,
    text_temperature=0.0,
    cfg_text_scale=4.0,
    cfg_img_scale=2.0,
    cfg_interval=[0.0, 1.0],
    timestep_shift=3.0,
    num_timesteps=50,
    cfg_renorm_min=0.0,
    cfg_renorm_type="text_channel",
)

INTERLEAVED_SYSTEM_PROMPT = '''You are an AI reasoning assistant capable of step-by-step interleaved text and visual chain of thought. Think step by step and use visual aids to enhance your problem-solving.'''
INTERLEAVED_SYSTEM_PROMPT = ''

prompt = '''
        My goal is to generate a visual guide for constructing a specific shape using a set of blocks. This involves multiple steps, each requiring the addition of a new block to progressively build the final shape. The initial input includes 4 images of multiple blocks that will be used -- block 1: <image_start>[problem_image_1]<image_end> block 2: <image_start>[problem_image_2]<image_end> block 3: <image_start>[problem_image_3]<image_end> block 4: <image_start>[problem_image_4]<image_end> and an image of the final desired shape: <image_start>[problem_image_5]<image_end>. I need to imagine and generate images of intermediate steps, leading up to the final construction.  Step 0 has been completed: a blue cube block has been placed on top of the ground. The image after step 0 is provided: <image_start>[problem_image_6]<image_end>. Now I need to generate the image for step 1, considering spatial relationships and stability.
        '''

image = []
base_path = '/lustre/fsw/portfolios/nvr/users/ymingli/projects/ljh/Show-o/show-o2/'
image_paths = [
    "/lustre/fsw/portfolios/nvr/users/ymingli/projects/ljh/Show-o/show-o2/semantic_blocks_part1/each_block_views_diffposes/cube_blue.png",    
    "/lustre/fsw/portfolios/nvr/users/ymingli/projects/ljh/Show-o/show-o2/semantic_blocks_part1/each_block_views_diffposes/cuboid3_yellow.png",
    "/lustre/fsw/portfolios/nvr/users/ymingli/projects/ljh/Show-o/show-o2/semantic_blocks_part1/each_block_views_diffposes/arch_red.png",  
    "/lustre/fsw/portfolios/nvr/users/ymingli/projects/ljh/Show-o/show-o2/semantic_blocks_part1/each_block_views_diffposes/triangle_orange.png", 
    "/lustre/fsw/portfolios/nvr/users/ymingli/projects/ljh/semantic_blocks_part1/015/final_state/015_final_1.png",  
    "/lustre/fsw/portfolios/nvr/users/ymingli/projects/ljh/semantic_blocks_part1/015/steps/view_1/015_step0_1.png",
]

print("Loading input images:")
for i, img_path in enumerate(image_paths):
    try:
        img = Image.open(img_path).convert('RGB')
        image.append(img)
        print(f"  ✓ Loaded problem_image_{i+1}: {img_path}")
        print(f"     Image size: {img.size}")
    except Exception as e:
        print(f"  ✗ Failed to load {img_path}: {e}")
        # Create a placeholder image if file not found
        img = Image.new('RGB', (512, 512), color='gray')
        image.append(img)
        print(f"  ⚠ Using placeholder for problem_image_{i+1}")

print(prompt)
print('-'*50)

# Create output folder with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_folder = f"reasoning_output_example_015_{timestamp}"
images_folder = os.path.join(output_folder, "images")
os.makedirs(images_folder, exist_ok=True)

print(f"Output will be saved to: {output_folder}")

# Save the original problem images if they exist
problem_image_paths = []
if image is not None:
    if isinstance(image, list):
        # Handle multiple images
        for i, img in enumerate(image):
            problem_image_path = os.path.join(images_folder, f"problem_image_{i+1}.png")
            relative_path = os.path.join("images", f"problem_image_{i+1}.png")
            img.save(problem_image_path)
            problem_image_paths.append(relative_path)
            print(f"Problem image {i+1} saved at '{problem_image_path}'")
    else:
        # Handle single image
        problem_image_path = os.path.join(images_folder, "problem_image.png")
        relative_path = os.path.join("images", "problem_image.png")
        image.save(problem_image_path)
        problem_image_paths.append(relative_path)
        print(f"Problem image saved at '{problem_image_path}'")

reasoning_text = []
reasoning_images = []
generated_image_paths = []  # Store relative paths to generated reasoning images

# Create input with multiple images properly flattened
if image is not None:
    if isinstance(image, list):
        current_input = [prompt] + image  # Flatten the list of images
    else:
        current_input = [prompt, image]
else:
    current_input = [prompt]

# Loop until no more vision_start tokens
iteration = 0
while True:    
    # Get understanding output
    print(f"iteration: {iteration}")
    output = inferencer.interleave_inference(current_input, understanding_output=True, system_prompt=INTERLEAVED_SYSTEM_PROMPT, **inference_hyper)

    # Check for stopping conditions
    has_final_answer = 'Final Answer:' in output[0] or '<answer>' in output[0]
    
    # Stop if we have a final answer OR if there's no vision token (no more images to generate)
    # should_stop = has_final_answer or not has_vision_token
    should_stop = has_final_answer


    if should_stop:
        if output[0].strip():
            extracted_text = output[0].split('<|im_end|>')[0].split('<|im_start|>')[1]
            reasoning_text.append(extracted_text)
            print(f"{extracted_text}")
            current_input = current_input + [extracted_text]
        break
    
    extracted_text = output[0].split('<|im_end|>')[0].split('<|im_start|>')[1]
    reasoning_text.append(extracted_text)
    print(f"{extracted_text}")
    
    # Generate image based on current reasoning
    current_input_with_reasoning = current_input + [extracted_text]
    output = inferencer.interleave_inference(current_input_with_reasoning, system_prompt=INTERLEAVED_SYSTEM_PROMPT, **inference_hyper)
    image_output = output[0]

    # Save and collect the generated image
    reasoning_images.append(image_output)
    image_filename = f'reasoning_image_{iteration + 1}.png'
    image_path = os.path.join(images_folder, image_filename)
    relative_image_path = os.path.join("images", image_filename)  # Relative path for JSON
    
    image_output.save(image_path)
    generated_image_paths.append(relative_image_path)
    print(f"Image saved at '{image_path}'")

    # Update input for next iteration
    current_input = current_input_with_reasoning + [image_output]
    
    iteration += 1
    print('-'*50)

# Save reasoning data to JSON
reasoning_data = {
    "timestamp": timestamp,
    "prompt": prompt,
    "system_prompt": INTERLEAVED_SYSTEM_PROMPT,
    "problem_image_paths": problem_image_paths if problem_image_paths else None,
    "response": [
        {
            "step": i + 1,
            "text": text,
            "image_path": generated_image_paths[i] if i < len(generated_image_paths) else None
        }
        for i, text in enumerate(reasoning_text)
    ],
    "total_steps": len(reasoning_text),
    "total_images": len(generated_image_paths)
}

# Save JSON file
json_path = os.path.join(output_folder, "reasoning_data.json")
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(reasoning_data, f, indent=2, ensure_ascii=False)

print(f"\nReasoning complete!")
print(f"Output folder: {output_folder}")
print(f"JSON metadata: {json_path}")
print(f"Generated {len(generated_image_paths)} images and {len(reasoning_text)} text steps")