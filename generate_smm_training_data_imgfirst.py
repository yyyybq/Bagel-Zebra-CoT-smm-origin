import json
import os
from pathlib import Path
from typing import List, Dict, Any

# 每种积木对应的图片路径模板
BLOCK_IMAGE_PATH_TEMPLATE = "/lustre/fsw/portfolios/nvr/users/ymingli/projects/ljh/Show-o/show-o2/semantic_blocks_part1/each_block_views_diffposes/{block_type}_{color}.png"

def get_depend_description(block: Dict, all_blocks: List[Dict]) -> str:
    """根据依赖关系生成描述"""
    depend_ids = block.get("depend", [])
    
    if 0 in depend_ids and len(depend_ids) == 1:
        return "on top of the ground"
    
    depend_blocks = []
    for dep_id in depend_ids:
        if dep_id == 0:
            depend_blocks.append("the ground")
        else:
            # 找到对应的block
            for b in all_blocks:
                if b["order"] == dep_id:
                    depend_blocks.append(f"the {b['color']} {b['type']}")
                    break
    
    if len(depend_blocks) == 0:
        return "on top of the ground"
    elif len(depend_blocks) == 1:
        return f"on top of {depend_blocks[0]}"
    elif len(depend_blocks) == 2:
        return f"on top of {depend_blocks[0]} and {depend_blocks[1]}"
    else:
        return f"on top of {', '.join(depend_blocks[:-1])}, and {depend_blocks[-1]}"


def generate_question_template(scene_name: str, blocks: List[Dict], view_angle: str = "Front45") -> str:
    """生成问题模板"""
    total_blocks = len(blocks)
    first_block = blocks[0]
    
    question = (
        "My goal is to generate a visual guide for constructing a specific shape using a set of blocks. "
        "This involves multiple steps, each requiring the addition of a new block to progressively build the final shape. "
    )

    question += f"The initial input includes {total_blocks} images of multiple blocks that will be used -- "

    for i in range(1, total_blocks + 1):
        question += f"block {i}: <image_start>[problem_image_{i}]<image_end> "

    question += f"and an image of the final desired shape: <image_start>[problem_image_{total_blocks+1}]<image_end>. "

    question += f"I need to imagine and generate images of intermediate steps, leading up to the final construction. "

    question += f" Step 0 has been completed: a {first_block['color']} {first_block['type']} block has been placed on top of the ground. The image after step 0 is provided: <image_start>[problem_image_{total_blocks+2}]<image_end>. "


    return question


def generate_thought_trace(blocks: List[Dict]) -> str:
    """生成推理过程模板"""
    thoughts = []
    thoughts.append("Now I need to generate the image for step 1.")
    # 为每个后续的block生成THOUGHT
    for i, block in enumerate(blocks[1:], start=1):
        step_num = i
        color = block['color']
        block_type = block['type']
        depend_desc = get_depend_description(block, blocks)

        # 定义多种风格的thought模板
        thought_templates = [
            # 风格1: 先想象，生成图片，再描述图片
            f"<image_start>[reasoning_image_{step_num}]<image_end> THOUGHT {step_num}: I imagine placing the {color} {block_type} {depend_desc} for step {step_num}. The generated image shows this imagination — the {color} {block_type} is now positioned {depend_desc}, and I can see how the construction progresses.",
            
            # 风格2: 生成图片后，将其作为对下一步的规划
            f"<image_start>[reasoning_image_{step_num}]<image_end> THOUGHT {step_num}: For step {step_num}, I visualize adding a {color} {block_type} {depend_desc}. This generated image captures my mental picture of the state after this step. Looking at this visualization, I can better plan the subsequent additions.",
            
            # 风格3: 强调生成的图像是想象的具现化
            f"<image_start>[reasoning_image_{step_num}]<image_end> THOUGHT {step_num}: I first envision step {step_num} where the {color} {block_type} should be placed {depend_desc}. The image I generated materializes this vision. In this visualization, the block has been successfully positioned, allowing me to review and refine my construction strategy.",
            
            # 风格4: 图像作为思考过程的一部分
            f"<image_start>[reasoning_image_{step_num}]<image_end> THOUGHT {step_num}: To proceed with step {step_num}, I generate an image showing my plan: placing the {color} {block_type} {depend_desc}. This visual representation of my thought helps validate the spatial arrangement and guides the next steps in the construction sequence.",
        ]
        
        # 根据步骤选择不同的模板风格
        template_idx = (i - 1) % len(thought_templates)
        
        # 特殊处理最后一步
        if i == len(blocks) - 1:
            thought = (
                f"<image_start>[reasoning_image_{step_num}]<image_end> "
                f"THOUGHT {step_num}: For the final step {step_num}, I visualize placing the {color} {block_type} {depend_desc}. "
                f"The generated image shows the completed construction with all blocks properly positioned. This completes the building process."
            )
        else:
            thought = thought_templates[template_idx]
        
        thoughts.append(thought)
    
    return " ".join(thoughts)


def generate_final_answer(total_blocks: int) -> str:
    """生成最终答案"""
    return f"<eoc>"
    # return f"Through the above {total_blocks} steps, I have successfully built the target shape from individual blocks."


def generate_image_paths(scene_folder: Path, scene_name: str, blocks: List[Dict], unique_blocks: List[Dict], view_num: int = 1) -> Dict[str, str]:
    """生成所有图片路径的映射"""
    image_paths = {}
    
    # problem_image_1 到 problem_image_{n}: 每个积木的多视角图片
    for i, block in enumerate(unique_blocks):
        block_type = block['type']
        color = block['color']
        block_image_path = BLOCK_IMAGE_PATH_TEMPLATE.format(block_type=block_type, color=color)
        image_paths[f"problem_image_{i+1}"] = block_image_path

    final_image_path = scene_folder / "final_state" / f"{scene_name}_{view_num}.png"
    image_paths[f"problem_image_{len(unique_blocks)+1}"] = str(final_image_path)

    # problem_image_{n+2}: step 0的图片
    step0_image_path = scene_folder / "steps" / f"view_{view_num}" / f"{scene_name}_step0_{view_num}.png"
    image_paths[f"problem_image_{len(unique_blocks)+2}"] = str(step0_image_path)
    
    # reasoning_image_1 到 reasoning_image_{n-1}: 每一步的图片
    for i in range(1, len(blocks)):
        step_image_path = scene_folder / "steps" / f"view_{view_num}" / f"{scene_name}_step{i}_{view_num}.png"
        image_paths[f"reasoning_image_{i}"] = str(step_image_path)
    
    return image_paths


def process_scene(scene_folder: Path, view_num: int = 1) -> Dict[str, Any]:
    """处理单个场景，生成训练数据"""
    scene_name = scene_folder.name
    json_file = scene_folder / f"final_state.json"
    
    # 读取JSON文件
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    blocks = data['blocks']
    unique_blocks = []
    seen = set()
    for b in blocks:
        key = (b["type"], b["color"])
        if key not in seen:
            unique_blocks.append(b)
            seen.add(key)

    total_blocks = len(blocks)
    
    # 生成各个部分
    question = generate_question_template(scene_name, unique_blocks)
    thought_trace = generate_thought_trace(blocks)
    final_answer = generate_final_answer(total_blocks)
    image_paths = generate_image_paths(scene_folder, scene_name, blocks, unique_blocks, view_num)
    
    # 组合成训练数据
    training_data = {
        "Question": question,
        "Text Reasoning Trace": thought_trace,
        "Final Answer": final_answer,
    }
    
    # 添加所有图片路径
    training_data.update(image_paths)
    
    return training_data


def main():
    # 设置输入输出路径
    input_folder = Path("/lustre/fsw/portfolios/nvr/users/ymingli/projects/ljh/random_blocks/")
    output_file = Path("./ranGenTraining_views1357_imgfirst.jsonl")
    
    all_training_data = []
    
    # 遍历所有场景文件夹
    scene_folders = sorted([f for f in input_folder.iterdir() if f.is_dir()])
    
    print(f"Found {len(scene_folders)} scenes to process")
    
    for scene_folder in scene_folders:
        scene_name = scene_folder.name
        print(f"Processing scene: {scene_name}")
        
        try:
            # 对每个视角生成训练数据（这里以view_1为例，可以扩展到所有视角）
            for i in range(1,8,2):
                training_data = process_scene(scene_folder, view_num=i)
                all_training_data.append(training_data)

        except Exception as e:
            print(f"Error processing scene {scene_name}: {e}")
            continue
    
    # 保存为JSONL文件（每行一个JSON对象）
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in all_training_data:
            print(item)
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"\nGenerated {len(all_training_data)} training samples")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    main()
