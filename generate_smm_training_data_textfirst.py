import json
import os
from pathlib import Path
from typing import List, Dict, Any

# 定义积木名称的映射关系（基于示例数据）
BLOCK_NAME_MAPPING = {
    "arch": "arch",
    "cube": "cube",
    "semi_cylinder": "semi_cylinder",
    "triangle": "triangle",
    "cylinder": "cylinder",
    "cuboid2": "cuboid2",
    "cuboid3": "cuboid3",
    "cuboid5": "cuboid5",
    "extra_long_plate": "extra_long_plate",
}

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
    
    question = (
        f"Based on the construction task shown below, follow the instructions to complete the build. "
        f"Given the final desired shape of blocks shown in the first image<image_start>[problem_image_1]<image_end> "
        f"which is viewed from a {view_angle} angle, perform a series of specified manipulations. "
        f"This involves multiple steps, each requiring the addition of a new block to progressively build the final shape. "
        f"The initial input also includes {total_blocks} images of multiple blocks that will be used."
    )
    
    # 添加每个积木的图片占位符
    for i in range(total_blocks):
        question += f"<image_start>[problem_image_{i+2}]<image_end>"
    
    # 添加step 0的描述
    first_block = blocks[0]
    question += (
        f" Step 0 has been completed: a {first_block['color']} {first_block['type']} block has been placed "
        f"on top of the ground. The image after step 0 is provided."
        f"<image_start>[problem_image_{total_blocks+2}]<image_end>"
    )
    
    return question


def generate_thought_trace(scene_name: str, blocks: List[Dict], total_blocks: int) -> str:
    """生成推理过程模板"""
    thoughts = []
    
    # THOUGHT 0
    thought_0 = (
        f"THOUGHT 0: I will begin by carefully observing the final desired shape of blocks presented in the problem image. "
        f"There are a total of {total_blocks} distinct blocks. My approach will be to execute each operation sequentially "
        f"and visualize the changes to the scene at each step to accurately build the blocks. "
        f"The first block has already been placed as described in the step 0. I should keep building upon that."
    )
    thoughts.append(thought_0)
    
    # 为每个后续的block生成THOUGHT
    for i, block in enumerate(blocks[1:], start=1):
        step_num = i
        color = block['color']
        block_type = block['type']
        depend_desc = get_depend_description(block, blocks)
        
        # 生成不同风格的描述
        thought_templates = [
            f"THOUGHT {step_num}: Place a {color} {block_type} {depend_desc} to continue the build. I will now generate an updated visual for step {step_num}. The updated image reflects the new block placement for step {step_num}. <image_start>[reasoning_image_{step_num}]<image_end>",
            f"THOUGHT {step_num}: For step {step_num}, the plan is to add the {color} {block_type} {depend_desc}. Then I produce the image for this step. <image_start>[reasoning_image_{step_num}]<image_end>",
            f"THOUGHT {step_num}: Step {step_num}: I first write the reasoning — place a {color} {block_type} block {depend_desc}. After finalizing the reasoning, I generate the image showing the state after step {step_num}. <image_start>[reasoning_image_{step_num}]<image_end>",
            f"THOUGHT {step_num}: Reasoning {step_num}: Add {color} {block_type} {depend_desc}. After this reasoning, I render the scene for step {step_num}. The scene now includes the new block as expected. <image_start>[reasoning_image_{step_num}]<image_end>",
        ]
        
        # 根据步骤选择不同的模板风格
        template_idx = i % len(thought_templates)
        
        # 特殊处理最后一步
        if i == len(blocks) - 1:
            thought = (
                f"THOUGHT {step_num}: Finally, place a {color} {block_type} {depend_desc} to finish the build. "
                f"<image_start>[reasoning_image_{step_num}]<image_end>"
            )
        else:
            thought = thought_templates[template_idx]
        
        thoughts.append(thought)
    
    return " ".join(thoughts)


def generate_final_answer(total_blocks: int) -> str:
    """生成最终答案"""
    # return f"Through the above {total_blocks} steps, I have successfully built the target shape from individual blocks."
    return f"<eoc>"


def generate_image_paths(scene_folder: Path, scene_name: str, blocks: List[Dict], view_num: int = 1) -> Dict[str, str]:
    """生成所有图片路径的映射"""
    image_paths = {}
    
    # problem_image_1: 最终状态图片
    final_image_path = scene_folder / "final_state" / f"{scene_name}_final_{view_num}.png"
    image_paths["problem_image_1"] = str(final_image_path)
    
    # problem_image_2 到 problem_image_{n+1}: 每个积木的多视角图片
    for i, block in enumerate(blocks):
        block_type = block['type']
        color = block['color']
        block_image_path = BLOCK_IMAGE_PATH_TEMPLATE.format(block_type=block_type, color=color)
        image_paths[f"problem_image_{i+2}"] = block_image_path
    
    # problem_image_{n+2}: step 0的图片
    step0_image_path = scene_folder / "steps" / f"view_{view_num}" / f"{scene_name}_step0_{view_num}.png"
    image_paths[f"problem_image_{len(blocks)+2}"] = str(step0_image_path)
    
    # reasoning_image_1 到 reasoning_image_{n-1}: 每一步的图片
    for i in range(1, len(blocks)):
        step_image_path = scene_folder / "steps" / f"view_{view_num}" / f"{scene_name}_step{i}_{view_num}.png"
        image_paths[f"reasoning_image_{i}"] = str(step_image_path)
    
    return image_paths


def process_scene(scene_folder: Path, view_num: int = 1) -> Dict[str, Any]:
    """处理单个场景，生成训练数据"""
    scene_name = scene_folder.name
    json_file = scene_folder / f"{scene_name}.json"
    
    # 读取JSON文件
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    blocks = data['blocks']
    total_blocks = len(blocks)
    
    # 生成各个部分
    question = generate_question_template(scene_name, blocks)
    thought_trace = generate_thought_trace(scene_name, blocks, total_blocks)
    final_answer = generate_final_answer(total_blocks)
    image_paths = generate_image_paths(scene_folder, scene_name, blocks, view_num)
    
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
    output_file = Path("./ranGenTraining_views1357_textfirst.jsonl")
    
    
    all_training_data = []
    
    # 遍历所有场景文件夹
    scene_folders = sorted([f for f in input_folder.iterdir() if f.is_dir()])
    
    print(f"Found {len(scene_folders)} scenes to process")
    
    for scene_folder in scene_folders:
        scene_name = scene_folder.name
        print(f"Processing scene: {scene_name}")
        
        try:
            # 对每个视角生成训练数据（这里以view_1为例，可以扩展到所有视角）
            training_data = process_scene(scene_folder, view_num=1)
            all_training_data.append(training_data)
            
        except Exception as e:
            print(f"Error processing scene {scene_name}: {e}")
            continue
    
    # 保存为JSONL文件（每行一个JSON对象）
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in all_training_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"\nGenerated {len(all_training_data)} training samples")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    main()
