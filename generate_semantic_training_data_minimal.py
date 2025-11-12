import json
import os
import argparse
from pathlib import Path
from typing import List, Dict, Any

# 定义积木名称的映射关系
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

# 要处理的视角列表（只处理1,3,5,7）
VIEWS_TO_PROCESS = [1, 3, 5, 7]

# Benchmark场景列表 (从bench文件夹中提取的100个场景)
BENCH_SCENES = {
    ("category/animal/cat", "001"),
    ("category/animal/dog", "002"),
    ("category/animal/dog", "003"),
    ("category/animal/dog", "005"),
    ("category/animal/dog", "007"),
    ("category/animal/dog", "009"),
    ("category/animal/dog", "011"),
    ("category/animal/dog", "013"),
    ("category/animal/dog", "015"),
    ("category/animal/dog", "018"),
    ("category/animal/duck", "001"),
    ("category/animal/duck", "002"),
    ("category/animal/duck", "007"),
    ("category/animal/elephant", "003"),
    ("category/animal/fish", "003"),
    ("category/animal/frog", "002"),
    ("category/animal/giraffe", "002"),
    ("category/animal/giraffe", "003"),
    ("category/animal/giraffe", "005"),
    ("category/animal/giraffe_working", "002"),
    ("category/animal/giraffe_working", "003"),
    ("category/animal/giraffe_working", "005"),
    ("category/animal/horse", "001"),
    ("category/animal/horse", "003"),
    ("category/animal/llama", "003"),
    ("category/animal/llama", "004"),
    ("category/animal/llama", "006"),
    ("category/animal/penguin", "002"),
    ("category/animal/snail", "001"),
    ("category/animal/snail", "003"),
    ("category/building/bridge", "003"),
    ("category/building/bridge", "007"),
    ("category/building/castle", "008"),
    ("category/building/castle", "011"),
    ("category/building/castle", "013"),
    ("category/building/factory", "001"),
    ("category/building/factory", "007"),
    ("category/building/gate", "002"),
    ("category/building/gate", "004"),
    ("category/building/gate", "006"),
    ("category/building/gate", "007"),
    ("category/building/gate", "009"),
    ("category/building/house", "002"),
    ("category/building/house", "007"),
    ("category/building/house", "008"),
    ("category/building/house", "011"),
    ("category/building/house", "012"),
    ("category/building/house", "014"),
    ("category/building/house", "015"),
    ("category/building/house", "017"),
    ("category/building/monument", "001"),
    ("category/building/monument", "004"),
    ("category/building/monument", "015"),
    ("category/building/nest", "002"),
    ("category/building/platform", "004"),
    ("category/building/pyramid", "002"),
    ("category/building/skyscraper", "001"),
    ("category/building/skyscraper", "004"),
    ("category/building/skyscraper", "007"),
    ("category/building/skyscraper", "008"),
    ("category/building/tower", "001"),
    ("category/building/tower", "004"),
    ("category/furniture/camera", "002"),
    ("category/furniture/chimney", "002"),
    ("category/furniture/dining_table", "001"),
    ("category/furniture/sofa", "002"),
    ("category/furniture/sofa", "006"),
    ("category/furniture/sofa", "007"),
    ("category/plant/flower", "001"),
    ("category/plant/flower", "004"),
    ("category/plant/tree", "005"),
    ("category/scene", "002"),
    ("category/scene", "003"),
    ("category/scene", "029"),
    ("category/scene", "031"),
    ("category/scene", "051"),
    ("category/scene", "056"),
    ("category/scene", "058"),
    ("category/traffic/bulldozer", "002"),
    ("category/traffic/car", "002"),
    ("category/traffic/car", "006"),
    ("category/traffic/car", "007"),
    ("category/traffic/car", "011"),
    ("category/traffic/excavator", "002"),
    ("category/traffic/housecar", "002"),
    ("category/traffic/rocket", "003"),
    ("category/traffic/rocket", "004"),
    ("category/traffic/rocket", "007"),
    ("category/traffic/rocket", "009"),
    ("category/traffic/rocket", "010"),
    ("category/traffic/rocket", "012"),
    ("category/traffic/rocket", "013"),
    ("category/traffic/tanker", "001"),
    ("category/traffic/tanker", "002"),
    ("category/traffic/tanker", "004"),
    ("category/traffic/tank", "001"),
    ("category/traffic/truck", "002"),
    ("category/traffic/truck", "006"),
    ("category/traffic/truck", "008"),
    ("category/traffic/truck", "009"),
}


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
                    block_desc = get_block_description(b)
                    depend_blocks.append(f"the {block_desc}")
                    break
    
    if len(depend_blocks) == 0:
        return "on top of the ground"
    elif len(depend_blocks) == 1:
        return f"on top of {depend_blocks[0]}"
    elif len(depend_blocks) == 2:
        return f"on top of {depend_blocks[0]} and {depend_blocks[1]}"
    else:
        return f"on top of {', '.join(depend_blocks[:-1])}, and {depend_blocks[-1]}"


def get_block_description(block: Dict) -> str:
    """获取积木的描述"""
    if 'type' in block and 'color' in block:
        return f"{block['color']} {block['type']}"
    elif 'name' in block:
        # 处理name字段：格式可能是 Type_Color_Number 或 Type_Number 或 Type_Color 或 Type
        # 需要保留类型和颜色（如果有），去掉编号
        name = block['name']
        
        # 移除.obj后缀（如果有）
        name = name.replace('.obj', '')
        
        # 分割名称
        parts = name.split('_')
        
        if len(parts) == 1:
            # 只有类型，如 "arch", "cylinder"
            return parts[0].lower()
        
        # 特殊处理：semi_cylinder 是一个完整的类型名
        if len(parts) == 2 and parts[0] == 'semi' and parts[1] == 'cylinder':
            return 'semi_cylinder'
        
        if len(parts) == 3 and parts[0] == 'semi' and parts[1] == 'cylinder':
            # semi_cylinder_001
            return 'semi_cylinder'
        
        # 检查最后一部分是否是数字（编号）
        last_is_number = parts[-1].isdigit()
        
        if len(parts) == 2:
            # 检查是否是 cuboid_2, cuboid_3 等格式（类型_数字）
            if parts[0] == 'cuboid' and parts[1].isdigit():
                # cuboid_2 或 cuboid_3 -> 返回 cuboid
                return 'cuboid'
            
            if last_is_number:
                # Type_Number，如 "cylinder_001", "triangle_002"
                return parts[0].lower()
            else:
                # Type_Color，如 "Cube_blue"
                return f"{parts[1]} {parts[0].lower()}"
        
        if len(parts) == 3:
            # 检查是否是 cuboid_2_001 或 cuboid_3_001 这种格式
            if parts[0] == 'cuboid' and parts[1].isdigit() and parts[2].isdigit():
                # cuboid_3_001 -> 返回 cuboid
                return 'cuboid'
            
            if last_is_number:
                # Type_Color_Number，如 "Cube_blue_001"
                return f"{parts[1]} {parts[0].lower()}"
            else:
                # Type_xxx_xxx
                return '_'.join(parts).lower()
        
        # 默认情况：返回原名称去掉最后的数字编号
        if last_is_number:
            result = '_'.join(parts[:-1])
            return result.lower()
        return name.lower()
    else:
        return "unknown block"


def generate_question_template(shape_name: str, blocks: List[Dict]) -> str:
    """生成问题模板 - 只包含final state和step 0两张图片"""
    total_blocks = len(blocks)
    first_block = blocks[0]

    question = (
        "My goal is to generate a visual guide for constructing a specific shape using blocks. "
        "This involves multiple steps, each requiring the addition of a new block to progressively build the final shape. "
    )

    # 只提供两张图片：final state 和 step 0
    question += (
        "The input includes an image of the final desired shape: <image_start>[problem_image_1]<image_end>, "
        "and an image showing the initial state (step 0): <image_start>[problem_image_2]<image_end>. "
    )

    first_block_desc = get_block_description(first_block)
    question += (
        f"Step 0 has been completed: a {first_block_desc} block has been placed on top of the ground. "
        f"I need to imagine and generate images of intermediate steps, starting from step 1, leading up to the final construction. "
    )
    
    return question


def generate_thought_trace(shape_name: str, blocks: List[Dict], total_blocks: int) -> str:
    """生成推理过程模板"""
    thoughts = []
    
    # THOUGHT 0
    thought_0 = (
        f"THOUGHT 0: I will begin by carefully observing the final desired shape presented in the problem image. "
        f"There are a total of {total_blocks} distinct blocks to be placed. My approach will be to execute each operation sequentially "
        f"and visualize the changes to the scene at each step to accurately build the structure. "
        f"The first block has already been placed as shown in step 0. I should keep building upon that."
    )
    thoughts.append(thought_0)
    
    # 为每个后续的block生成THOUGHT
    for i, block in enumerate(blocks[1:], start=1):
        step_num = i
        block_desc = get_block_description(block)
        depend_desc = get_depend_description(block, blocks)
        
        # 生成不同风格的描述
        thought_templates = [
            f"THOUGHT {step_num}: Place a {block_desc} {depend_desc} to continue the build. I will now generate an updated visual for step {step_num}. The updated image reflects the new block placement for step {step_num}. <image_start>[reasoning_image_{step_num}]<image_end>",
            f"THOUGHT {step_num}: For step {step_num}, the plan is to add the {block_desc} {depend_desc}. Then I produce the image for this step. <image_start>[reasoning_image_{step_num}]<image_end>",
            f"THOUGHT {step_num}: Step {step_num}: I first write the reasoning — place a {block_desc} block {depend_desc}. After finalizing the reasoning, I generate the image showing the state after step {step_num}. <image_start>[reasoning_image_{step_num}]<image_end>",
            f"THOUGHT {step_num}: Reasoning {step_num}: Add {block_desc} {depend_desc}. After this reasoning, I render the scene for step {step_num}. The scene now includes the new block as expected. <image_start>[reasoning_image_{step_num}]<image_end>",
        ]
        
        # 根据步骤选择不同的模板风格
        template_idx = i % len(thought_templates)
        
        # 特殊处理最后一步
        if i == len(blocks) - 1:
            thought = (
                f"THOUGHT {step_num}: Finally, place a {block_desc} {depend_desc} to finish the build. "
                f"<image_start>[reasoning_image_{step_num}]<image_end>"
            )
        else:
            thought = thought_templates[template_idx]
        
        thoughts.append(thought)
    
    return " ".join(thoughts)


def generate_final_answer(total_blocks: int) -> str:
    """生成最终答案"""
    return f"<eoc>"


def check_images_exist(image_paths: Dict[str, str]) -> tuple:
    """检查所有图片路径是否存在
    
    Returns:
        (all_exist, missing_paths): 所有图片是否都存在，以及缺失的图片路径列表
    """
    missing_paths = []
    for key, path in image_paths.items():
        if not Path(path).exists():
            missing_paths.append(f"{key}: {path}")
    
    return len(missing_paths) == 0, missing_paths


def generate_image_paths(scene_folder: Path, shape_name: str, blocks: List[Dict], view_num: int = 0) -> Dict[str, str]:
    """生成所有图片路径的映射 - 只包含final state和step 0"""
    image_paths = {}
    
    # problem_image_1: 最终状态图片 (使用最后一个step的图片)
    final_step_num = len(blocks) - 1
    final_image_path = scene_folder / "steps" / f"view_{view_num}" / f"{shape_name}_step{final_step_num}_{view_num}.png"
    image_paths["problem_image_1"] = str(final_image_path)
    
    # problem_image_2: step 0的图片
    step0_image_path = scene_folder / "steps" / f"view_{view_num}" / f"{shape_name}_step0_{view_num}.png"
    image_paths["problem_image_2"] = str(step0_image_path)
    
    # reasoning_image_1 到 reasoning_image_{n-1}: 每一步的图片 (n = len(blocks))
    for i in range(1, len(blocks)):
        step_image_path = scene_folder / "steps" / f"view_{view_num}" / f"{shape_name}_step{i}_{view_num}.png"
        image_paths[f"reasoning_image_{i}"] = str(step_image_path)
    
    return image_paths


def process_scene(scene_folder: Path, view_num: int = 0) -> Dict[str, Any]:
    """处理单个场景，生成训练数据"""
    # 找到JSON文件（可能是shape_name.json或其他名称）
    json_files = list(scene_folder.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON file found in {scene_folder}")
    
    json_file = json_files[0]
    shape_name = json_file.stem
    
    # 读取JSON文件
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    blocks = data['blocks']
    total_blocks = len(blocks)
    
    # 生成各个部分
    question = generate_question_template(shape_name, blocks)
    thought_trace = generate_thought_trace(shape_name, blocks, total_blocks)
    final_answer = generate_final_answer(total_blocks)
    image_paths = generate_image_paths(scene_folder, shape_name, blocks, view_num)
    
    # 检查所有图片路径是否存在
    all_exist, missing_paths = check_images_exist(image_paths)
    if not all_exist:
        raise FileNotFoundError(f"Missing images for scene {scene_folder.name} view {view_num}:\n" + "\n".join(missing_paths))
    
    # 组合成训练数据
    training_data = {
        "Question": question,
        "Text Reasoning Trace": thought_trace,
        "Final Answer": final_answer,
        "blocks": blocks,  # 保留原始blocks信息
    }
    
    # 添加所有图片路径
    training_data.update(image_paths)
    
    return training_data


def find_all_scene_folders(root_path: Path) -> List[tuple]:
    """递归查找所有包含JSON文件和steps文件夹的场景文件夹
    返回: List[(scene_folder_path, category_path, subcategory_name, scene_name)]
    """
    scene_folders = []
    
    # 遍历category和long两个主目录
    for main_dir in ['category', 'long']:
        main_path = root_path / main_dir
        if not main_path.exists():
            continue
            
        # 遍历每个分类（animal, building等）
        for category_dir in main_path.iterdir():
            if not category_dir.is_dir():
                continue
            
            category_name = category_dir.name
            
            # 遍历每个子分类（cat, dog等）
            for subcategory_dir in category_dir.iterdir():
                if not subcategory_dir.is_dir():
                    continue
                
                subcategory_name = subcategory_dir.name
                
                # 遍历每个场景编号（001, 002等）
                for scene_dir in subcategory_dir.iterdir():
                    if not scene_dir.is_dir():
                        continue
                    
                    # 检查是否包含必要的文件和文件夹
                    has_json = any(scene_dir.glob("*.json"))
                    has_steps = (scene_dir / "steps").exists()
                    has_final = (scene_dir / "final_state").exists()
                    
                    if has_json and has_steps and has_final:
                        scene_name = scene_dir.name
                        category_path = f"{main_dir}/{category_name}/{subcategory_name}"
                        scene_folders.append((scene_dir, category_path, subcategory_name, scene_name))
    
    return scene_folders


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Generate minimal semantic training data with only final state and step 0 images')
    parser.add_argument('--source-dir',
                        default='/lustre/fsw/portfolios/nvr/users/ymingli/projects/ljh/semantic_all/semantic_all/semantic_blocks',
                        help='Root directory for semantic_blocks (default: semantic_all/semantic_blocks)')
    parser.add_argument('--output-dir',
                        default='/lustre/fsw/portfolios/nvr/users/ymingli/projects/ljh',
                        help='Output directory for JSONL files (default: current data folder)')
    
    args = parser.parse_args()
    
    # 设置输入输出路径
    input_folder = Path(args.source_dir)
    output_folder = Path(args.output_dir)
    
    # 使用全局常量 VIEWS_TO_PROCESS (1, 3, 5, 7)
    view_nums = VIEWS_TO_PROCESS
    
    # 为所有视角和单独视角分别创建数据容器
    all_views_data = []  # 包含所有视角的数据
    single_view_data = {view: [] for view in view_nums}  # 每个视角单独的数据
    
    # 查找所有场景文件夹
    scene_folders = find_all_scene_folders(input_folder)
    
    print(f"Found {len(scene_folders)} scenes to process")
    print(f"Processing views: {view_nums}")
    print("Generating minimal training data with only 2 problem images (final state + step 0)")
    
    for scene_folder, category_path, subcategory_name, scene_name in scene_folders:
        print(f"Processing: {category_path}/{scene_name}")
        
        try:
            # 对每个视角生成训练数据
            for view_num in view_nums:
                training_data = process_scene(scene_folder, view_num=view_num)
                
                # 添加元数据
                training_data["category"] = category_path
                training_data["subcategory"] = subcategory_name
                training_data["scene_name"] = scene_name
                
                # 添加到所有视角的集合中
                all_views_data.append(training_data)
                
                # 添加到对应单独视角的集合中
                single_view_data[view_num].append(training_data)
            
        except Exception as e:
            print(f"Error processing scene {category_path}/{scene_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 创建输出目录
    output_folder.mkdir(exist_ok=True)
    
    # 保存所有视角的JSONL文件
    all_views_file = output_folder / "semantic_training_all_views_minimal.jsonl"
    with open(all_views_file, 'w', encoding='utf-8') as f:
        for item in all_views_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"\nGenerated {len(all_views_data)} training samples (all views)")
    print(f"Saved to: {all_views_file}")
    
    # 保存每个单独视角的JSONL文件
    for view_num in view_nums:
        view_file = output_folder / f"semantic_training_view{view_num}_minimal.jsonl"
        with open(view_file, 'w', encoding='utf-8') as f:
            for item in single_view_data[view_num]:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"Generated {len(single_view_data[view_num])} training samples (view {view_num})")
        print(f"Saved to: {view_file}")


if __name__ == "__main__":
    main()

# python generate_semantic_training_data_minimal.py
