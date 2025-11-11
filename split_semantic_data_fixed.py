import json
import re
from math import ceil
from pathlib import Path

# 配置输入输出路径
BASE_DIR = Path("/lustre/fsw/portfolios/nvr/users/ymingli/projects/ljh")
VIEWS_TO_PROCESS = [1, 3, 5, 7]  # 要处理的视角列表

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
    ("category/scenes/scene", "002"),
    ("category/scenes/scene", "003"),
    ("category/scenes/scene", "029"),
    ("category/scenes/scene", "031"),
    ("category/scenes/scene", "051"),
    ("category/scenes/scene", "056"),
    ("category/scenes/scene", "058"),
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

pat_problem = re.compile(r"^problem_image_(\d+)$")
pat_reasoning = re.compile(r"^reasoning_image_(\d+)$")

def collect_problem_keys(obj):
    """收集并排序所有problem_image_*键"""
    ks = []
    for k in obj.keys():
        m = pat_problem.match(k)
        if m:
            ks.append((int(m.group(1)), k))
    return [k for _, k in sorted(ks, key=lambda x: x[0])]

def collect_reasoning_keys(obj):
    """收集并排序所有reasoning_image_*键"""
    ks = []
    for k in obj.keys():
        m = pat_reasoning.match(k)
        if m:
            ks.append((int(m.group(1)), k))
    return [k for _, k in sorted(ks, key=lambda x: x[0])]

def split_reasoning_by_thought(reasoning_text: str):
    """按THOUGHT分割推理文本，保持每个THOUGHT完整
    
    返回的chunks列表：
    - 如果有THOUGHT 0: chunks = [THOUGHT 0, THOUGHT 1, THOUGHT 2, ..., THOUGHT N]
    - 每个THOUGHT包含完整的文本，包括其引用的reasoning_image
    """
    # 使用THOUGHT作为分隔符进行分割
    # 分割后会得到：['', ' 0: ...', ' 1: ...', ...]
    parts = re.split(r'THOUGHT\s+', reasoning_text)
    
    if len(parts) <= 1:
        return [reasoning_text] if reasoning_text.strip() else []
    
    chunks = []
    # parts[0]是空字符串（THOUGHT之前的内容，应该为空）
    # parts[1:]是每个THOUGHT的内容（不包括"THOUGHT"关键字本身）
    for i, part in enumerate(parts[1:]):
        # 重新添加"THOUGHT"关键字和编号
        chunk = f"THOUGHT {part.strip()}"
        chunks.append(chunk)
    
    return chunks

def renumber_thoughts(text: str, start_num: int = 0):
    """重新编号THOUGHT"""
    counter = [start_num]
    def replacer(match):
        result = f"THOUGHT {counter[0]}:"
        counter[0] += 1
        return result
    return re.sub(r'THOUGHT\s+\d+:', replacer, text)

def renumber_steps(text: str, start_num: int = 1):
    """重新编号Step"""
    counter = [start_num]
    def replacer(match):
        result = f"Step {counter[0]}:"
        counter[0] += 1
        return result
    return re.sub(r'Step\s+\d+:', replacer, text)

def renumber_reasoning_images(text: str, old_to_new_map: dict):
    """重新编号reasoning_image引用"""
    def replacer(match):
        old_num = int(match.group(1))
        new_num = old_to_new_map.get(old_num, old_num)
        return f"<image_start>[reasoning_image_{new_num}]<image_end>"
    return re.sub(r'<image_start>\[reasoning_image_(\d+)\]<image_end>', replacer, text)

def renumber_problem_images(text: str, old_to_new_map: dict):
    """重新编号problem_image引用"""
    def replacer(match):
        old_num = int(match.group(1))
        new_num = old_to_new_map.get(old_num, old_num)
        if new_num is None:
            return match.group(0)  # 保持不变
        return f"<image_start>[problem_image_{new_num}]<image_end>"
    return re.sub(r'<image_start>\[problem_image_(\d+)\]<image_end>', replacer, text)

def update_block_count(text: str, new_count: int):
    """更新积木总数"""
    return re.sub(r'There are a total of \d+ distinct blocks', 
                  f'There are a total of {new_count} distinct blocks', text)

def get_block_description_from_step(chunk: str):
    """从步骤文本中提取积木描述"""
    patterns = [
        r'place a (.*?) block',
        r'add (?:the )?(.*?)(?: block)?\s+(?:on top of|to)',
        r'Finally, place a (.*?)(?: block)?\s+',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, chunk, re.IGNORECASE)
        if match:
            desc = match.group(1).strip()
            desc = re.sub(r'\s+', ' ', desc)
            return desc
    
    return "the next block"

def process_line(obj):
    """
    处理单行数据，决定是否需要分割
    返回：
      - None: 跳过该行（2*x + y > 22）
      - [obj]: 保持原样（x+y <= 11）
      - [obj1, obj2]: 分割后的两个对象
    """
    problem_keys = collect_problem_keys(obj)
    reasoning_keys = collect_reasoning_keys(obj)
    
    x = len(problem_keys)
    y = len(reasoning_keys)
    
    # 规则1: 推理步数 <= 10，保持原样
    if y <= 10:
        return [obj]
    
    # 规则2: 推理步数 > 20，跳过（太长）
    if y > 20:
        return None
    
    # 规则3: 需要分割 (10 < y <= 20)
    if y == 0 or "Question" not in obj or "Text Reasoning Trace" not in obj:
        return [obj]
    
    # 分割点：保留前k个reasoning steps
    k = ceil(y / 2)
    k = max(1, min(k, y))
    
    # 解析推理文本
    reasoning_text = obj.get("Text Reasoning Trace", "")
    chunks = split_reasoning_by_thought(reasoning_text)
    
    if not chunks:
        return [obj]
    
    # 确定THOUGHT 0的位置
    has_thought0 = chunks[0].startswith("THOUGHT 0:")
    
    # chunks包含所有的THOUGHT（包括THOUGHT 0和THOUGHT 1-y）
    # 但reasoning_image只有1到y-1
    # 如果有THOUGHT 0，chunks = [THOUGHT 0, THOUGHT 1, ..., THOUGHT y-1]，共y个
    # 我们需要保留 THOUGHT 0 + THOUGHT 1 到 THOUGHT k-1（因为THOUGHT k对应reasoning_image_k）
    
    if has_thought0:
        # chunks[0] = THOUGHT 0
        # chunks[1] = THOUGHT 1 (对应 reasoning_image_1)
        # chunks[k] = THOUGHT k (对应 reasoning_image_k)
        # 我们保留 chunks[0] 到 chunks[k]（共k+1个），即THOUGHT 0到THOUGHT k
        first_chunks = chunks[:k+1]  # THOUGHT 0 + THOUGHT 1-k
        second_chunks = chunks[k+1:]  # THOUGHT (k+1) to end
    else:
        # 没有THOUGHT 0的情况（旧数据格式）
        first_chunks = chunks[:k]
        second_chunks = chunks[k:]
    
    # ========== 构建第一部分 ==========
    first_obj = {}
    
    # Question保持不变
    first_obj["Question"] = obj.get("Question", "")
    
    # Text Reasoning Trace
    first_obj["Text Reasoning Trace"] = " ".join(first_chunks).strip()
    
    # ✅ 第一部分有Final Answer（作为结束符）
    if "Final Answer" in obj:
        first_obj["Final Answer"] = obj["Final Answer"]
    
    # 保留候选积木的problem_image（1到n）
    for i, pk in enumerate(problem_keys):
        if i < len(problem_keys) - 2:  # 只保留候选积木，不包括final_image和step0_image
            first_obj[pk] = obj[pk]
    
    # 第一部分的目标图片应该是step k，而不是最终状态
    step_k_key = f"reasoning_image_{k}"
    if step_k_key in obj:
        # problem_image_{n+1} = step k的图片（第一部分的目标）
        final_image_key = problem_keys[-2]  # 倒数第二个是final_image
        first_obj[final_image_key] = obj[step_k_key]
    
    # 保留step 0的图片
    step0_key = problem_keys[-1]  # 最后一个是step0_image
    first_obj[step0_key] = obj[step0_key]
    
    # 保留前k个reasoning_image_*（包括第k个，因为Text Reasoning Trace中会引用它）
    for rk in reasoning_keys:
        m = pat_reasoning.match(rk)
        if m:
            idx = int(m.group(1))
            if idx <= k:  # 保留1到k的所有reasoning_image
                first_obj[rk] = obj[rk]
    
    # ========== 构建第二部分 ==========
    second_obj = {}
    
    # 提取最后一步的积木描述
    if first_chunks:
        last_chunk = first_chunks[-1]
        block_desc = get_block_description_from_step(last_chunk)
    else:
        block_desc = "blocks"
    
    # 计算剩余积木数量 = 剩余reasoning步骤数
    remaining_blocks = y - k
    
    # 构建新的Question
    original_q = obj.get("Question", "")
    
    if "Step 0 has been completed:" in original_q:
        # 提取原始Question的候选积木部分（保持不变）
        # 格式：从开头到"Step 0 has been completed:"之前
        step0_match = re.search(r'(.*?)(Step 0 has been completed:.*)', original_q, re.DOTALL)
        if step0_match:
            # 保留原始的候选积木列表部分
            candidate_blocks_part = step0_match.group(1)
            
            # 构建新的Step k completed部分
            # 注意：在第二部分中，step k的图片已经被移到problem_image_{n+2}位置
            last_problem_idx = len(problem_keys)  # 最后一个problem_image的索引
            new_step_part = (
                f"Previous {k} steps have been completed. "
                f"The image after {k} steps is provided: <image_start>[problem_image_{last_problem_idx}]<image_end>. "
            )
            
            second_obj["Question"] = candidate_blocks_part + new_step_part
        else:
            # Fallback
            second_obj["Question"] = original_q
    else:
        # Fallback
        second_obj["Question"] = original_q
    
    # ✅ 保留所有候选积木图片（1到n）
    num_blocks = len(problem_keys) - 2  # 减去final_image和step0_image
    
    for i, pk in enumerate(problem_keys):
        if i < len(problem_keys) - 2:  # 保留候选积木
            second_obj[pk] = obj[pk]
    
    # 第二部分的目标图片应该是原始的最终状态（step y）
    final_image_key = problem_keys[-2]  # 倒数第二个是final_image
    second_obj[final_image_key] = obj[final_image_key]  # 保留原始的最终目标
    
    # 添加reasoning_image_k作为新的起始状态（替代原来的step 0）
    step_k_key = f"reasoning_image_{k}"
    if step_k_key in obj:
        # 使用problem_image的最后一个key位置（原来的step 0位置）
        last_problem_key = problem_keys[-1]
        second_obj[last_problem_key] = obj[step_k_key]
    
    # Text Reasoning Trace: 剩余的推理步骤，重新编号
    second_reasoning = " ".join(second_chunks).strip()
    
    # 创建reasoning_image的映射
    old_to_new = {}
    new_r_idx = 1
    for rk in reasoning_keys:
        m = pat_reasoning.match(rk)
        if m:
            old_idx = int(m.group(1))
            if old_idx > k:
                old_to_new[old_idx] = new_r_idx
                new_key = f"reasoning_image_{new_r_idx}"
                second_obj[new_key] = obj[rk]
                new_r_idx += 1
    
    # 重新编号THOUGHT, Step, reasoning_image
    second_reasoning = renumber_thoughts(second_reasoning, start_num=0)
    second_reasoning = renumber_steps(second_reasoning, start_num=1)
    second_reasoning = renumber_reasoning_images(second_reasoning, old_to_new)
    
    # ✅ 不修改THOUGHT 0中的积木总数（保持与原始一致）
    # second_reasoning = update_block_count(second_reasoning, remaining_blocks)  # 注释掉这行
    
    second_obj["Text Reasoning Trace"] = second_reasoning
    
    # ✅ 第二部分也有Final Answer
    if "Final Answer" in obj:
        second_obj["Final Answer"] = obj["Final Answer"]
    
    # 保留元数据
    for meta_key in ["category", "subcategory", "scene_name"]:
        if meta_key in obj:
            first_obj[meta_key] = obj[meta_key]
            second_obj[meta_key] = obj[meta_key]
    
    return [first_obj, second_obj]

def process_file(input_path: Path):
    """处理单个文件并返回所有数据和短数据"""
    total = 0
    skipped = 0
    split_count = 0
    all_data = []
    short_data = []  # 只包含原始推理步数<11的数据
    
    with input_path.open("r", encoding="utf-8") as fin:
        for line_num, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"⚠️ Line {line_num} JSON decode error — skip.")
                skipped += 1
                continue
            
            total += 1
            
            # 在处理之前，检查原始推理步数
            original_reasoning_keys = collect_reasoning_keys(obj)
            original_y = len(original_reasoning_keys)
            
            result = process_line(obj)
            
            if result is None:
                skipped += 1
                continue
            
            if len(result) == 1:
                all_data.append(result[0])
                # 只有原始推理步数<11的才加入short_data
                if original_y < 11:
                    short_data.append(result[0])
            else:
                # 分割成两部分
                all_data.append(result[0])
                all_data.append(result[1])
                # 原始数据被分割了，说明原始推理步数>=11，不加入short_data
                split_count += 1
    
    return all_data, short_data, total, skipped, split_count


def main():
    # 创建输出文件夹
    train_dir = BASE_DIR / "train"
    bench_dir = BASE_DIR / "bench"
    train_short_dir = BASE_DIR / "train_short"
    train_dir.mkdir(exist_ok=True)
    bench_dir.mkdir(exist_ok=True)
    train_short_dir.mkdir(exist_ok=True)
    
    print(f"Processing {len(VIEWS_TO_PROCESS)} views and all_views file...")
    print(f"Using {len(BENCH_SCENES)} predefined bench scenes\n")
    
    # 处理所有视角文件
    all_files = {
        "all_views": BASE_DIR / "semantic_training_all_views_textfirst.jsonl"
    }
    for view_num in VIEWS_TO_PROCESS:
        all_files[f"view{view_num}"] = BASE_DIR / f"semantic_training_view{view_num}_textfirst.jsonl"
    
    for file_key, input_path in all_files.items():
        if not input_path.exists():
            print(f"⚠️ File not found: {input_path}")
            continue
        
        print(f"📄 Processing {file_key}: {input_path.name}")
        
        # 处理文件
        all_data, short_data, total, skipped, split_count = process_file(input_path)
        
        # 定义最大图片数限制
        MAX_IMAGES = 20
        
        print(f"   ✅ Total input lines: {total}")
        print(f"   ✅ Lines written: {len(all_data)}")
        print(f"   ⚠️ Lines skipped: {skipped}")
        print(f"   📊 Lines split into 2 parts: {split_count}")
        print(f"   📝 Short data (original y<11): {len(short_data)}")
        
        # 划分train和bench，同时过滤掉图片过多的样本
        train_data = []
        bench_data = []
        train_short_data = []
        filtered_by_images = 0
        print(f"   🗑️ Filtered by image count (>{MAX_IMAGES}): {filtered_by_images}")
        
        # 划分train和bench，同时过滤掉图片过多的样本
        MAX_IMAGES = 20  # 最多允许20张图片
        train_data = []
        bench_data = []
        train_short_data = []
        filtered_by_images = 0
        
        for item in all_data:
            # 统计图片数量
            num_images = len([k for k in item.keys() if 'image' in k])
            
            # 过滤掉图片过多的样本
            if num_images > MAX_IMAGES:
                filtered_by_images += 1
                continue
            
            category = item.get("category", "")
            scene_name = item.get("scene_name", "")
            
            if (category, scene_name) in BENCH_SCENES:
                bench_data.append(item)
            else:
                train_data.append(item)
        
        # train_short只包含原始推理步数<11且不在bench中的数据
        for item in short_data:
            # 统计图片数量
            num_images = len([k for k in item.keys() if 'image' in k])
            
            # 过滤掉图片过多的样本
            if num_images > MAX_IMAGES:
                continue
            
            category = item.get("category", "")
            scene_name = item.get("scene_name", "")
            
            if (category, scene_name) not in BENCH_SCENES:
                train_short_data.append(item)
        
        # 保存train文件
        train_file = train_dir / f"semantic_training_{file_key}_textfirst.jsonl"
        with train_file.open("w", encoding="utf-8") as f:
            for item in train_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"   ✅ Train set: {len(train_data)} samples → {train_file}")
        
        # 保存bench文件
        bench_file = bench_dir / f"semantic_training_{file_key}_textfirst.jsonl"
        with bench_file.open("w", encoding="utf-8") as f:
            for item in bench_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"   ✅ Bench set: {len(bench_data)} samples → {bench_file}")
        
        # 保存train_short文件
        train_short_file = train_short_dir / f"semantic_training_{file_key}_textfirst.jsonl"
        with train_short_file.open("w", encoding="utf-8") as f:
            for item in train_short_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"   ✅ Train short set (y<11): {len(train_short_data)} samples → {train_short_file}")
        print()
    
    print(f"🎉 All done!")
    print(f"📁 Train files saved to: {train_dir}")
    print(f"📁 Bench files saved to: {bench_dir}")
    print(f"📁 Train short files (y<11) saved to: {train_short_dir}")

if __name__ == "__main__":
    main()
    '''
python split_semantic_data_fixed.py
'''