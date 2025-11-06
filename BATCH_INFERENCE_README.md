# Batch Inference for Bagel Model

批量推理脚本，用于处理JSONL格式的数据集，生成模型输出并保存为便于评估的格式。

## 文件说明

- `batch_inference.py`: 主要的批量推理脚本
- `evaluate_results.py`: 结果评估和统计脚本
- `run_batch_inference.sh`: 批量推理示例脚本
- `run_evaluation.sh`: 评估示例脚本

## 输入数据格式

输入JSONL文件中的每条数据应包含以下字段：

```json
{
    "Question": "问题文本，包含<image_start>[problem_image_X]<image_end>占位符",
    "Text Reasoning Trace": "真实的推理轨迹（用于评估）",
    "Final Answer": "真实答案（用于评估）",
    "problem_image_1": "/path/to/image1.png",
    "problem_image_2": "/path/to/image2.png",
    ...
    "reasoning_image_1": "/path/to/ground_truth_step1.png",
    "reasoning_image_2": "/path/to/ground_truth_step2.png",
    ...
}
```

## 使用方法

### 1. 批量推理

基本用法：

```bash
python batch_inference.py \
    --input_jsonl path/to/input.jsonl \
    --output_dir ./output \
    --checkpoint_dir path/to/checkpoint \
    --checkpoint_file model.safetensors
```

完整参数说明：

```bash
python batch_inference.py \
    --input_jsonl path/to/input.jsonl \          # 输入JSONL文件路径
    --output_dir ./output \                       # 输出目录
    --checkpoint_dir path/to/checkpoint \         # 模型检查点目录
    --checkpoint_file model.safetensors \         # 检查点文件名
    --do_sample \                                 # 是否使用采样（默认False）
    --text_temperature 0.0 \                      # 文本生成温度
    --cfg_text_scale 4.0 \                        # CFG文本缩放
    --cfg_img_scale 2.0 \                         # CFG图像缩放
    --num_timesteps 50 \                          # 扩散步数
    --start_idx 0 \                               # 处理起始索引
    --end_idx 100 \                               # 处理结束索引（可选）
    --seed 42                                     # 随机种子
```

或使用提供的脚本：

```bash
bash run_batch_inference.sh
```

### 2. 结果评估

```bash
python evaluate_results.py \
    --results_jsonl ./output/batch_inference_xxx/results.jsonl \
    --output_eval_json ./output/batch_inference_xxx/evaluation_data.json
```

或使用提供的脚本：

```bash
bash run_evaluation.sh
```

## 输出格式

批量推理完成后，会在输出目录中生成以下结构：

```
output/
└── batch_inference_20250104_120000/
    ├── results.jsonl                    # 所有结果（JSONL格式）
    ├── summary.json                      # 推理摘要信息
    ├── evaluation_data.json             # 评估格式数据（运行评估后生成）
    ├── statistics.json                  # 统计信息（运行评估后生成）
    └── output_images/
        ├── 0/                           # 第一个样本（sample_id=0）的图片
        │   ├── problem_image_1.png      # 问题图片（复制）
        │   ├── problem_image_2.png
        │   ├── output_image_1.png       # 模型生成的图片
        │   ├── output_image_2.png
        │   └── ...
        ├── 1/                           # 第二个样本（sample_id=1）的图片
        │   ├── problem_image_1.png
        │   ├── output_image_1.png
        │   └── ...
        └── ...
```

### results.jsonl 格式

每行包含原始数据加上模型输出：

```json
{
    "Question": "原始问题",
    "Text Reasoning Trace": "真实推理轨迹",
    "Final Answer": "真实答案",
    "problem_image_1": "原始图片路径",
    ...
    "reasoning_image_1": "真实推理图片路径",
    ...
    "output_reasoning": "模型生成的推理文本",
    "output_images": [
        "output/batch_inference_xxx/output_images/0/output_image_1.png",
        "output/batch_inference_xxx/output_images/0/output_image_2.png"
    ],
    "num_reasoning_steps": 4,
    "num_generated_images": 3,
    "sample_id": 0
}
```

### evaluation_data.json 格式

简化的评估格式：

```json
[
    {
        "sample_id": 0,
        "question": "问题文本",
        "ground_truth_reasoning": "真实推理",
        "ground_truth_answer": "真实答案",
        "ground_truth_images": ["路径1", "路径2"],
        "output_reasoning": "模型推理",
        "output_images": ["输出路径1", "输出路径2"],
        "num_reasoning_steps": 4,
        "num_generated_images": 3
    },
    ...
]
```

### statistics.json

包含详细统计信息：

```json
{
    "total_samples": 100,
    "successful": 98,
    "failed": 2,
    "avg_reasoning_steps": 4.5,
    "avg_generated_images": 3.2,
    "reasoning_steps_distribution": {
        "3": 10,
        "4": 45,
        "5": 43
    },
    "image_count_distribution": {
        "2": 5,
        "3": 50,
        "4": 43
    }
}
```

## 特点

1. **完整保存**: 所有生成的图片都保存到独立的文件夹中，便于后续检查
2. **评估友好**: 输出格式包含真实值和预测值，便于计算评估指标
3. **错误处理**: 如果某个样本失败，会记录错误信息并继续处理其他样本
4. **断点续传**: 可以通过 `--start_idx` 和 `--end_idx` 参数分批处理数据
5. **详细统计**: 自动生成推理步数、生成图片数量等统计信息

## 后续评估建议

使用生成的 `evaluation_data.json` 文件，可以进行以下评估：

1. **文本评估**: 比较 `output_reasoning` 和 `ground_truth_reasoning`
   - BLEU, ROUGE, BERTScore 等指标
   
2. **图像评估**: 比较 `output_images` 和 `ground_truth_images`
   - PSNR, SSIM, FID 等指标
   - 使用预训练的视觉模型计算相似度

3. **步骤准确性**: 比较 `num_generated_images` 和真实步数

## 注意事项

1. 确保所有输入图片路径都是绝对路径或相对于运行目录的正确路径
2. 确保有足够的磁盘空间存储生成的图片
3. 对于大规模数据集，建议分批处理（使用 `--start_idx` 和 `--end_idx`）
4. 如果遇到内存不足，可以减少 GPU 数量或调整 `max_mem_per_gpu` 参数

## 示例

处理前100个样本：

```bash
python batch_inference.py \
    --input_jsonl data/test.jsonl \
    --output_dir ./output \
    --checkpoint_dir /path/to/checkpoint \
    --start_idx 0 \
    --end_idx 100
```

然后评估结果：

```bash
python evaluate_results.py \
    --results_jsonl ./output/batch_inference_xxx/results.jsonl
```

这将输出详细的统计信息并生成评估数据文件。
