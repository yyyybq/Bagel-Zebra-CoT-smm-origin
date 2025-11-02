# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from .interleave_datasets import UnifiedEditIterableDataset
from .t2i_dataset import T2IIterableDataset
from .vlm_dataset import SftJSONLIterableDataset
from .interleave_datasets.think_trace_dataset import ThinkTraceJSONLIterableDataset


DATASET_REGISTRY = {
    't2i_pretrain': T2IIterableDataset,
    'vlm_sft': SftJSONLIterableDataset,
    'unified_edit': UnifiedEditIterableDataset,
    'think_trace': ThinkTraceJSONLIterableDataset,
    'block_dataset': ThinkTraceJSONLIterableDataset,
    'block_dataset_random': ThinkTraceJSONLIterableDataset,
}


DATASET_INFO = {
    'think_trace': {
        'think_trace_dataset': {
            'data_dir': '/scratch/by2593/project/SpaCU/interleaved-co3dv2/data',
            'jsonl_path': '/scratch/by2593/project/SpaCU/interleaved-co3dv2/data/merged_train.jsonl',
            'image_prefix_dir': '/scratch/by2593/project/SpaCU/restored_data2',  # Base path for relative image paths
            # 'num_total_samples': 100,
        },
    },
    'block_dataset': {
        'block_dataset': {
            'data_dir': "/scratch/by2593/project/SMM/semantic_blocks_part1",
            # 'jsonl_path': '/scratch/by2593/project/SMM/SMM_data/semantic_block_train_part1_v2_reordered.jsonl',
            'jsonl_path': '/scratch/by2593/project/SMM/SMM_data/semantic_block_train_part1.jsonl',
            'image_prefix_dir': '/scratch/by2593/project/SMM/semantic_blocks_part1',  # Base path for relative image paths
            # 'num_total_samples': 100,
        },
    },
        'block_dataset_random': {
        'block_dataset_random': {
            'data_dir': "",
            'jsonl_path': './ranGenTraining_views1357_imgfirst.jsonl',
            'image_prefix_dir': '',  # Base path for relative image paths
            # 'num_total_samples': 100,
        },
    },
}