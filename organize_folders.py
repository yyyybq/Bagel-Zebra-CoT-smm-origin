import os
import re
import shutil
from pathlib import Path
from collections import defaultdict

def extract_category_and_number(folder_name):
    """
    从文件夹名称中提取类别和序号
    例如: 'castle10' -> ('castle', 10)
          'dog' -> ('dog', 0)
          'plane2' -> ('plane', 2)
    """
    # 匹配以字母开头，后面可能跟数字的模式
    match = re.match(r'^([a-zA-Z_]+)(\d*)$', folder_name)
    if match:
        category = match.group(1)
        number = int(match.group(2)) if match.group(2) else 0
        return category, number
    return None, None

def organize_folders(source_dir, target_base_dir):
    """
    整理文件夹：按类别分组，重命名为001, 002, 003...
    """
    source_path = Path(source_dir)
    target_path = Path(target_base_dir)
    
    # 确保目标基础目录存在
    target_path.mkdir(parents=True, exist_ok=True)
    
    # 收集所有文件夹及其类别
    category_folders = defaultdict(list)
    
    print("扫描文件夹...")
    for item in source_path.iterdir():
        if item.is_dir():
            category, number = extract_category_and_number(item.name)
            if category:
                category_folders[category].append((item, number))
                print(f"  发现: {item.name} -> 类别: {category}, 序号: {number}")
    
    # 为每个类别创建文件夹并移动文件
    print(f"\n找到 {len(category_folders)} 个类别")
    
    for category, folders in sorted(category_folders.items()):
        print(f"\n处理类别: {category} ({len(folders)} 个文件夹)")
        
        # 创建类别文件夹
        category_dir = target_path / category
        category_dir.mkdir(parents=True, exist_ok=True)
        
        # 按原始序号排序
        folders.sort(key=lambda x: x[1])
        
        # 重命名并移动
        for idx, (folder_path, original_num) in enumerate(folders, start=1):
            new_name = f"{idx:03d}"  # 格式化为001, 002, 003...
            target_folder = category_dir / new_name
            
            try:
                # 移动文件夹
                shutil.move(str(folder_path), str(target_folder))
                print(f"  ✓ {folder_path.name} -> {category}/{new_name}")
            except Exception as e:
                print(f"  ✗ 错误: {folder_path.name} - {e}")
    
    print("\n整理完成!")
    
    # 打印统计信息
    print("\n统计信息:")
    for category in sorted(category_folders.keys()):
        count = len(category_folders[category])
        print(f"  {category}: {count} 个文件夹")

if __name__ == "__main__":
    source_directory = r"C:\Users\user\Desktop\code\data\semantic_blocks_part2"
    target_directory = r"C:\Users\user\Desktop\code\data\semantic_blocks\normal"
    
    print("=" * 60)
    print("文件夹整理工具")
    print("=" * 60)
    print(f"源目录: {source_directory}")
    print(f"目标目录: {target_directory}")
    print("=" * 60)
    
    organize_folders(source_directory, target_directory)
