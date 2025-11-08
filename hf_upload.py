import os
import sys
from huggingface_hub import HfApi, create_repo, upload_file, login
from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub.hf_api import whoami
from tqdm import tqdm
from pathlib import Path

def ensure_login():
    """检查是否已登录 Hugging Face"""
    try:
        info = whoami()
        username = info.get("name", "unknown")
        print(f"✅ 已登录 Hugging Face 用户: {username}")
        return True
    except Exception:
        print("⚠️ 检测到未登录 Hugging Face，请输入 token 登录：")
        print("（可在 https://huggingface.co/settings/tokens 获取）")
        token = input("🔑 请输入你的 Hugging Face token: ").strip()
        if not token:
            print("❌ 未输入 token，退出。")
            sys.exit(1)
        login(token=token, add_to_git_credential=True)
        print("✅ 登录成功！")
        return True

def ensure_repo(repo_id: str, token: str = None):
    """如果 repo 不存在，则自动创建"""
    api = HfApi()
    try:
        api.repo_info(repo_id, token=token)
        print(f"✅ Repo 存在：{repo_id}")
    except RepositoryNotFoundError:
        print(f"📦 未找到 {repo_id}，正在创建...")
        api.create_repo(repo_id=repo_id, private=True, exist_ok=True)
        print(f"✅ 已创建 {repo_id}")

def chunked_upload(file_path, repo_id, token=None, path_in_repo=None):
    """
    分片上传 + 断点续传（huggingface_hub 自动处理）
    """
    api = HfApi()
    file_path = Path(file_path)
    path_in_repo = path_in_repo or file_path.name

    print(f"🚀 上传文件: {file_path} → {repo_id}/{path_in_repo}")
    upload_file(
        path_or_fileobj=str(file_path),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        token=token,
        repo_type="model",  # 可改为 'dataset' 或 'space'
    )
    print(f"✅ 上传完成: {path_in_repo}")

def main():
    if len(sys.argv) < 3:
        print("用法: python hf_upload.py <repo_id> <file_or_dir_path>")
        print("例如: python hf_upload.py ybq/test-repo ./checkpoint")
        sys.exit(1)

    repo_id = sys.argv[1]
    file_or_dir = Path(sys.argv[2])

    ensure_login()
    token = os.environ.get("HF_TOKEN", None)
    ensure_repo(repo_id, token)

    if file_or_dir.is_file():
        chunked_upload(file_or_dir, repo_id, token)
    else:
        for f in tqdm(list(file_or_dir.rglob("*")), desc="📂 上传目录"):
            if f.is_file():
                rel_path = f.relative_to(file_or_dir)
                chunked_upload(f, repo_id, token, path_in_repo=str(rel_path))

if __name__ == "__main__":
    main()

# python hf_upload.py yinbq/text_4v_random_1800 /lustre/fsw/portfolios/nvr/users/ymingli/projects/ybq/results/checkpoints_random_views1357_textfirst/0001800/model.safetensors
# python hf_upload.py yinbq/sem_img_mv /lustre/fsw/portfolios/nvr/users/ymingli/projects/ybq/results/checkpoints_img_sem1_1107/0000450/model.safetensors
