# -*- coding: utf-8 -*-
import os
from pathlib import Path
from huggingface_hub import hf_hub_download
from datasets import load_dataset

# ===== 你想保存到的目录（修改这里）=====
BASE_DIR = Path("./").resolve()
DEST_DIR = BASE_DIR / "Code-Contests-Plus" / "default_single"   # 自定义保存位置
DEST_DIR.mkdir(parents=True, exist_ok=True)

# ===== 可选：镜像与大文件加速（与之前一致）=====
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")  # 大文件更快

REPO_ID = "ByteDance-Seed/Code-Contests-Plus"
SHARD_NAME = "part-00000-of-00010.parquet"   # 只要这一片

print(f"[Info] Downloading {REPO_ID}/={SHARD_NAME} ...")

# 方式一：直接下载目标单文件到指定目录（推荐）
local_file = hf_hub_download(
    repo_id=REPO_ID,
    repo_type="dataset",
    filename=f"{SHARD_NAME}",   # 子集目录 + 具体文件名
    local_dir=str(DEST_DIR),                  # 目标保存目录
    local_dir_use_symlinks=False,             # 直接落盘
)
print(f"[Done] Saved to: {local_file}")

# ==== 可选：验证能否读取 ====
ds = load_dataset("parquet", data_files=local_file, split="train")
print(ds)
print(ds.select(range(5)))
