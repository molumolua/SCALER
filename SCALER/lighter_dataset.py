#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import gc
from pathlib import Path
from process_dataset import load_and_prepare_dataset, save_output_jsonl  # 保留原有 save 函数以防需要
from logger import setup_logger

def extract_cpp_solution_from_item(item):
    """
    从 item 中尽可能稳健地提取第一个 cpp solution 到顶层键 'solution'，
    并移除原来的 'solutions' 键（若存在）。
    """
    try:
        sols = item.get('solutions')
        if isinstance(sols, dict):
            # 常见结构： {'solution': [...], ...}
            if 'solution' in sols and 'language' in sols:
                seq = sols['solution']
                lan = sols['language']
                assert len(seq) == len(lan)
                for (l,s) in zip(lan,seq):
                    if l == 2: #cpp
                        item['solution']=s
            else:
                # 尝试其他备选键
                for k in ('code', 'answer'):
                    if k in sols:
                        item['solution'] = sols[k]
                        break
                else:
                    # 找不到，置 None
                    item['solution'] = None
        else:
            # 如果 solutions 不存在或格式不同，保留顶层已有 solution 或置 None
            item.setdefault('solution', None)
    except Exception:
        item.setdefault('solution', None)
    # 移除原来的 solutions 字段，避免冗余
    if 'solutions' in item:
        try:
            item.pop('solutions', None)
        except Exception:
            pass
    return item

def process_parquet_stream(parquet_path: Path, load_dir: Path, save_dir: Path, logger):
    logger.info(f"Processing parquet: {parquet_path.name}")
    try:
        # load_and_prepare_dataset 只加载当前文件（通过 file_glob）
        dataset = load_and_prepare_dataset(
            file_glob=parquet_path.name,
            drop_list=["public_tests", "private_tests", "generated_tests", "incorrect_solutions", "input_file", "output_file"],
            load_dir=load_dir,
            load_type="parquet",
            logger=logger
        )

        # 输出文件（增量写入），与输入名相同但后缀为 .jsonl
        out_path = save_dir / parquet_path.with_suffix('.jsonl').name
        out_path.parent.mkdir(parents=True, exist_ok=True)

        written = 0
        # 以流式模式遍历 dataset（不要转换为 list）
        # 注意：datasets 库的 Dataset 支持迭代（yield dicts），或若返回 generator 亦可
        with out_path.open('w', encoding='utf-8') as fout:
            for item in dataset:
                # 提取 solution 字段并移除其它冗余字段
                item = extract_cpp_solution_from_item(item)
                if item.get('solution',None):
                    # 你可以在这里对 item 做额外筛选（例如只保留 solution 非空的项）
                    fout.write(json.dumps(item, ensure_ascii=False) + '\n')
                    written += 1
                    # 可选：每写若干条 flush 一次（降低内存峰值）
                    if written % 1000 == 0:
                        fout.flush()

        logger.info(f"Finished {parquet_path.name}, wrote {written} lines to {out_path}")
    except Exception as e:
        logger.exception(f"Error processing {parquet_path.name}: {e}")
    finally:
        # 释放内存
        try:
            del dataset
        except Exception:
            pass
        gc.collect()

if __name__ == "__main__":
    logger = setup_logger()

    load_dir = Path("/inspire/hdd/global_user/xucaijun-253108120121/Dataset/hf_datasets/code_contest")
    save_dir = Path("/inspire/hdd/global_user/xucaijun-253108120121/Dataset/hf_datasets/code_contest_light")

    save_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(load_dir.glob("*.parquet"))
    if not parquet_files:
        logger.warning(f"No parquet files found in {load_dir}")
    for pq in parquet_files:
        process_parquet_stream(pq, load_dir=load_dir, save_dir=save_dir, logger=logger)

    logger.info("All done.")
