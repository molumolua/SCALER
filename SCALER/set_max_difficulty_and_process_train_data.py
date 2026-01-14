# -*- coding: utf-8 -*-
import os
import math
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from api import batch_get_chat_api
    from logger import setup_logger
    from process_dataset import load_and_prepare_dataset, save_output_parquet, prepare_examples,save_output_json
    from extract import extract_last_code_block, split_with_input_section, safe_format_template
    from after_extract import find_max_difficulty
except:
    from SCALER.api import batch_get_chat_api
    from SCALER.logger import setup_logger
    from SCALER.process_dataset import load_and_prepare_dataset, save_output_parquet, prepare_examples,save_output_json
    from SCALER.extract import extract_last_code_block, split_with_input_section, safe_format_template
    from SCALER.after_extract import find_max_difficulty


import numpy as np

def generate_difficulty_dict(vmin, vmax):
    # 计算区间大小
    range_size = vmax - vmin + 1
    # 确定 difficulty 数量，但最多 100 个
    num_difficulties = min(30, range_size)
    difficulty_dict = {}

    # 对于较小范围，使用等差数列（确保没有两个相邻的 difficulty 是相同的）
    if range_size <= 100:
        step = range_size / num_difficulties
        for difficulty in range(num_difficulties):
            v = int(vmin + step * difficulty)
            while difficulty > 0 and difficulty_dict[difficulty - 1] == v:
                v += 1 
            difficulty_dict[difficulty] = v

    elif range_size <= 1000:
        base = 1.3 
        for difficulty in range(num_difficulties):
            if difficulty > 0:
                v = round(vmin + (base ** difficulty))  # 采用指数增长
            else:
                v = vmin
            if v>=vmax:
                difficulty_dict[difficulty] = vmax
                break
            difficulty_dict[difficulty] = v
    else:
        base = 1.6  # 这里我们选择基数为 10 来实现更快的增长
        for difficulty in range(num_difficulties):
            if difficulty > 0:
                v = round(vmin + (base ** difficulty))  # 采用指数增长
            else:
                v = vmin
            if v>=vmax:
                difficulty_dict[difficulty] = vmax
                break
            difficulty_dict[difficulty] = v
    return difficulty_dict



def main():
    parser = argparse.ArgumentParser(description="Batch prompting on local Parquet (CodeContests-like)")
    # Load args
    parser.add_argument("--load_type",type=str,default="json",help="json or parquet")
    parser.add_argument("--load_dir", type=str,
                        default="../Dataset",
                        help="Directory containing local parquet shards")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test", "valid", "validation"],
                        help="Which split to load (matched by filename prefix e.g., train-*.parquet)")
    parser.add_argument("--file_glob", type=str, default=None,
                        help="Optional custom glob, e.g. 'train-*.parquet'; if set, overrides --split matching")
    parser.add_argument("--start_problem_idx", type=int, default=0,
                        help="Start index in the merged dataset")
    parser.add_argument("--max_rows", type=int, default=None,
                        help="Limit number of rows to load after start_problem_idx (None = all)")
    # Save args
    parser.add_argument("--save_dir", type=str, default="./save")
    parser.add_argument("--save_name",type=str,default="output_problems.jsonl")
    parser.add_argument("--save_meta_name",type=str,default="output_problems_meta.json")

    # 推理与并行
    parser.add_argument("--model", type=str, default="gpt-5", help="Model name for batch_get_chat_api")
    parser.add_argument("--n_processes", type=int, default=16, help="Parallel processes for API calls")
    parser.add_argument("--temperature", type=float, default=1, help="Sampling temperature")
    parser.add_argument("--timeout", type=int, default=20, help="Per-request timeout (seconds)")
    parser.add_argument("--think", action="store_true", default=False, help="Enable think mode for API (if supported)")
    
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument("--sandbox_url",type=str,default=None,help="The sandboxfusion url for code execution.")
    # 批次与重试
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size per attempt")
    parser.add_argument("--max_attempts", type=int, default=3, help="Outer retry attempts over remaining problems")
    parser.add_argument("--inner_max_try", type=int, default=3, help="Inner retry count passed to batch_get_chat_api")
    
    args = parser.parse_args()

    logger = setup_logger()
    logger.info(f"Args: {vars(args)}")

    save_dir_path = Path(args.save_dir)
    save_dir_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output dir: {save_dir_path}")

    # 读取 parquet + 组装为带 "code" 的 examples
    dataset = load_and_prepare_dataset(
        load_type=args.load_type,
        load_dir=Path(args.load_dir),
        split=args.split,
        file_glob=args.file_glob,
        drop_list=[],
        logger=logger
    )
    
    examples = prepare_examples(
        ds=dataset,
        start_idx=args.start_problem_idx,
        max_rows=args.max_rows,
        logger=logger,
        extract_code=False)
    
    if not examples:
        logger.info("No examples with usable code. Exit.")
        return
    examples=find_max_difficulty(examples,logger,debug=True,sandboxfusion_url=args.sandbox_url,max_prompt_length=args.max_prompt_length)
    examples_processed = []
    for example in examples:
        examples_processed.append({
            **example,
            "difficulty_dict":generate_difficulty_dict(0,example['scale_range'])
        })
    
    json_train_configs={}
    for example in examples_processed:
        example['parsed_json'] = example['parsed_json'].replace("'", '"')
        example['params']=json.loads(example['parsed_json'])
        example['params']['difficulty']={
            "version": 1,
            "params": {
                "dmax": len(example['difficulty_dict'])-1,
                "ema_beta":0.0,
                "activate_function":"base",
                "history_len": 20,
                "slope_scale": 0.05,
                "age_scale": 100,
                "alpha": 1.0,
                "beta": 0.1
            }
        }
        example.pop("parsed_json")
        example.pop("answer")
        json_train_configs[example['name']]=example
        
    save_output_json(json_train_configs, save_dir_path=save_dir_path,  logger=logger, save_name=args.save_name, meta_name=args.save_meta_name)
if __name__ == "__main__":
    main()
