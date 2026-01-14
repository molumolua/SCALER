# -*- coding: utf-8 -*-
import os
import math
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from api import batch_get_chat_api
from prompt import generate_generator_prompt
from logger import setup_logger
from process_dataset import load_and_prepare_dataset, save_output_parquet, prepare_examples,save_output_jsonl
from extract import extract_last_code_block, split_with_input_section, safe_format_template
from after_extract import verify_json,filter_easy_hack_environment
import copy
from prompt import scale_param_extractor_prompt

from datasets import load_from_disk



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
    
    parser.add_argument("--test_times",type=int,default=100)
    parser.add_argument("--different_output_limit", type=int, default=10)
    parser.add_argument("--max_output_rate", type=float, default=0.4)
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
    
    examples_processed,_ = filter_easy_hack_environment(examples,logger,debug=True,
                                                      test_times=args.test_times,
                                                      sandboxfusion_url=args.sandbox_url,
                                                      different_output_limit=args.different_output_limit,
                                                      max_output_rate=args.max_output_rate)
    
    save_output_jsonl(examples_processed, save_dir_path=save_dir_path,  logger=logger, save_name=args.save_name, meta_name=args.save_meta_name)
    

if __name__ == "__main__":
    main()
