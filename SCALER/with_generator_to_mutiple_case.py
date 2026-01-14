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
from process_dataset import load_and_prepare_dataset, save_output_jsonl, prepare_examples
from extract import extract_last_code_block, split_with_input_section, safe_format_template
from after_extract import verify_and_exec_generator
import copy


def main():
    parser = argparse.ArgumentParser(description="Batch prompting on local Parquet (CodeContests-like)")
    # 数据与加载
    parser.add_argument("--load_type", type=str, default="json", help="json or parquet")
    parser.add_argument("--load_dir", type=str, default="../Dataset", help="Directory containing local parquet shards")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test", "valid", "validation"],
                        help="Which split to load (matched by filename prefix e.g., train-*.parquet)")
    parser.add_argument("--file_glob", type=str, default=None, help="Optional custom glob, e.g. 'train-*.parquet'; if set, overrides --split matching")
    parser.add_argument("--drop_list", type=list, default=[], help="Drop heavy columns if needed (e.g., 'private_test_cases')")
    parser.add_argument("--start_problem_idx", type=int, default=0, help="Start index in the merged dataset")
    parser.add_argument("--max_rows", type=int, default=None, help="Limit number of rows to load after start_problem_idx (None = all)")
    parser.add_argument("--save_dir", type=str, default="./save")
    parser.add_argument("--save_name", type=str, default="output_problems.jsonl")
    parser.add_argument("--save_meta_name", type=str, default="output_problems_meta.json")
    
    # 推理与并行
    parser.add_argument("--extract_code", action="store_true", default=False, help="Whether to extract code from dataset")
    
    parser.add_argument("--filter_numerical", action="store_true", default=False, help="Whether only need numerical problems")
    parser.add_argument("--check_number", type=int, default=3, help="The number of submissions for check output.")
    parser.add_argument("--max_number", type=int, default=1000, help="The maximum number allowed in the generator.")
    parser.add_argument("--max_len", type=int, default=500, help="The maximum len allowed in the test case.")
    parser.add_argument("--num_of_test_case", type=int, default=30, help="The number of test cases to be generated.")
    parser.add_argument("--max_try_of_test_case", type=int, default=1000, help="The maximum trials to generate enough test cases.")
    parser.add_argument("--sandbox_url",type=str,default=None,help="The sandboxfusion url for code execution.")
    parser.add_argument("--error_cnt_limit",type=int,default=100,help="The error count limit to stop trying.")
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
        drop_list=args.drop_list,
        logger=logger
    )
    examples = prepare_examples(
        ds=dataset,
        start_idx=args.start_problem_idx,
        max_rows=args.max_rows,
        logger=logger,
        extract_code=args.extract_code)
    
    if not examples:
        logger.info("No examples with usable code. Exit.")
        return




    success_problems, todo_problems = verify_and_exec_generator(examples, 
                                                                logger,
                                                                debug=True,
                                                                test_case_num=args.num_of_test_case,
                                                                test_case_max_len=args.max_len,
                                                                test_case_max_number=args.max_number,
                                                                max_try_num=args.max_try_of_test_case,
                                                                check_number=args.check_number,
                                                                filter_numerical=args.filter_numerical,
                                                                sandboxfusion_url=args.sandbox_url,
                                                                error_cnt_limit=args.error_cnt_limit)

            
    save_output_jsonl(success_problems, save_dir_path=save_dir_path,  logger=logger, save_name=args.save_name, meta_name=args.save_meta_name)
            
    logger.info(f"Done. total_completed={len(success_problems)} | total_input={len(examples)}")

if __name__ == "__main__":
    main()
