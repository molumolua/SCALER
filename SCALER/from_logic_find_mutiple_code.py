# -*- coding: utf-8 -*-
import os
import math
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd



from prompt import train_prompt
from logger import setup_logger
from process_dataset import load_and_prepare_dataset,prepare_examples,save_output_parquet,save_output_jsonl
from process_token_len import filter_examples_by_token_budget
from extract import extract_last_code_block,split_with_input_section,safe_format_template
from after_extract import verify_and_extract_test_case,is_valid_number
import copy
from tqdm import tqdm




# ---------- main ----------

def main():
    parser = argparse.ArgumentParser(description="Batch prompting on local Parquet (CodeContests-like)")
    # 数据与加载
    parser.add_argument("--load_type",type=str,default="json",help="json or parquet")
    parser.add_argument("--load_dir", type=str,
                        default="../Dataset",
                        help="Directory containing local parquet shards")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test", "valid", "validation"],
                        help="Which split to load (matched by filename prefix e.g., train-*.parquet)")
    parser.add_argument("--file_glob", type=str, default=None,
                        help="Optional custom glob, e.g. 'train-*.parquet'; if set, overrides --split matching")
    parser.add_argument("--drop_list", type=list, default=[],
                        help="Drop heavy columns if needed (e.g., 'private_test_cases')")

    parser.add_argument("--code_load_type",type=str,default="json",help="json or parquet")
    parser.add_argument("--code_load_dir", type=str,
                        default="../Dataset",
                        help="Directory containing local parquet shards")
    parser.add_argument("--code_split", type=str, default="train", choices=["train", "test", "valid", "validation"],
                        help="Which split to load (matched by filename prefix e.g., train-*.parquet)")
    parser.add_argument("--code_file_glob", type=str, default=None,
                        help="Optional custom glob, e.g. 'train-*.parquet'; if set, overrides --split matching")
    parser.add_argument("--code_drop_list", type=list, default=[],
                        help="Drop heavy columns if needed (e.g., 'private_test_cases')")
    
    parser.add_argument("--ref_load_type",type=str,default="json",help="json or parquet")
    parser.add_argument("--ref_load_dir", type=str,
                        default="../Dataset",
                        help="Directory containing local parquet shards")
    parser.add_argument("--ref_split", type=str, default="train", choices=["train", "test", "valid", "validation"],
                        help="Which split to load (matched by filename prefix e.g., train-*.parquet)")
    parser.add_argument("--ref_file_glob", type=str, default=None,
                        help="Optional custom glob, e.g. 'train-*.parquet'; if set, overrides --split matching")
    parser.add_argument("--ref_drop_list", type=list, default=[],
                        help="Drop heavy columns if needed (e.g., 'private_test_cases')")
    
    parser.add_argument("--use_ref",action="store_true", default=False)
    

    parser.add_argument("--save_dir", type=str, default="./save")
    parser.add_argument("--save_name",type=str,default="output_problems.jsonl")
    parser.add_argument("--save_meta_name",type=str,default="output_problems_meta.json")


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
        logger=logger)
    
    
    # 读取 parquet + 组装为带 "code" 的 examples
    code_dataset = load_and_prepare_dataset(
        load_type=args.code_load_type,
        load_dir=Path(args.code_load_dir),
        split=args.code_split,
        file_glob=args.code_file_glob,
        drop_list=args.code_drop_list,
        logger=logger
    )
    code_examples = prepare_examples(
        ds=code_dataset,
        logger=logger)
    
    
    ref_dataset = load_and_prepare_dataset(
        load_type=args.ref_load_type,
        load_dir=Path(args.ref_load_dir),
        split=args.ref_split,
        file_glob=args.ref_file_glob,
        drop_list=args.ref_drop_list,
        logger=logger
    )
    ref_examples = prepare_examples(
        ds=ref_dataset,
        logger=logger)
    
    if not examples or not code_examples:
        logger.info("No examples with usable code. Exit.")
        return
    
    output_data_list = []
    raw_id_set = set()
    for idx,example in tqdm(enumerate(examples)):
        raw_id_set.add(example['raw_id'])
    
    for idx,code_example in tqdm(enumerate(code_examples)):
        if code_example['id'] in raw_id_set:
            only_number_flag = True
            for test_case in code_example['test_case_list']:
                if not is_valid_number(test_case['input']):
                    only_number_flag = False
                    
            if not only_number_flag:
                if args.use_ref:
                    for ref_example in ref_examples:
                        if code_example['id'] == ref_example['id']:
                            code_example['answer']=ref_example['answer']
                            
                output_data_list.append(code_example)
        
 
    
    save_output_jsonl(output_data_list,save_dir_path,logger,args.save_name,args.save_meta_name)

if __name__ == "__main__":
    main()
