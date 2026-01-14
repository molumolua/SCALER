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
from process_dataset import load_and_prepare_dataset,prepare_examples,save_output_parquet
# from process_token_len import filter_examples_by_token_budget
from extract import extract_last_code_block,split_with_input_section,safe_format_template
from after_extract import verify_and_extract_test_case
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
    parser.add_argument("--start_problem_idx", type=int, default=0,
                        help="Start index in the merged dataset")
    parser.add_argument("--max_rows", type=int, default=None,
                        help="Limit number of rows to load after start_problem_idx (None = all)")
    parser.add_argument("--save_dir", type=str, default="./save")
    parser.add_argument("--save_name",type=str,default="output_problems.jsonl")
    parser.add_argument("--save_meta_name",type=str,default="output_problems_meta.json")

    parser.add_argument("--extract_code", action="store_true", default=False, help="Whether to extract code from dataset")
    
    parser.add_argument("--max_tokens", type=int,default=1024, help="max token for train")
    parser.add_argument("--train_model_path", type=str, default="/inspire/hdd/global_public/public_models/Qwen/Qwen2.5-7B", help="Whether to extract code from dataset")
    parser.add_argument("--example_level",type=int,default="1",help="1 means array of item , 2 means array of array of item")
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
    output_train_data_list = []
    code_example_len = None
    if args.example_level == 1:
        raw_id_set = set()
        for idx,example in tqdm(enumerate(examples)):
            raw_id_set.add(example['raw_id'])
            output_train_data_list.append({
                'prompt': train_prompt(example['problem']),
                'reward_model':{
                    "ground_truth":example['reward_model.ground_truth']
                },
                "data_source": example['source'],
                "extra_info":{"index": idx,"raw_id":example['raw_id']},
            })
        code_example_len = len(raw_id_set)
    elif args.example_level == 2:
        code_example_len = len(examples)
        for example_list in examples:
            for idx,example in tqdm(enumerate(example_list.values())):
                if example:
                # print("hhh",example)
                    output_train_data_list.append({
                        'prompt': train_prompt(example['problem']),
                        'reward_model':{
                            "ground_truth":example['reward_model.ground_truth']
                        },
                        "data_source": example['source'],
                        "extra_info":{"index": idx,"raw_id":example['raw_id']},
                    })
            
    #output_train_data_list = filter_examples_by_token_budget(output_train_data_list,model_path=args.train_model_path,max_token=args.max_tokens,logger=logger)
    
    logger.info(f"Filter token len <= {args.max_tokens}, left output train data list {len(output_train_data_list)}.")
    logger.info(f"Total:{len(output_train_data_list)}, Get Example:{code_example_len}, Avg logic problem from example:{len(output_train_data_list)/code_example_len}")
    processed_df = pd.DataFrame(output_train_data_list)
    processed_df.to_parquet(Path(str(save_dir_path)+"/"+args.save_name))
    
        
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_meta = save_dir_path / (args.save_meta_name or "meta.json")


    # Write meta
    meta = {
        "timestamp": ts,
        "code_len":code_example_len,
        "total":len(output_train_data_list)
    }
    
    with out_meta.open("w", encoding="utf-8") as mf:
        json.dump(meta, mf, ensure_ascii=False, indent=2)
        
        
        
if __name__ == "__main__":
    main()
