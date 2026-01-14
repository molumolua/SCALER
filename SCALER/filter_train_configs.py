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
from process_dataset import load_and_prepare_dataset, save_output_parquet, prepare_examples,save_output_json
from extract import extract_last_code_block, split_with_input_section, safe_format_template
from after_extract import verify_json,filter_easy_hack_environment
import copy
from prompt import scale_param_extractor_prompt

from datasets import load_from_disk
import re


def main():
    parser = argparse.ArgumentParser(description="Batch prompting on local Parquet (CodeContests-like)")
    # Load args
    parser.add_argument("--load_dir", type=str,
                        default="../Dataset")
    parser.add_argument("--load_name",type=str,default="output_problems.json")
    # Save args
    parser.add_argument("--save_dir", type=str, default="./save")
    parser.add_argument("--save_name",type=str,default="output_problems.json")
    parser.add_argument("--save_meta_name",type=str,default="output_problems_meta.json")
    
    args = parser.parse_args()

    logger = setup_logger()
    logger.info(f"Args: {vars(args)}")
    
    load_dir_path = Path(args.load_dir)
    load_dir_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Input dir: {load_dir_path}")
    
    save_dir_path = Path(args.save_dir)
    save_dir_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output dir: {save_dir_path}")

    load_json = load_dir_path / args.load_name 

    with open(load_json, 'r') as json_file:
        examples = json.load(json_file)
    output_json = {}
    for problem_name,train_config in examples.items():
        text = train_config['raw_description']
        instruction_test_flag = False
        match1 = re.search(r'Output\n(.*?)\nExamples', text, re.DOTALL)
        match2 = re.search(r'Output\n(.*?)\nExample', text, re.DOTALL)
        # 如果匹配成功，打印提取的内容
        if match1:
            instruction_test_flag = True
            output_content = match1.group(1).strip()  # 去除多余的空白字符
            print("提取的内容：")
            print(output_content)
            train_config['instruction'] = output_content
        elif match2:
            instruction_test_flag = True
            output_content = match2.group(1).strip()  # 去除多余的空白字符
            print("提取的内容：")
            print(output_content)
            train_config['instruction'] = output_content
        else:
            print("没有找到匹配的内容")
            print(train_config['raw_description'])
            # break
        mod_pattern = re.compile(r'\b(mod|modulo)\b')
        if (not mod_pattern.search(text)) and instruction_test_flag:
            output_json[problem_name] = train_config
    save_output_json(output_json,save_dir_path=save_dir_path,logger=logger,save_name=args.save_name,meta_name=args.save_meta_name)
if __name__ == "__main__":
    main()
