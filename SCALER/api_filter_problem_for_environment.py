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
    from prompt import generate_generator_prompt
    from logger import setup_logger
    from process_dataset import load_and_prepare_dataset, save_output_parquet, prepare_examples,save_output_jsonl
    from extract import extract_last_code_block, split_with_input_section, safe_format_template
    from after_extract import verify_meta_json
    from prompt import scale_param_extractor_prompt,problem_meta_extractor_prompt
except:
    from SCALER.api import batch_get_chat_api
    from SCALER.prompt import generate_generator_prompt
    from SCALER.logger import setup_logger
    from SCALER.process_dataset import load_and_prepare_dataset, save_output_parquet, prepare_examples,save_output_jsonl
    from SCALER.extract import extract_last_code_block, split_with_input_section, safe_format_template
    from SCALER.after_extract import verify_meta_json
    from SCALER.prompt import scale_param_extractor_prompt,problem_meta_extractor_prompt

from datasets import load_from_disk
def cut_desc_at_input(desc: str) -> str:
    lines = desc.splitlines()
    kept = []
    for line in lines:
        # 一旦出现“单独的一行 Input”，就截断
        if line.strip() == "Input":
            break
        kept.append(line)
    # 去掉首尾空行，防止多余 \n
    return "\n".join(kept).strip()

def preprocess_example(example):
    description = cut_desc_at_input(example["description"])
    return description

def pre_fun(example):
    return problem_meta_extractor_prompt.format(problem=example['raw_description'])

def post_fun(example, reply):
    example["answer"] = reply
def pre_filter(examples,filter_languages = [2,3],filter_languages_cnt = 3,keep_len = 10):
    ds_processed = []
    print(examples[0]["description"])
    # 批量处理，可以设置 num_proc 并行
    for example in examples:
        description = preprocess_example(example)
        if example['description'] == description: #只要有Input section的
            continue
        
        solutions = {
            "solution":[],
            "language":[]
        }
        for code,lang in zip(example['solutions']['solution'],example['solutions']['language']):
            if lang in filter_languages:
                solutions['solution'].append(code)
                solutions['language'].append(lang)
            if len(solutions['solution']) >= keep_len:
                break
    

        if len(solutions["solution"]) < filter_languages_cnt :
            continue
        
        ds_processed.append({
            "name":example['name'],
            "logic_description":description,
            "raw_description":example['description'],
            "solutions":solutions
            # "solution":sol,
            # "language":lang
        })


    # 看一下效果
    print("------ after ------")
    print(ds_processed[0])
    return ds_processed

def scale_param_extract(examples,logger,save_dir_path,args):
    output_problems = []
    left_problems = examples
    next_attempt_problems = []
    for attempt in range(1, args.max_attempts + 1):
        total_problems = len(left_problems)
        if total_problems == 0:
            logger.info("No remaining problems. Stopping.")
            break

        total_batches = math.ceil(total_problems / args.batch_size)
        logger.info(f"Attempt {attempt}/{args.max_attempts} | remaining={total_problems} | batches={total_batches}")
        # 数据集可能太大了，例如 10w   --- > 16 
        # 所以，我就先把 10w 个 --- > N组 256 
        # 256 里面有一些成功的 ok，失败的，加入下一轮 attempt
        # 第一轮结束，先存这256   第二轮结束，就存512
        for b in range(total_batches):
            b_start = b * args.batch_size
            b_end = min((b + 1) * args.batch_size, total_problems)
            batch_problems = left_problems[b_start:b_end]

            logger.info(f"  Batch {b+1}/{total_batches} | size={len(batch_problems)}")

            batch_get_chat_api(
                examples=batch_problems,
                eng=args.model,
                pre_fun=pre_fun,  # Pass max_number to pre_fun
                post_fun=post_fun,
                logger=logger,
                n_processes=args.n_processes,
                temperature=args.temperature,
                timeout=args.timeout,
                max_try=args.inner_max_try,
                think=args.think,
            )
            # success_problems=batch_problems
            # todo_problems=[]
            success_problems, todo_problems = verify_meta_json(batch_problems, logger=logger,debug=True)

            output_problems.extend(success_problems)

            next_attempt_problems.extend(todo_problems)
            
            save_output_jsonl(output_problems, save_dir_path=save_dir_path,  logger=logger, save_name=args.save_name, meta_name=args.save_meta_name)
            logger.info(f"success={len(output_problems)} | retry_next={len(todo_problems)}")

        left_problems = next_attempt_problems
        next_attempt_problems = []
        logger.info(f"End of Attempt {attempt}: accumulated={len(output_problems)} | remaining={len(left_problems)}")

    logger.info(f"Done. total_completed={len(output_problems)} | total_input={len(examples)}")

# def filter_only_one_scale_problem(examples):
#     output_examples = []
#     for example in examples:
#         obj = json.loads(example['parsed_json'])
#         if len(obj) == 1:
#             output_examples.append(example)
    
#     return output_examples


def filter_output_problems(examples,allowed_output_types):
    output_examples = []
    for example in examples:
        obj = json.loads(example['parsed_json'])
        if obj['output_type'] in allowed_output_types and obj['is_output_unique'] and len(obj['scale_params'])>0:
            output_examples.append(example)
            
            
    return output_examples
import re
def append_instruction(examples):
    output_examples = []
    for example in examples:
        text = example['raw_description']
        match1 = re.search(r'Output\n(.*?)\nExamples', text, re.DOTALL)
        match2 = re.search(r'Output\n(.*?)\nExample', text, re.DOTALL)
        output_content=""
        if match1:
            output_content = match1.group(1).strip()  # 去除多余的空白字符
        elif match2:
            output_content = match2.group(1).strip()  # 去除多余的空白字符
        
        output_examples.append({
            **example,
            "instruction":output_content
        })

    return output_examples

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
    
    parser.add_argument("--filter_numerical", action="store_true", default=False, help="Whether only need numerical problems")
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
    
    examples_processed = pre_filter(examples)
    examples_processed = scale_param_extract(examples_processed,logger,save_dir_path,args)
    
    

if __name__ == "__main__":
    main()
