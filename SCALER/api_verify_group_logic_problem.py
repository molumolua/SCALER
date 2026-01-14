# -*- coding: utf-8 -*-
import os
import math
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


from api import batch_get_chat_api
from prompt import answer_problem_prompt,train_prompt
from logger import setup_logger
from process_dataset import load_and_prepare_dataset,prepare_examples,save_output_parquet
from extract import extract_last_code_block,split_with_input_section,safe_format_template
from after_extract import verify_and_extract_test_case
import copy
from tqdm import tqdm


from math_verify.errors import TimeoutException
from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig


def pre_fun(example):
    prompt = answer_problem_prompt.format(problem=example['problem'])
    # print(prompt)
    return prompt


def post_fun(example, reply):
    example["answer"] = reply

def verify_correctness(answer_str,ground_truth_str,logger):
    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )
    try:
        acc,_=verify_func([ground_truth_str], [answer_str])
        if acc == 1:
            return True
        return False
    except Exception as e:
        logger.error(f"Error in verify {e}.")
        return False
    except TimeoutException:
        logger.error("TimeoutException in math-verify.")
        return False
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
    
    # 推理与并行
    parser.add_argument("--model", type=str, default="gpt-5", help="Model name for batch_get_chat_api")
    parser.add_argument("--n_processes", type=int, default=16, help="Parallel processes for API calls")
    parser.add_argument("--temperature", type=float, default=1, help="Sampling temperature")
    parser.add_argument("--timeout", type=int, default=20, help="Per-request timeout (seconds)")
    parser.add_argument("--think", action="store_true", default=False, help="Enable think mode for API (if supported)")
    parser.add_argument("--check_number", type=int, default=3, help="The number of submissions for check output.")
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
    # print(examples[0])
    left_problems = examples[:]       # list
    next_attempt_problems: List[Dict[str, Any]] = []
    
    for attempt in range(1, args.max_attempts + 1):
        total_problems = len(left_problems)
        if total_problems == 0:
            logger.info("No remaining problems. Stopping.")
            break

        total_batches = math.ceil(total_problems / args.batch_size)
        logger.info(f"Attempt {attempt}/{args.max_attempts} | remaining={total_problems} | batches={total_batches}")

        for b in range(total_batches):
            b_start = b * args.batch_size
            b_end = min((b + 1) * args.batch_size, total_problems)
            batch_problems = left_problems[b_start:b_end]

            logger.info(f"  Batch {b+1}/{total_batches} | size={len(batch_problems)}")
                    
                    
                    
            verify_problems  = []
            for group_problems in batch_problems:
                group_problems_list = [value for value in group_problems.values() if value is not None]
                verify_number = max(math.ceil(args.check_number),1)

                verify_problems.extend(group_problems_list[:verify_number]) # 最简单的K个
                

            batch_get_chat_api(
                examples=verify_problems,
                eng=args.model,
                pre_fun=pre_fun,
                post_fun=post_fun,
                logger=logger,
                n_processes=args.n_processes,
                temperature=args.temperature,
                timeout=args.timeout,
                max_try=args.inner_max_try,
                think=args.think,
            )
            now_pos = 0
            for group_problems in batch_problems:
                group_problems_list = [value for value in group_problems.values() if value is not None]
                verify_number = max(math.ceil(args.check_number),1)
                corresponding_examples = verify_problems[now_pos:now_pos+verify_number]
                add_to_next_flag = False
                
                
                group_problems_list = [value for value in group_problems.values() if value is not None]
                for (problem,example) in zip(group_problems_list,corresponding_examples):
                    #防止因为API的问题被干扰。多给他几次机会，让他尽可能不要timeout
                    if example['answer']:
                        problem['verify_flag']=verify_correctness(example['answer'],example['reward_model.ground_truth'],logger)
                    else:
                        add_to_next_flag = True    
                now_pos+=verify_number
                
                if add_to_next_flag:
                    next_attempt_problems.append(group_problems)  
                    
        left_problems = next_attempt_problems
        next_attempt_problems = []
        logger.info(f"End of Attempt {attempt}:  remaining={len(left_problems)}")
        
        
    output_problems: List[Dict[str, Any]] = []
    for group_problems in examples:
        group_problems_list = [value for value in group_problems.values() if value is not None]
        verify_number = max(math.ceil(args.check_number),1)
        output_flag = True
        for problem in group_problems_list[:verify_number]:
            if not problem.get('verify_flag',False):
                output_flag = False
                break
        if output_flag:
            # print("ok!",group_problems_list)
            output_problems.extend(group_problems_list)
    
    
    save_output_parquet(output_problems, save_dir_path=save_dir_path,  logger=logger, save_name=args.save_name,meta_name=args.save_meta_name)

    logger.info(f"Done. total_completed={len(output_problems)} | total_input={len(examples)}")
    
    

if __name__ == "__main__":
    main()
