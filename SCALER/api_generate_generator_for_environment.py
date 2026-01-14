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
    from prompt import testcase_generator_prompt
    from logger import setup_logger
    from process_dataset import load_and_prepare_dataset, save_output_jsonl, prepare_examples
    from extract import extract_last_code_block, split_with_input_section, safe_format_template
    from after_extract import verify_and_exec_generator_for_environment_combined,new_random_get_json_object
    from api_filter_problem_for_environment import filter_output_problems,append_instruction
except:
    from SCALER.api import batch_get_chat_api
    from SCALER.prompt import testcase_generator_prompt
    from SCALER.logger import setup_logger
    from SCALER.process_dataset import load_and_prepare_dataset, save_output_jsonl, prepare_examples
    from SCALER.extract import extract_last_code_block, split_with_input_section, safe_format_template
    from SCALER.after_extract import verify_and_exec_generator_for_environment_combined,new_random_get_json_object
    from SCALER.api_filter_problem_for_environment import filter_output_problems,append_instruction
    

# ---------- main ----------

def pre_fun(example):
    example_json_obj = new_random_get_json_object(example['parsed_json'])

        
    return testcase_generator_prompt.format(problem=example['raw_description'],example_json_obj=example_json_obj)

def post_fun(example, reply):
    example["answer"] = reply

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
    parser.add_argument("--model", type=str, default="gpt-5", help="Model name for batch_get_chat_api")
    parser.add_argument("--n_processes", type=int, default=16, help="Parallel processes for API calls")
    parser.add_argument("--temperature", type=float, default=1, help="Sampling temperature")
    parser.add_argument("--timeout", type=int, default=20, help="Per-request timeout (seconds)")
    parser.add_argument("--think", action="store_true", default=False, help="Enable think mode for API (if supported)")
    parser.add_argument("--extract_code", action="store_true", default=False, help="Whether to extract code from dataset")
    
    # 过滤与生成设置
    parser.add_argument("--check_number", type=int, default=3, help="The number of submissions for check output.")
    parser.add_argument("--sandbox_url", type=str, default=None, help="The sandboxfusion url for code execution.")
    
    # 环境验证相关参数（verify_and_exec_generator_for_environment_combined 会用到）
    parser.add_argument("--deep_test_times", type=int, default=50,
                        help="Number of deep tests (same seed, multiple runs) for each generator.")
    parser.add_argument("--breadth_test_times", type=int, default=10,
                        help="Number of breadth tests (different seeds / inputs) for each generator.")
    parser.add_argument("--different_output_limit", type=int, default=50,
                        help="Maximum allowed different outputs before treating generator as unstable.")
    parser.add_argument("--max_output_rate", type=float, default=0.6,
                        help="Maximum ratio of distinct outputs / total runs; larger means generator is too unstable.")
    
    # 批次与重试
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size per attempt")
    parser.add_argument("--max_attempts", type=int, default=3, help="Outer retry attempts over remaining problems")
    parser.add_argument("--inner_max_try", type=int, default=3, help="Inner retry count passed to batch_get_chat_api")
    
    # 输出类型过滤
    parser.add_argument(
        "--allowed_output_types",
        type=str,
        nargs="+",
        default=["array", "number","string"],
        help="Allowed output types for problems, e.g. array number string"
    )
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
        start_idx=0,
        max_rows=args.max_rows,
        logger=logger,
        extract_code=args.extract_code)
    
    if not examples:
        logger.info("No examples with usable code. Exit.")
        return

    examples = filter_output_problems(examples,allowed_output_types=args.allowed_output_types)
    examples = append_instruction(examples)
    
    examples = [example for example in examples if example.get('instruction')]
    
    examples = [example for example in examples if "<image"  not in example['raw_description'].lower()]
    
    output_problems: List[Dict[str, Any]] = []
    left_problems = examples[args.start_problem_idx:]       # list
    next_attempt_problems: List[Dict[str, Any]] = []
    
    
    print(len(examples))

    # for attempt in range(1, args.max_attempts + 1):
    #     total_problems = len(left_problems)
    #     if total_problems == 0:
    #         logger.info("No remaining problems. Stopping.")
    #         break

    #     total_batches = math.ceil(total_problems / args.batch_size)
    #     logger.info(f"Attempt {attempt}/{args.max_attempts} | remaining={total_problems} | batches={total_batches}")
    #     # 数据集可能太大了，例如 10w   --- > 16 
    #     # 所以，我就先把 10w 个 --- > N组 256 
    #     # 256 里面有一些成功的 ok，失败的，加入下一轮 attempt
    #     # 第一轮结束，先存这256   第二轮结束，就存512
    #     for b in range(total_batches):
    #         b_start = b * args.batch_size
    #         b_end = min((b + 1) * args.batch_size, total_problems)
    #         batch_problems = left_problems[b_start:b_end]

    #         logger.info(f"  Batch {b+1}/{total_batches} | size={len(batch_problems)}")

    #         batch_get_chat_api(
    #             examples=batch_problems,
    #             eng=args.model,
    #             pre_fun=pre_fun,  # Pass max_number to pre_fun
    #             post_fun=post_fun,
    #             logger=logger,
    #             n_processes=args.n_processes,
    #             temperature=args.temperature,
    #             timeout=args.timeout,
    #             max_try=args.inner_max_try,
    #             think=args.think,
    #         )
    #         success_problems, todo_problems = verify_and_exec_generator_for_environment_combined(batch_problems, 
    #                                                                    logger,
    #                                                                    debug=True,
    #                                                                    check_number=args.check_number,
    #                                                                    deep_test_times=args.deep_test_times,
    #                                                                    breadth_test_times=args.breadth_test_times,
    #                                                                    sandboxfusion_url=args.sandbox_url,
    #                                                                    different_output_limit=args.different_output_limit,
    #                                                                    max_output_rate=args.max_output_rate)

    #         output_problems.extend(success_problems)

    #         next_attempt_problems.extend(todo_problems)
            
    #         save_output_jsonl(output_problems, save_dir_path=save_dir_path,  logger=logger, save_name=args.save_name, meta_name=args.save_meta_name)
    #         logger.info(f"success={len(output_problems)} | retry_next={len(todo_problems)}")

    #     left_problems = next_attempt_problems
    #     next_attempt_problems = []
    #     logger.info(f"End of Attempt {attempt}: accumulated={len(output_problems)} | remaining={len(left_problems)}")

    # logger.info(f"Done. total_completed={len(output_problems)} | total_input={len(examples)}")

if __name__ == "__main__":
    main()
