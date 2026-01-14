# -*- coding: utf-8 -*-
import os
import math
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional



from prompt import answer_problem_prompt,train_prompt
from logger import setup_logger
from process_dataset import load_and_prepare_dataset,prepare_examples,save_output_parquet
from extract import show_literal_newlines
from after_extract import verify_and_extract_test_case
import copy
from exec_and_verify import sandboxfusion_run
from tqdm.auto import tqdm




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
    parser.add_argument("--sandbox_url",type=str,default=None,help="The sandboxfusion url for code execution.")
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
    
    group_problem_list = []

    # 统一的进度条样式（可按需调整）
    BAR_FMT = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"

    examples_bar = tqdm(
        examples,
        desc="Examples",
        unit="ex",
        position=0,
        leave=True,
        ncols=110,
        colour="cyan",
        bar_format=BAR_FMT,
    )

    for example in examples_bar:
        generate_logic_problem_function = example['generate_logic_problem']['raw_code']
        group_problems = []
        success_cnt = 0
        error_cnt = 0

        # 内层进度条（每个 example 一条）
        tc_total = len(example['test_case_list'])
        inner_desc = f"TC | {example.get('id', '')} · {example.get('title', '')}"
        tc_bar = tqdm(
            total=tc_total,
            desc=inner_desc[:80],  # 避免标题过长撑爆
            unit="tc",
            position=1,
            leave=False,
            ncols=110,
            colour="green",
            bar_format=BAR_FMT,
        )

        # 如果本地执行，需要提前 exec 一次函数定义
        if not args.sandbox_url:
            try:
                exec(generate_logic_problem_function, globals())
            except Exception as e:
                logger.error(f"Error in exec generate_logic_problem function. {e}")
                tc_bar.close()
                # 继续到下一个 example
                continue

        for idx, test_case in enumerate(example['test_case_list']):
            try:
                if args.sandbox_url:
                    literal_input = show_literal_newlines(test_case['input'])
                    sandbox_generate_logic_problem_function = generate_logic_problem_function + f'''
if __name__ == "__main__":
    print(generate_logic_problem("{literal_input}"))
'''
                    ret = sandboxfusion_run(
                        args.sandbox_url,
                        sandbox_generate_logic_problem_function,
                        logger=logger,
                        language='python',
                        stdin=""
                    )
                    if ret["ok"]:
                        logic_problem = ret['run_result']["stdout"]
                    else:
                        logger.error(f"Sandbox error for test case {test_case}, {ret}")
                        error_cnt += 1
                        tc_bar.set_postfix_str(f"ok={success_cnt} err={error_cnt}")
                        tc_bar.update(1)
                        continue
                else:
                    logic_problem = generate_logic_problem(test_case['input'])

                if not logic_problem or str(logic_problem).startswith("None"):
                    logger.error("Error logic_problem is None/empty.")
                    error_cnt += 1
                    tc_bar.set_postfix_str(f"ok={success_cnt} err={error_cnt}")
                    tc_bar.update(1)
                    continue

                # 成功生成
                group_problems.append({
                    "problem": logic_problem,
                    "reward_model": {
                        "ground_truth": f"\\boxed{{{test_case['output']}}}"
                    },
                    "source": f"logic_{example['source']}",
                    "id": f"{example['id']}_{idx}",
                    "raw_id": example['id'],
                    "title": f"{example['title']}_{idx}"
                })
                success_cnt += 1

            except Exception as e:
                logger.error(f"Error in generate logic problem for test case {test_case}, {e}")
                error_cnt += 1

            # 更新内层进度条状态
            tc_bar.set_postfix_str(f"ok={success_cnt} err={error_cnt}")
            tc_bar.update(1)

        tc_bar.close()

        # 外层条也展示一下当前 example 的累计情况
        examples_bar.set_postfix_str(f"last_ok={success_cnt} last_err={error_cnt}")

        if len(group_problems) > 0:
            group_problem_list.append(group_problems)

    save_output_parquet(group_problem_list, save_dir_path, logger, args.save_name, args.save_meta_name)

if __name__ == "__main__":
    main()
