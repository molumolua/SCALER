# -*- coding: utf-8 -*-
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from tqdm import tqdm

# 依赖你现有工程中的工具
from prompt import train_prompt,no_think_prompt
from logger import setup_logger

# 新增：直接读取 .arrow
try:
    from datasets import Dataset
except Exception as e:
    raise ImportError(
        "Please install `datasets` (pip install datasets) to read .arrow files."
    ) from e


# ----------------- 可配置常量 -----------------
ARROW_PATH = "/inspire/hdd/global_user/xucaijun-253108120121/Dataset/GPQA/diamond_train_saved/data-00000-of-00001.arrow"
SAVE_DIR = "./Dataset"
SAVE_NAME = "NOTHINK-GPQA-Diamond-Test.parquet"  # 统一为 Parquet 扩展名
DATA_SOURCE_TAG = "option_GPQA-Diamond"

# 新增：重复保存的次数
REPEAT_TIMES = 1
# --------------------------------------------


def _first_nonempty(d: Dict[str, Any], keys: List[str], default=None):
    """按顺序返回第一个存在且不为 None 的字段值；支持点号路径。"""
    for k in keys:
        if "." in k:
            cur = d
            ok = True
            for part in k.split("."):
                if isinstance(cur, dict) and part in cur:
                    cur = cur[part]
                else:
                    ok = False
                    break
            if ok and cur is not None:
                return cur
        else:
            if k in d and d[k] is not None:
                return d[k]
    return default


def _format_mc_question(question_text: str,
                        options: Optional[List[str]] = None) -> str:
    """把题干与选项（若提供）拼成统一的 MC 题目文本。"""
    if options and isinstance(options, (list, tuple)):
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        lines = [question_text.strip(), ""]
        for i, opt in enumerate(options):
            if opt is None:
                continue
            lines.append(f"({letters[i]}). {str(opt).strip()}")
        return "\n".join(lines)
    return question_text.strip()


def _normalize_ground_truth(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    归一化 ground truth：
    - 优先提取 A/B/C/D 之一（不区分大小写）
    - 若给定正确索引（0-based/1-based），映射为字母
    - 最终返回 {"ground_truth": "A"/"B"/...}，尽量保证下游一致
    """
    # 直接的字母答案
    ans_letter = _first_nonempty(
        sample,
        [
            "reward_model.ground_truth",  # 你的旧字段
            "ground_truth",
            "answer",
            "label_text",      # 有些数据会直接存字母
            "gt_letter",
        ],
        default=None,
    )
    if isinstance(ans_letter, str) and len(ans_letter.strip()) == 1:
        ch = ans_letter.strip().upper()
        if ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            return {"ground_truth": f"\\boxed{{{ch}}}"}

    # 选项索引（0 或 1 开始）
    idx = _first_nonempty(
        sample,
        [
            "answer_index",
            "correct_index",
            "label",          # 有时是数字
            "gt_index",
        ],
        default=None,
    )
    if isinstance(idx, (int, float)):
        idx = int(idx)
        # 猜测 0/1 开始：都试着映射
        for base in (0, 1):
            mapped = idx - base
            if 0 <= mapped < 26:
                return {"ground_truth": chr(ord('A') + mapped)}

    # 选项文本与正确文本比对（如果给了 correct_text）
    correct_text = _first_nonempty(sample, ["correct_text", "answer_text"], default=None)
    options = _first_nonempty(sample, ["options", "choices", "options_text"], default=None)
    if correct_text and isinstance(options, list):
        for i, opt in enumerate(options):
            if str(opt).strip() == str(correct_text).strip():
                if 0 <= i < 26:
                    return {"ground_truth": chr(ord('A') + i)}

    # 实在不行就返回原始的（可能是自由文本）
    if ans_letter is not None:
        return {"ground_truth": str(ans_letter).strip()}
    return {"ground_truth": None}


def load_arrow_dataset(path: str) -> Dataset:
    """使用 HF datasets 直接加载 .arrow 单分片文件。"""
    if not Path(path).exists():
        raise FileNotFoundError(f"Arrow file not found: {path}")
    ds = Dataset.from_file(path)
    return ds


def build_output_rows(ds: Dataset, logger) -> List[Dict[str, Any]]:
    """从原始 Dataset 构造输出训练样本行。"""
    out: List[Dict[str, Any]] = []
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    for idx in tqdm(range(len(ds)), desc="Building samples"):
        row = ds[int(idx)]

        # 可能的题干字段
        question = _first_nonempty(
            row,
            ["question", "prompt", "problem", "stem", "input", "question_text"],
            default=None,
        )

        # 选项/choices（若有）
        options = _first_nonempty(row, ["options", "choices", "options_text"], default=None)

        # 过滤明显错误的记录
        problem_marker = _first_nonempty(row, ["problem", "status"], default="")
        if isinstance(problem_marker, str) and problem_marker.startswith("Error"):
            continue

        if not question:
            # 无题干，跳过
            logger.warning(f"[{idx}] missing question/stem; skipped.")
            continue

        # 格式化题目文本（附上 A-D）
        mc_text = _format_mc_question(question, options)

        problem = (
            "Please solve this multiple choice question.\n"
            f"{mc_text}\n"
            "Provide your answer in the format '\\boxed{{X}}' where X is one uppercase letter.\n"
        )

        gt = _normalize_ground_truth(row)
        sample = {
            "prompt": no_think_prompt(problem),
            "reward_model": gt,
            "data_source": DATA_SOURCE_TAG,
            "extra_info": {"index": idx},
        }
        out.append(sample)

    return out


def main():
    logger = setup_logger()
    logger.info(f"Loading Arrow from: {ARROW_PATH}")

    save_dir_path = Path(SAVE_DIR)
    save_dir_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output dir: {save_dir_path}")

    ds = load_arrow_dataset(ARROW_PATH)
    logger.info(f"Loaded dataset: {ds}")

    rows = build_output_rows(ds, logger)
    if not rows:
        logger.info("No usable rows. Exit.")
        return

    df = pd.DataFrame(rows)

    # ===== 关键改动：将数据复制 REPEAT_TIMES 遍 =====
    if REPEAT_TIMES > 1:
        logger.info(f"Replicating dataset {REPEAT_TIMES}x before saving...")
        df = pd.concat([df] * REPEAT_TIMES, ignore_index=True)
    # ==============================================

    out_path = save_dir_path / SAVE_NAME
    df.to_parquet(out_path, index=False)
    logger.info(f"Saved {len(df)} rows to {out_path.resolve()}")


if __name__ == "__main__":
    main()
