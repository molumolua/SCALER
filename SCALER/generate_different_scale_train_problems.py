from __future__ import annotations
import random
from typing import Any, Dict, List, Sequence, Union
# -*- coding: utf-8 -*-
import os
import math
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd


from prompt import answer_problem_prompt,train_prompt
from logger import setup_logger
from process_dataset import load_and_prepare_dataset,prepare_examples,save_output_parquet
from extract import show_literal_newlines
from after_extract import verify_and_extract_test_case
import copy
from exec_and_verify import sandboxfusion_run
from tqdm.auto import tqdm


def stratified_packs_by_nested_key(
    dataset: Any,
    logger,
    sizes: Union[int, Sequence[int]],
    key_path: str = "extra_info.raw_id",   # ← 现在走嵌套路径
    seed: int = 42,
    sample_with_replacement: bool = False,
    shuffle_each_pack: bool = True,
    max_size=10000
):
    """
    按嵌套字段路径(默认 extra_info.raw_id)分层；对 sizes 中每个 k：
      - 每个分层随机取 k 条（可选有放回）
      - 合并为一个新数据集
      - 新增：对每个数据集进行 repeat 操作，重复次数为 max_size / size
    返回与 sizes 等长的“新数据集列表”，类型与输入保持一致。
    """

    # ---- 类型判断 ----
    def _is_hf(obj) -> bool:
        try:
            from datasets import Dataset
            return isinstance(obj, Dataset)
        except Exception:
            return False

    def _is_df(obj) -> bool:
        try:
            import pandas as pd
            return isinstance(obj, pd.DataFrame)
        except Exception:
            return False

    def _len(obj) -> int:
        if _is_hf(obj) or _is_df(obj) or isinstance(obj, list): return len(obj)
        raise TypeError("Unsupported dataset type (HF Dataset / pandas DataFrame / list[dict]).")

    # ---- 嵌套列提取 ----
    keys = key_path.split(".")
    def _extract_from_row(row: Dict[str, Any]):
        cur = row
        for k in keys:
            if not isinstance(cur, dict) or (k not in cur):
                raise KeyError(f"Missing nested key '{key_path}' in a row.")
            cur = cur[k]
        return cur

    def _get_nested_column(obj, key_path: str) -> List[Any]:
        if _is_hf(obj):
            # HF: 直接读需要的顶层列，再逐行取嵌套
            # 先尽量只拿第一段列，避免 to_list 全列开销
            top = keys[0]
            if top not in obj.column_names:
                raise KeyError(f"HF Dataset missing top-level column '{top}' for path '{key_path}'.")
            col = obj[top]  # list-like of dicts
            return [_extract_from_row({top: v}) for v in col]  # 包一层顶级键以复用提取逻辑
        elif _is_df(obj):
            import pandas as pd
            top = keys[0]
            if top not in obj.columns:
                raise KeyError(f"DataFrame missing top-level column '{top}' for path '{key_path}'.")
            s = obj[top]
            # 若 top 列为 dict/obj，按行提取
            return [ _extract_from_row({top: v}) for v in s.tolist() ]
        elif isinstance(obj, list):
            return [ _extract_from_row(r) for r in obj ]
        else:
            raise TypeError("Unsupported dataset type for nested column access.")

    def _subset(obj, idxs: List[int]):
        if _is_hf(obj):
            return obj.select(idxs)
        elif _is_df(obj):
            return obj.iloc[idxs].reset_index(drop=True)
        elif isinstance(obj, list):
            return [obj[i] for i in idxs]
        else:
            raise TypeError("Unsupported dataset type for indexing.")

    def _concat_like(objs: List[Any]):
        if not objs:
            if _is_hf(dataset):
                from datasets import Dataset
                return Dataset.from_list([])
            elif _is_df(dataset):
                import pandas as pd
                return pd.DataFrame()
            else:
                return []
        if _is_hf(dataset):
            # 为简洁与稳妥：转换为 list 再重建
            from datasets import Dataset
            as_list = []
            for part in objs:
                as_list.extend(part.to_list())
            return Dataset.from_list(as_list)
        elif _is_df(dataset):
            import pandas as pd
            return pd.concat(objs, axis=0, ignore_index=True)
        else:
            merged = []
            for part in objs: merged.extend(part)
            return merged

    # ---- 基本校验 ----
    n = _len(dataset)
    if n == 0:
        logger.warning("输入数据集为空，返回空列表。")
        return []

    sizes = [sizes] if isinstance(sizes, int) else list(sizes)

    # ---- 分层索引 ----
    try:
        raw_ids = _get_nested_column(dataset, key_path)
    except Exception as e:
        raise KeyError(f"无法按嵌套路径 '{key_path}' 读取分层键。") from e

    buckets: Dict[Any, List[int]] = {}
    for i, rid in enumerate(raw_ids):
        buckets.setdefault(rid, []).append(i)
    class_sizes = {}
    filter_buckets = {}
    print("Max_size,",max_size)
    for k,v in buckets.items():
        if len(v)>=max_size:
            class_sizes[k]=len(v)
            filter_buckets[k]=v
            
    logger.info(f"[stratified] 分层键='{key_path}' | 类数={len(filter_buckets)} | 总样本={n} | 每类计数={class_sizes}")

    rng = random.Random(seed)
    packs = []
    for k in sizes:
        pack_indices: List[int] = []
        truncated = {}
        for rid, idxs in filter_buckets.items():
            cnt = len(idxs)
            if cnt == 0: continue
            if sample_with_replacement:
                chosen = [rng.choice(idxs) for _ in range(k)]
            else:
                kk = min(k, cnt)
                if kk < k: truncated[rid] = (k, kk)
                if kk == 0: continue
                chosen = rng.sample(idxs, kk)
            pack_indices.extend(chosen)

        if shuffle_each_pack and len(pack_indices) > 1:
            rng.shuffle(pack_indices)

        if not sample_with_replacement:
            if truncated:
                logger.warning(f"[stratified] k={k}（无放回）部分类不足已截断：{truncated}")
            logger.info(f"[stratified] k={k}（无放回）期望≤{k*len(filter_buckets)}，得到={len(pack_indices)}")
        else:
            logger.info(f"[stratified] k={k}（有放回）期望={k*len(filter_buckets)}，得到={len(pack_indices)}")

        packs.append(_subset(dataset, pack_indices))
    
    # ---- 重复数据集 ----
    repeated_packs = []
    for pack, size in zip(packs, sizes):
        repeat_count = max_size // size  # 计算每个数据集的重复次数
        # print(repeat_count,len(pack))
        repeated_pack = list(pack) *repeat_count
        repeated_packs.append(repeated_pack) 

    # 返回重复后的数据集
    return repeated_packs



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
    parser.add_argument("--size_list", type=int, nargs="+", required=True)
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
        extract_code=False)
    
    if not examples:
        logger.info("No examples with usable code. Exit.")
        return
    
    sizes=args.size_list
    max_size = sizes[-1]
    print("sizes,",sizes)
    
    output_datasets=stratified_packs_by_nested_key(
        dataset=examples,
        logger=logger,
        sizes=sizes,
        max_size=max_size
    )
    for size,output_datasets in zip(sizes,output_datasets):
        save_name = f"train_logic_{size}.parquet"
        save_meta_name = f"train_logic_{size}_meta.json"
        processed_df = pd.DataFrame(output_datasets)
        processed_df.to_parquet(Path(str(save_dir_path)+"/"+save_name))
        
            
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_meta = save_dir_path / (save_meta_name or "meta.json")


        # Write meta
        meta = {
            "timestamp": ts,
            "total":len(output_datasets)
        }
        
        with out_meta.open("w", encoding="utf-8") as mf:
            json.dump(meta, mf, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
