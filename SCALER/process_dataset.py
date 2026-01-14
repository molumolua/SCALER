from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from datasets import load_dataset
from typing import Any, Dict, List, Optional
from pathlib import Path
try:
    from extract import _extract_code_from_row
except:
    from SCALER.extract import _extract_code_from_row
from datetime import datetime
import json
import random

def _drop_heavy_columns(ds, drop_list: list, logger):
    # 根据需要删除超大列
    if not drop_list:
        return ds
    ds = ds.remove_columns(drop_list)
    logger.info(f"Dropped columns: {drop_list}")
    return ds


def _build_parquet_data_files(parquet_dir: Path, split=None, file_glob=None) -> List[str]:
    """根据 split 和通配符收集本地 parquet 分片。"""
    if file_glob:
        patterns = [file_glob]
    else:
        # 默认按 split 前缀收集：train-*.parquet / test-*.parquet / valid-*.parquet
        patterns = [f"{split}-*.parquet"]

    files: List[str] = []
    for pat in patterns:
        files.extend([str(p) for p in sorted(parquet_dir.glob(pat))])

    return files

def _build_jsonl_data_files(jsonl_dir: Path, split=None, file_glob=None) -> List[str]:
    """根据 split 和通配符收集本地 parquet 分片。"""
    if file_glob:
        patterns = [file_glob]
    else:
        # 默认按 split 前缀收集：train-*.jsonl / test-*.jsonl / valid-*.jsonl
        patterns = [f"{split}-*.jsonl"]

    files: List[str] = []
    for pat in patterns:
        files.extend([str(p) for p in sorted(jsonl_dir.glob(pat))])

    return files


from collections.abc import Mapping
from typing import Any

def restore_scales_map_fn(
    example: Any,
    dict_keys=("default_scale",),
    list_dict_keys=("small_scales", "large_scales"),
    drop_json_shadow=True,
    json_suffix="_json",
):
    """
    将 *_json 字段恢复为原名（default_scale/small_scales/large_scales），并可删除影子字段。
    - 不在原地修改；构造新 dict，避免“dictionary changed size during iteration”
    - 兼容 datasets.formatting.formatting.LazyRow / Mapping
    - 递归处理嵌套结构（list / dict）
    """
    def _try_json_load(x):
        if isinstance(x, str):
            try:
                return json.loads(x,
                                  parse_constant=str,
                                  parse_float=str,
                                  parse_int=str)  # 遇到 NaN/Infinity/null 直接返回 None
            except Exception:
                return x
        return x

    def _restore(obj):
        # 1) 映射类型：构造新 dict
        if isinstance(obj, Mapping):
            # 先将可能的 LazyRow 转成普通 dict，再遍历快照
            src = dict(obj)
            res = {}
            for k, v in src.items():  # 遍历“快照”，不改 src
                v_restored = _restore(v)

                if isinstance(k, str) and k.endswith(json_suffix):
                    base = k[: -len(json_suffix)]
                    if base in dict_keys or base in list_dict_keys:
                        # 写入原名字段
                        if isinstance(v_restored, list):
                            res[base] = [_try_json_load(e) for e in v_restored]
                        else:
                            res[base] = _try_json_load(v_restored)
                        # 是否保留影子字段
                        if not drop_json_shadow:
                            res[k] = v_restored
                        # 注意：不把影子字段写入 res（=删除）
                    else:
                        # 普通 *_json，但不在白名单：原样放回
                        res[k] = v_restored
                else:
                    # 非 *_json：原样放回
                    res[k] = v_restored
            return res

        # 2) 列表：逐元素恢复
        if isinstance(obj, (list, tuple)):
            return [_restore(x) for x in obj]

        # 3) 标量
        return obj

    # 入口：把 LazyRow 等转为普通 dict 再恢复
    example = dict(example) if isinstance(example, Mapping) else example
    return _restore(example)

def load_and_prepare_dataset(
    load_dir,
    load_type: str,
    logger,
    split=None,
    file_glob=None,
    drop_list=[],
):
    if isinstance(load_dir, str):
        from pathlib import Path
        load_dir = Path(load_dir)

    if load_type == "json":
        data_files = _build_jsonl_data_files(load_dir, split, file_glob)
    elif load_type == "parquet":
        data_files = _build_parquet_data_files(load_dir, split, file_glob)
    else:
        raise ValueError(f"Unsupported load_type={load_type!r}")

    if not data_files:
        raise FileNotFoundError(
            f"No {load_type} files matched in {load_dir} (split={split}, glob={file_glob or f'{split}*.{load_type}'})"
        )

    logger.info(
        f"Loading {load_type} files ({len(data_files)} found). "
        f"Example paths: {data_files[:3]}{' ...' if len(data_files) > 3 else ''}"
    )
    ds = load_dataset(load_type, data_files=data_files, split="train")
    logger.info(f"Columns: {ds.column_names}")

    # ds = _drop_heavy_columns(ds, drop_list=drop_list, logger=logger)
    # # 关键：恢复并删除 *_json 影子字段，只保留 default_scale / small_scales / large_scales
    # ds = ds.map(
    #     restore_scales_map_fn,
    #     fn_kwargs=dict(
    #         dict_keys=("default_scale",),
    #         list_dict_keys=("small_scales", "large_scales"),
    #         drop_json_shadow=True,   # ← 打开
    #         json_suffix="_json",
    #     ),
    #     desc="Restoring *_json scales → dict/list[dict] (and dropping shadows)",
    #     load_from_cache_file=False  # ← 强制不走缓存
    # )

    # print(ds[0]['extract_number']['small_scales'])
    return ds


def prepare_examples(ds,logger,start_idx=0,max_rows=None,extract_code=False):
    '''
    从 各种可能的字段中获取code，并且存在code字段中
    '''
    total = len(ds)
    if start_idx >= total:
        logger.warning(f"start_problem_idx ({start_idx}) >= dataset size ({total}); nothing to do.")
        return []
    
    # 决定要取多少条
    end_idx = total if (max_rows is None  or max_rows == -1 )else min(total, start_idx + max_rows)
    n_rows = end_idx - start_idx
    logger.info(f"Dataset size={total}, taking rows [{start_idx}:{end_idx}) -> {n_rows} rows")
    # 为了稳定、低内存：分块 to_list
    # 这里的块大小跟 batch_size 无关，只是读取阶段的切片大小
    CHUNK = 10_000
    examples: List[Dict[str, Any]] = []
    taken = 0
    for begin in range(start_idx, end_idx, CHUNK):
        end = min(begin + CHUNK, end_idx)
        rows = ds.select(range(begin, end)).to_list()

        for r in rows:
            ex = dict(r)
            if extract_code:
                code = _extract_code_from_row(r)
                if not code:
                    # 没拿到代码就跳过
                    continue
                ex["code"] = code  # 统一字段名，供 pre_fun 使用
            examples.append(ex)

        taken += len(rows)
        logger.info(f"Prepared examples: {len(examples)} / scanned rows: {taken}")

    if not examples:
        logger.warning("No example with usable code was found. Check column mapping / dataset schema.")

    return examples


def save_output_json(
    output_problems: List[Dict[str, Any]],
    save_dir_path: Path,
    logger,
    save_name: str | None = None,
    meta_name: str | None = None,
    *,
    # 可按需扩展：哪些 key 的值若是 dict / list[dict]，要转为 JSON 字符串
    dict_keys_to_dump: tuple[str, ...] = ("default_scale",),
    list_of_dict_keys_to_dump: tuple[str, ...] = ("small_scales", "large_scales"),
) -> None:
    """
    将 output_problems 写为 JSON（保持一层嵌套结构），
    但对 default_scale / small_scales / large_scales 做“定点 JSON 字符串化”，
    避免 Arrow/HF Datasets 对变形 dict 推断 schema 出错。
    """

    def _dump_json(o: Any) -> str:
        # 统一风格，便于去重/可读性
        return json.dumps(o, ensure_ascii=False, sort_keys=True)

    def _jsonable_with_policy(obj: Any) -> Any:
        """
        深度遍历，仅当 key 命中策略时将 dict/list[dict] 转为 JSON 字符串；
        其他情况遵循你的原始 _to_jsonable 逻辑。
        """
        # 标量直接返回
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj

        # list/tuple：逐元素处理（保持嵌套 list，不做扁平化）
        if isinstance(obj, (list, tuple)):
            return [_jsonable_with_policy(x) for x in obj]

        # dict：保留一层嵌套结构，仅对命中 key 的值进行“定点字符串化”
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                # 情况1：default_scale 等“单个 dict” → JSON 字符串
                if k in dict_keys_to_dump and isinstance(v, dict):
                    out[k + "_json"] = _dump_json(v)
                    # 若你想保留原字段，也可同时 out[k] = None 或直接不保留
                # 情况2：small_scales / large_scales 等“list[dict]” → list[str]
                elif k in list_of_dict_keys_to_dump and isinstance(v, (list, tuple)):
                    if all(isinstance(x, dict) for x in v):
                        out[k + "_json"] = [_dump_json(x) for x in v]  # list[str]
                    else:
                        # 混合类型时，仍递归处理，保证健壮性
                        out[k] = _jsonable_with_policy(v)
                else:
                    out[k] = _jsonable_with_policy(v)
            return out

        # 其他不可序列化类型降级为字符串
        return str(obj)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_json = save_dir_path / (save_name or "output_problems.json")
    out_meta = save_dir_path / (meta_name or "meta.json")

    with out_json.open("w", encoding="utf-8") as mf:
        json.dump(output_problems, mf, ensure_ascii=False, indent=2)
        
    # 写 meta
    meta = {
        "timestamp": ts,
        "total_completed": len(output_problems),
        "output_path": str(out_json),
        "policy": {
            "dict_keys_to_dump": list(dict_keys_to_dump),
            "list_of_dict_keys_to_dump": list(list_of_dict_keys_to_dump),
        }
    }
    if len(output_problems)>0:
        meta['example']=random.choice(list(output_problems.values()))
    else:
        logger.error("No example in output_problems.")
        
    with out_meta.open("w", encoding="utf-8") as mf:
        json.dump(meta, mf, ensure_ascii=False, indent=2)

    logger.info(f"Saved outputs: {out_json}  (rows={len(output_problems)})")
    logger.info(f"Saved meta:    {out_meta}")


def save_output_jsonl(
    output_problems: List[Dict[str, Any]],
    save_dir_path: Path,
    logger,
    save_name: str | None = None,
    meta_name: str | None = None,
    *,
    # 可按需扩展：哪些 key 的值若是 dict / list[dict]，要转为 JSON 字符串
    dict_keys_to_dump: tuple[str, ...] = ("default_scale",),
    list_of_dict_keys_to_dump: tuple[str, ...] = ("small_scales", "large_scales"),
) -> None:
    """
    将 output_problems 写为 JSONL（保持一层嵌套结构），
    但对 default_scale / small_scales / large_scales 做“定点 JSON 字符串化”，
    避免 Arrow/HF Datasets 对变形 dict 推断 schema 出错。
    """

    def _dump_json(o: Any) -> str:
        # 统一风格，便于去重/可读性
        return json.dumps(o, ensure_ascii=False, sort_keys=True)

    def _jsonable_with_policy(obj: Any) -> Any:
        """
        深度遍历，仅当 key 命中策略时将 dict/list[dict] 转为 JSON 字符串；
        其他情况遵循你的原始 _to_jsonable 逻辑。
        """
        # 标量直接返回
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj

        # list/tuple：逐元素处理（保持嵌套 list，不做扁平化）
        if isinstance(obj, (list, tuple)):
            return [_jsonable_with_policy(x) for x in obj]

        # dict：保留一层嵌套结构，仅对命中 key 的值进行“定点字符串化”
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                # 情况1：default_scale 等“单个 dict” → JSON 字符串
                if k in dict_keys_to_dump and isinstance(v, dict):
                    out[k + "_json"] = _dump_json(v)
                    # 若你想保留原字段，也可同时 out[k] = None 或直接不保留
                # 情况2：small_scales / large_scales 等“list[dict]” → list[str]
                elif k in list_of_dict_keys_to_dump and isinstance(v, (list, tuple)):
                    if all(isinstance(x, dict) for x in v):
                        out[k + "_json"] = [_dump_json(x) for x in v]  # list[str]
                    else:
                        # 混合类型时，仍递归处理，保证健壮性
                        out[k] = _jsonable_with_policy(v)
                else:
                    out[k] = _jsonable_with_policy(v)
            return out

        # 其他不可序列化类型降级为字符串
        return str(obj)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_jsonl = save_dir_path / (save_name or "output_problems.jsonl")
    out_meta = save_dir_path / (meta_name or "meta.json")

    # 写 JSONL（逐行，将策略应用于每条记录）
    with out_jsonl.open("w", encoding="utf-8") as wf:
        for ex in output_problems:
            safe = _jsonable_with_policy(ex)
            wf.write(json.dumps(safe, ensure_ascii=False) + "\n")

    # 写 meta
    meta = {
        "timestamp": ts,
        "total_completed": len(output_problems),
        "output_path": str(out_jsonl),
        "policy": {
            "dict_keys_to_dump": list(dict_keys_to_dump),
            "list_of_dict_keys_to_dump": list(list_of_dict_keys_to_dump),
        }
    }
    if len(output_problems)>0:
        meta['example']=random.choice(output_problems)
    else:
        logger.error("No example in output_problems.")
        
    with out_meta.open("w", encoding="utf-8") as mf:
        json.dump(meta, mf, ensure_ascii=False, indent=2)

    logger.info(f"Saved outputs: {out_jsonl}  (rows={len(output_problems)})")
    logger.info(f"Saved meta:    {out_meta}")


import pyarrow as pa
import pyarrow.parquet as pq
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

def save_output_parquet(
    output_problems ,
    save_dir_path: Path,
    logger,
    save_name: str | None = None,
    meta_name: str | None = None,
    *,
    # 可按需扩展：哪些 key 的值若是 dict / list[dict]，要转为 JSON 字符串
    dict_keys_to_dump: tuple[str, ...] = ("default_scale",),
    list_of_dict_keys_to_dump: tuple[str, ...] = ("small_scales", "large_scales"),
) -> None:
    """
    将 output_problems 写为 Parquet（保持一层嵌套结构），
    但对 default_scale / small_scales / large_scales 做“定点 JSON 字符串化”，
    避免 Arrow/HF Datasets 对变形 dict 推断 schema 出错。
    """

    def _dump_json(o: Any) -> str:
        # 统一风格，便于去重/可读性
        return json.dumps(o, ensure_ascii=False, sort_keys=True)

    def _jsonable_with_policy(obj: Any) -> Any:
        """
        深度遍历，仅当 key 命中策略时将 dict/list[dict] 转为 JSON 字符串；
        其他情况遵循你的原始 _to_jsonable 逻辑。
        """
        # 标量直接返回
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj

        # list/tuple：逐元素处理（保持嵌套 list，不做扁平化）
        if isinstance(obj, (list, tuple)):
            return [_jsonable_with_policy(x) for x in obj]

        # dict：保留一层嵌套结构，仅对命中 key 的值进行“定点字符串化”
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                # 情况1：default_scale 等“单个 dict” → JSON 字符串
                if k in dict_keys_to_dump and isinstance(v, dict):
                    out[k + "_json"] = _dump_json(v)
                # 情况2：small_scales / large_scales 等“list[dict]” → list[str]
                elif k in list_of_dict_keys_to_dump and isinstance(v, (list, tuple)):
                    if all(isinstance(x, dict) for x in v):
                        out[k + "_json"] = [_dump_json(x) for x in v]  # list[str]
                    else:
                        # 混合类型时，仍递归处理，保证健壮性
                        out[k] = _jsonable_with_policy(v)
                else:
                    out[k] = _jsonable_with_policy(v)
            return out

        # 其他不可序列化类型降级为字符串
        return str(obj)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_parquet = save_dir_path / (save_name or "output_problems.parquet")
    out_meta = save_dir_path / (meta_name or "meta.json")

    # 处理数据：将每条记录转换成可以保存为表格格式的数据
    processed_data = [_jsonable_with_policy(ex) for ex in output_problems]

    # Convert processed data to PyArrow Table
    try:
        # We can directly use the pyarrow Table.from_pandas if the data is in DataFrame format
        import pandas as pd
        df = pd.json_normalize(processed_data)
        table = pa.Table.from_pandas(df)
    except Exception as e:
        logger.error(f"Error during converting data to PyArrow Table: {e}")
        raise

    # Save the table as a Parquet file
    pq.write_table(table, out_parquet)

    # Write meta
    meta = {
        "timestamp": ts,
        "total_completed": len(output_problems),
        "output_path": str(out_parquet),
        "policy": {
            "dict_keys_to_dump": list(dict_keys_to_dump),
            "list_of_dict_keys_to_dump": list(list_of_dict_keys_to_dump),
        },
    }
    
    if len(output_problems)>0:
        meta['example']=random.choice(output_problems)
    else:
        logger.error("No example in output_problems.")
        
    with out_meta.open("w", encoding="utf-8") as mf:
        json.dump(meta, mf, ensure_ascii=False, indent=2)

    logger.info(f"Saved outputs: {out_parquet}  (rows={len(output_problems)})")
    logger.info(f"Saved meta:    {out_meta}")
    

# fix_jsonl_for_arrow.py
import json, re, math, pathlib

# 支持：1 000 / 1 000 / 1,000 / 10^6 / 10^{18} / 标准科学计数
SPACE_SEP = re.compile(r'^[+-]?\d{1,3}(?:[ \u00A0\u202F]\d{3})+(?:\.\d+)?$')  # 普通空格/不换行空格/窄空格
COMMA_SEP = re.compile(r'^[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?$')
SCI       = re.compile(r'^[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?$')
POW       = re.compile(r'^\s*([+-]?\d+(?:\.\d+)?)\s*\^\s*\{?\s*([+-]?\d+)\s*\}?\s*$')

def to_number_if_possible(x):
    if isinstance(x, (int, float)):  # 已是数
        return x
    if not isinstance(x, str):
        return x

    s = x.strip()
    # 1) 千分位空格：去掉所有空格/不换行空格/窄空格
    if SPACE_SEP.match(s):
        t = s.replace(' ', '').replace('\u00A0', '').replace('\u202F', '')
        try:
            v = float(t); return int(v) if v.is_integer() else v
        except: return x

    # 2) 千分位逗号
    if COMMA_SEP.match(s):
        t = s.replace(',', '')
        try:
            v = float(t); return int(v) if v.is_integer() else v
        except: return x

    # 3) caret 幂：10^6 / 10^{18}
    m = POW.match(s)
    if m:
        base = float(m.group(1)); exp = int(m.group(2))
        try:
            v = base ** exp; return int(v) if float(v).is_integer() else float(v)
        except OverflowError:
            return s
        except:
            return s

    # 4) 标准数字/科学计数
    if SCI.match(s):
        try:
            v = float(s); return int(v) if v.is_integer() else v
        except: return x

    return x

def normalize(obj):
    if isinstance(obj, dict):
        return {k: normalize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [normalize(v) for v in obj]
    return to_number_if_possible(obj)


def normalize_jsonl_file(in_path, out_path):
    with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            obj = json.loads(line)
            obj2 = normalize(obj)
            fout.write(json.dumps(obj2, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_path", type=str, required=True)
    parser.add_argument("--out_path", type=str, required=True)
    args = parser.parse_args()
    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    normalize_jsonl_file(in_path, out_path)
    print(f"Normalized {in_path} -> {out_path}")