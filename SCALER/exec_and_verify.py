# -*- coding: utf-8 -*-
import os
import math
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from tqdm import tqdm
try:
    from logger import setup_logger
    from process_dataset import load_and_prepare_dataset,prepare_examples,save_output_parquet
except:
    from SCALER.logger import setup_logger
    from SCALER.process_dataset import load_and_prepare_dataset,prepare_examples,save_output_parquet
import copy
import shlex
import subprocess
from pathlib import Path

def fix_newlines_in_cpp_strings(code: str) -> str:
    """
    将 C++ 源码中【双引号字符串内部】的真实换行符替换为字面量 '\\n'，
    以修复像 printf("%d %d\n", ...) 被错误写成 printf("%d %d
    ", ...) 的情况。
    仅处理双引号字符串，不动原本的 \\n、\\t、外部换行、注释等。
    """
    out = []
    in_str = False      # 是否在双引号字符串内部
    escaped = False     # 上一个字符是否是反斜杠（处理 \" \\ 等）
    i = 0
    while i < len(code):
        ch = code[i]

        if in_str:
            if escaped:
                # 前一个字符是反斜杠，当前字符原样放入（保持已有的转义如 \"、\\n）
                out.append(ch)
                escaped = False
            else:
                if ch == '\\':
                    out.append(ch)
                    escaped = True
                elif ch == '"':   # 结束字符串
                    out.append(ch)
                    in_str = False
                elif ch == '\r':  # 处理 \r\n 或单独 \r
                    # 丢弃 \r，自行判断下一位是否 \n
                    # 不把它输出到源码字符串里
                    if i + 1 < len(code) and code[i+1] == '\n':
                        # 将 CRLF 作为一个换行处理
                        out.append('\\n')
                        i += 1  # 跳过 \n
                    else:
                        out.append('\\n')
                elif ch == '\n':
                    # 这是不合法的：字符串内部的真实换行，替换为字面量 \n
                    out.append('\\n')
                else:
                    out.append(ch)
        else:
            if ch == '"':         # 进入字符串
                out.append(ch)
                in_str = True
                escaped = False
            else:
                out.append(ch)

        i += 1

    # 保证文件末尾有换行
    if not out or out[-1] != '\n':
        out.append('\n')
    return ''.join(out)
def fix_newlines_in_python_strings(code: str) -> str:
    """
    将 Python 源码中【字符串内部】的真实换行符替换为字面量 '\\n'，
    以修复像 print("Hello\nWorld") 被错误写成 print("Hello
    World") 的情况。
    仅处理字符串，不动原本的 \\n、\\t、外部换行、注释等。
    支持单引号和双引号字符串。
    """
    out = []
    in_str = False      # 是否在字符串内部
    in_single_quote = False  # 是否在单引号字符串内部
    in_double_quote = False  # 是否在双引号字符串内部
    escaped = False     # 上一个字符是否是反斜杠（处理 \' \" 等）
    i = 0

    while i < len(code):
        ch = code[i]

        if in_str:
            if escaped:
                # 前一个字符是反斜杠，当前字符原样放入（保持已有的转义如 \'、\"、\\n）
                out.append(ch)
                escaped = False
            else:
                if ch == '\\':
                    out.append(ch)
                    escaped = True
                elif ch == "'" and in_single_quote:   # 结束单引号字符串
                    out.append(ch)
                    in_single_quote = False
                    in_str = False
                elif ch == '"' and in_double_quote:  # 结束双引号字符串
                    out.append(ch)
                    in_double_quote = False
                    in_str = False
                elif ch == '\r':  # 处理 \r\n 或单独 \r
                    # 丢弃 \r，自行判断下一位是否 \n
                    if i + 1 < len(code) and code[i+1] == '\n':
                        # 将 CRLF 作为一个换行处理
                        out.append('\\n')
                        i += 1  # 跳过 \n
                    else:
                        out.append('\\n')
                elif ch == '\n':
                    # 这是不合法的：字符串内部的真实换行，替换为字面量 \n
                    out.append('\\n')
                else:
                    out.append(ch)

        else:
            if ch == '"':         # 进入双引号字符串
                out.append(ch)
                in_str = True
                in_double_quote = True
                escaped = False
            elif ch == "'":       # 进入单引号字符串
                out.append(ch)
                in_str = True
                in_single_quote = True
                escaped = False
            else:
                out.append(ch)

        i += 1

    # 保证文件末尾有换行
    if not out or out[-1] != '\n':
        out.append('\n')
    return ''.join(out)



def write_and_build_referenece_solution(
    solution_code_str,
    lang,
    sol_code,
    sol_bin,
    debug,
    logger,
    cpp_std: str = "c++17",
    compile_timeout_sec: int = 20,
):
    if lang == "cpp":
        # 处理字符串里的裸换行，避免 printf("...\n") 被拆断
        sol_code.write_text(fix_newlines_in_cpp_strings(solution_code_str), encoding="utf-8")

        cxx = os.environ.get("CXX", "g++")
        common_flags = ["-std=" + cpp_std, "-O2", "-pipe", "-static-libstdc++", "-static-libgcc"]

        if logger and debug: logger.info("Compiling reference C++ solution...")
        try:
            subprocess.run(
                [cxx, str(sol_code), "-o", str(sol_bin), *common_flags],
                check=True, timeout=compile_timeout_sec, capture_output=True
            )
            return True
        except subprocess.CalledProcessError as e:
            if logger and debug:
                logger.error("Failed to compile reference solution.")
                logger.error(e.stderr.decode(errors="ignore"))
                # 编译失败，返回空 outputs（长度与 inputs 对齐）
            return False
        except subprocess.TimeoutExpired:
            if logger and debug: logger.error("Reference solution compilation timed out.")
            return False
    elif lang == "python":
        sol_code.write_text(solution_code_str, encoding="utf-8")
        return True
    else:
        if logger and debug: logger.error("Not CPP or Python code.")
        return False

def run_reference_solution(
    inputs: List[str], #多组输入的str
    sol_code:Path,
    sol_bin:Path,
    logger,
    debug,
    lang="cpp",
    run_timeout_sec: int = 10
):
    checked_list=[]
    if lang =="cpp":
        for i, inp in enumerate(inputs):
            try:
                proc = subprocess.run(
                    [str(sol_bin)],
                    input=inp,
                    text=True,
                    check=False,
                    timeout=run_timeout_sec,
                    capture_output=True
                )
            except subprocess.TimeoutExpired:
                if logger and debug: logger.warning(f"[ref #{i}] Solution timed out.")
                checked_list.append({
                    "input":inp,
                    "output":None,
                    "flag":False
                })
                continue
            except FileNotFoundError:
                if logger and debug: logger.error("Reference binary missing unexpectedly.")
                checked_list.append({
                    "input":inp,
                    "output":None,
                    "flag":False
                })
                continue
            

            if proc.returncode == 0:
                checked_list.append({
                    "input":inp,
                    "output":proc.stdout,
                    "flag":True
                })
            else:
                if logger and debug:
                    logger.warning(f"[ref #{i}] Non-zero exit {proc.returncode}. stderr:\n{proc.stderr}")
                checked_list.append({
                    "input":inp,
                    "output":None,
                    "flag":False
                })
    elif lang == "python":
        python_exe = os.environ.get("PYTHON", "python")
        for i, inp in enumerate(inputs, 1):
            try:
                proc = subprocess.run(
                    [python_exe, str(sol_code)],
                    input=inp,
                    text=True,
                    check=False,
                    timeout=run_timeout_sec,
                    capture_output=True
                )
            except subprocess.TimeoutExpired:
                if logger and debug: logger.warning(f"[ref #{i}] Python solution timed out.")
                checked_list.append({
                    "input":inp,
                    "output":None,
                    "flag":False
                })
                continue
            except FileNotFoundError:
                if logger and debug: logger.error("Python interpreter not found.")
                checked_list.append({
                    "input":inp,
                    "output":None,
                    "flag":False
                })
                continue

            if proc.returncode == 0:
                checked_list.append({
                    "input":inp,
                    "output":proc.stdout,
                    "flag":True
                })
            else:
                if logger and debug:
                    logger.warning(f"[ref #{i}] Non-zero exit {proc.returncode}. stderr:\n{proc.stderr}")
                checked_list.append({
                    "input":inp,
                    "output":None,
                    "flag":False
                })
    return checked_list

def build_and_run_reference_solution(
    solution_code: str,
    inputs: List[str],
    logger=None,
    debug=False,
    lang="cpp",
    *,
    cpp_std: str = "c++17",
    compile_timeout_sec: int = 20,
    run_timeout_sec: int = 10,
) -> List[str]:
    """
    编译/准备参考解，并对每个 input 运行获得标准输出。
    支持 C++ (g++) 与 Python（系统 python）。
    返回 outputs（与 inputs 一一对应）。
    """
    cwd = Path.cwd()
    src_dir = (cwd / ".." / "testlib").resolve()
    bin_dir = (cwd / "testlib").resolve()
    src_dir.mkdir(parents=True, exist_ok=True)
    bin_dir.mkdir(parents=True, exist_ok=True)

    
    checked_list = []

    if lang == "cpp":
        sol_code = src_dir / "solution.cpp"
        sol_bin = bin_dir / "sol"
    elif lang == "python":  # python
        sol_code = bin_dir / "sol.py"
        sol_bin = None
    

    write_and_build_flag = write_and_build_referenece_solution(solution_code_str=solution_code,lang=lang,sol_code=sol_code,\
                                sol_bin=sol_bin,debug=debug,logger=logger,cpp_std=cpp_std,compile_timeout_sec=compile_timeout_sec)

    if not write_and_build_flag:
        if logger and debug: logger.error("Error in write and build.")
        return None

    
    
    checked_list = run_reference_solution(
        inputs=inputs,
        sol_code=sol_code,
        sol_bin=sol_bin,
        logger=logger,
        debug=debug,
        lang=lang,
        run_timeout_sec=run_timeout_sec
    )
    return checked_list

def import_needed_module_for_python(code_str):
    wrapped_code = f"""
import traceback
from string import *
from re import *
from datetime import *
from collections import *
from heapq import *
from bisect import *
from copy import *
from math import *
from random import *
from statistics import *
from itertools import *
from functools import *
from operator import *
from io import *
from sys import *
from json import *
from builtins import *
from typing import *
import string
import re
import datetime
import collections
import heapq
import bisect
import copy
import math
import random
import statistics
import itertools
import functools
import operator
import io
import sys
import json
{code_str}
"""
    return wrapped_code


import signal

class _Timeout:
    def __init__(self, seconds: int):
        self.seconds = seconds
        self._old_handler = None

    def __enter__(self):
        # 安装超时处理器
        try:
            self._old_handler = signal.getsignal(signal.SIGALRM)
            signal.signal(signal.SIGALRM, self._raise_timeout)
            signal.alarm(self.seconds)
        except Exception:
            # 这里不抛，让外层 with 捕获；run_generator_with_alarm 会统一处理
            pass
        return self

    def __exit__(self, exc_type, exc, tb):
        # 清理闹钟与恢复旧的 handler
        try:
            signal.alarm(0)
            if self._old_handler is not None:
                signal.signal(signal.SIGALRM, self._old_handler)
        except Exception:
            pass
        # 返回 False 表示不在此处吞异常；由外层统一 try/except 处理并记录日志
        return False

    @staticmethod
    def _raise_timeout(signum, frame):
        raise TimeoutError("Timeout")


def run_generator_with_alarm(code: str, seconds: int = 10, logger=None):
    """
    执行用户提供的 code，查找并调用其中的 generator()。
    任意阶段（编译、exec、调用、超时）一旦出错：logger.error(...) 并返回 None，不向外抛异常。
    """
    g = {}

    # 1) 编译
    try:
        compiled = compile(code, "<user_code>", "exec")
    except Exception as e:
        if logger:
            logger.error(f"[compile] 用户代码编译失败: {e}", exc_info=True)
        return None

    # 2) exec 运行用户代码
    try:
        # 用同一个 dict 作为 globals/locals，确保函数能访问到同一命名空间里的变量
        exec(compiled, g, g)
    except Exception as e:
        if logger:
            logger.error(f"[exec] 执行用户代码时出错: {e}", exc_info=True)
        return None

    # 3) 检查 generator 是否存在且可调用
    gen = g.get("generator", None)
    if not callable(gen):
        if logger:
            logger.error("generator() 未找到或不可调用。")
        return None

    # 4) 带超时地调用 generator()
    try:
        with _Timeout(seconds):
            return gen()
    except TimeoutError:
        if logger:
            logger.error(f"[timeout] generator() 超过 {seconds} 秒未完成。")
        return None
    except Exception as e:
        if logger:
            logger.error(f"[run] 调用 generator() 时出错: {e}", exc_info=True)
        return None
    finally:
        # 双保险：确保闹钟关闭
        try:
            signal.alarm(0)
        except Exception:
            pass

# pip install requests  (如未安装)
import requests

def sandboxfusion_run(base_url: str,
                      code_str: str,
                      *,
                      language: str = "python",
                      stdin: str = "",
                      args=None,
                      time_limit: int = 10,
                      memory_limit_mb: int = 4096,
                      logger=None) -> dict:
    """
    调用 SandboxFusion 执行代码（不抛异常）。
    - base_url: 例如 "http://127.0.0.1:8080"
    - code_str: 待执行代码的完整字符串
    - language: 语言标识（示例: "python", "cpp", "js"...）
    - stdin: 运行时标准输入
    - args: 传给程序的命令行参数列表
    - time_limit: 运行时间限制（秒）
    - memory_limit_mb: 内存上限（MB）
    - token: 若服务启用鉴权，可传 Bearer Token
    - logger: 可选 logger，对错误进行 logger.error
    - endpoint: 具体执行 API 路径，默认 “/api/v1/run”
    - extra_headers: 附加自定义 header

    返回标准化结果字典：
    {
      "ok": True/False,
      "exit_code": int | None,
      "stdout": str,
      "stderr": str,
      "time_ms": int | None,
      "status": int | None,   # HTTP 状态码(出错时)
      "error": str | None,    # 出错信息
      "meta": dict            # 其余原始字段
    }
    """
    url = base_url
    payload = {
        "language": language,
        "code": code_str,
        "stdin": stdin,
        "args": args or [],
        "limits": {"time": time_limit, "memory_mb": memory_limit_mb},
        # 如果 SandboxFusion 需要挂载文件，可扩展：
        # "files": [{"path": "main.py", "content": code_str}]
    }

    headers = {"Content-Type": "application/json"}

    # 1) HTTP 请求阶段
    try:
        resp = requests.post(url, json=payload,timeout=time_limit + 5)
        status = resp.status_code
    except Exception as e:
        msg = f"[SandboxFusion] HTTP request failed: {e}"
        if logger:
            logger.error(msg)
        return {"ok": False, "error": str(e), "stage": "http", "status": None,
                "exit_code": None, "stdout": "", "stderr": "" , "time_ms": None, "meta": {}}

    # 2) 解析响应
    try:
        data = resp.json()
    except Exception:
        data = {"raw": resp.text}
    run_result = data.get("run_result",{})
    # 3) 非 200 直接判失败
    if status != 200:
        err_msg = (run_result.get("status") if isinstance(data, dict) else resp.text) or f"HTTP {status}"
        if logger:
            logger.error(f"[SandboxFusion] status={status} error={err_msg}")
        return {"ok": False,**data}

    # 4) 标准化成功/失败
    
    exit_code = run_result.get("return_code", -1)
    ok = bool(data.get("ok", exit_code == 0))
    result = {
        "ok": ok,
        **data
    }

    if not result["ok"] and logger:
        logger.error(f"[SandboxFusion] run failed: exit_code={exit_code} ")

    return result
