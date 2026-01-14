try:
    from process_dataset import load_and_prepare_dataset,prepare_examples
    from extract import extract_last_code_block,parse_gen_script,parse_one_gen_script,get_function_code_from_str
    from logger import setup_logger
    from exec_and_verify import build_and_run_reference_solution,write_and_build_referenece_solution,run_reference_solution,import_needed_module_for_python,fix_newlines_in_python_strings,run_generator_with_alarm,sandboxfusion_run
except:
    from SCALER.process_dataset import load_and_prepare_dataset,prepare_examples
    from SCALER.extract import extract_last_code_block,parse_gen_script,parse_one_gen_script,get_function_code_from_str
    from SCALER.logger import setup_logger
    from SCALER.exec_and_verify import build_and_run_reference_solution,write_and_build_referenece_solution,run_reference_solution,import_needed_module_for_python,fix_newlines_in_python_strings,run_generator_with_alarm,sandboxfusion_run

import copy
from typing import Tuple, Optional, List, Dict, Any
import inspect
import json

from tqdm import tqdm
import json
from pathlib import Path
from collections import defaultdict



import json

import json
import ast
from typing import Optional, Tuple

def parse_testcase(testcase_string: str, logger=None, debug: bool = False) -> Tuple[Optional[str], Optional[dict]]:
    try:
        raw = testcase_string


        # 去掉首尾空白
        s = raw.strip()

        # 兼容外层有括号的 tuple 写法：(..., {...})
        if s.startswith("(") and s.endswith(")"):
            s = s[1:-1].strip()

        # 用第一个逗号分割：前面是 string 部分，后面是 dict 部分
        parts = s.split(",", 1)
        if len(parts) != 2:
            if logger is not None:
                logger.error(
                    "[parse_testcase] split by first comma failed, expected 2 parts, got %d, s=%r",
                    len(parts), s
                )
            return None, None

        string_part_raw = parts[0].strip()
        dict_part_str = parts[1].strip()

        # ---- 解析 string 部分 ----
        string_part = string_part_raw
        # 如果是带引号的 Python 字面量，尝试用 literal_eval 还原
        if (len(string_part_raw) >= 2 and
            string_part_raw[0] in ("'", '"') and
            string_part_raw[-1] == string_part_raw[0]):
            try:
                val = ast.literal_eval(string_part_raw)
                if isinstance(val, str):
                    string_part = val
            except Exception:
                if logger is not None :
                    logger.info("[parse_testcase] ast.literal_eval on string_part failed, use raw")


        # ---- 解析 dict 部分 ----
        dict_part = None

        # 先做一个 JSON 化的 candidate：把单引号替换成双引号
        dict_json_candidate = dict_part_str.replace("'", '"')

        # 优先尝试 json.loads
        try:
            dict_part = json.loads(dict_json_candidate)
        except Exception as e1:
            if logger is not None:
                logger.error(
                    "[parse_testcase] json.loads failed, candidate=%r, error=%r",
                    dict_json_candidate, e1
                )
            # fallback: 尝试按 Python 字面量解析
            try:
                tmp = ast.literal_eval(dict_part_str)
                if isinstance(tmp, dict):
                    dict_part = tmp
                else:
                    if logger is not None:
                        logger.error(
                            "[parse_testcase] ast.literal_eval result is not dict, type=%s, value=%r",
                            type(tmp).__name__, tmp
                        )
                    return None, None
            except Exception as e2:
                if logger is not None:
                    logger.error(
                        "[parse_testcase] ast.literal_eval failed on dict_part_str=%r, error=%r",
                        dict_part_str, e2
                    )
                return None, None

        if not isinstance(dict_part, dict):
            if logger is not None:
                logger.error(
                    "[parse_testcase] parsed dict_part is not dict, type=%s, value=%r",
                    type(dict_part).__name__, dict_part
                )
            return None, None

 
        return string_part, dict_part

    except Exception as e:
        if logger is not None:
            logger.error(
                "[parse_testcase] unexpected exception, error=%r, input=%r",
                e, testcase_string
            )
        return None, None



def exec_and_return_values(code_str: str, var_names: List[str], logger) -> Optional[Dict[str, Any]]:
    """
    在隔离的命名空间中执行代码字符串，并返回指定变量的值。
    如果执行失败或变量缺失，返回 None 并记录日志。
    """
    _logger = logger
    try:
        ns: Dict[str, Any] = {}
        exec(code_str, {}, ns)

        result = {}
        for var in var_names:
            if var not in ns:
                _logger.error("Missing required variable: %s", var)
                return None
            result[var] = ns[var]
        return result

    except Exception as e:
        _logger.exception("exec_and_return_values failed: %s", e)
        return None
    
        
import re

def is_valid_number(s: str) -> bool:
    # 使用正则表达式检查是否是有效的整数或浮点数
    return bool(re.fullmatch(r'^\s*-?\d+(\.\d+)?\s*$', s.strip()))

def filter_only_numerical_problem(correct_test_cases):
    numerical_flag = True
    for test_case in correct_test_cases:
        if not is_valid_number(test_case['output']):
            numerical_flag = False
    return numerical_flag
            

      
def verify_meta_json(examples, logger=None, debug=False):
    """
    检查每个 example['answer'] 是否是符合“问题元信息”约定结构的 JSON 代码块。
    约定结构：
      顶层 JSON 必须是一个 object，且至少包含字段：
        - "scale_params": dict
            - 每个值是形如 {"min": int, "max": int} 的对象（int 且不是 bool）
        - "output_type": 字符串，∈ {"string","number","array","graph","matrix","bool","others"}

    其它要求：
      - 有且只有一个 JSON code block（我们取最后一个）
      - code block 标记语言为 `json`
      - 能被 json.loads
      - 顶层是 dict

    返回 (valid_examples, invalid_examples) 两个列表。
    """
    valid_examples = []
    invalid_examples = []

    allowed_output_types = {"string", "number", "array", "graph", "matrix", "bool", "others"}

    for example in examples:
        ok = True

        # 1. 提取最后一个 code block
        try:
            code, lang = extract_last_code_block(example['answer'])
        except Exception as e:
            if logger and debug:
                logger.error(f"Failed to extract code block: {e}")
            invalid_examples.append(example)
            continue

        # 2. 必须是 json 代码块
        if lang != "json":
            if logger and debug:
                logger.error("Not json code block.")
            invalid_examples.append(example)
            continue

        # 3. 能否 parse 成 JSON
        try:
            obj = json.loads(code)
        except json.JSONDecodeError as e:
            if logger and debug:
                logger.error(f"JSON decode error: {e}")
            invalid_examples.append(example)
            continue

        # 4. 顶层必须是 dict
        if not isinstance(obj, dict):
            if logger and debug:
                logger.error("Top-level JSON is not an object.")
            invalid_examples.append(example)
            continue

        # 5. 检查必需字段是否存在
        required_keys = {"scale_params", "output_type"}
        if not required_keys.issubset(obj.keys()):
            ok = False
            if logger and debug:
                missing = required_keys - set(obj.keys())
                logger.error(f"Missing required top-level keys: {missing}")
        if not ok:
            invalid_examples.append(example)
            continue

        # 6. 检查 scale_params
        scale_params = obj.get("scale_params")
        if not isinstance(scale_params, dict):
            ok = False
            if logger and debug:
                logger.error('"scale_params" must be an object.')
        else:
            # 可以为空 dict
            for key, val in scale_params.items():
                if not isinstance(val, dict):
                    ok = False
                    if logger and debug:
                        logger.error(f'Value for scale_param "{key}" is not an object.')
                    break

                keys = set(val.keys())
                if keys != {"min", "max"}:
                    ok = False
                    if logger and debug:
                        logger.error(
                            f'Scale param "{key}" must have exactly "min" and "max", got {keys}.'
                        )
                    break

                for bound_name in ("min", "max"):
                    v = val[bound_name]
                    # int 且不是 bool（因为 bool 是 int 的子类）
                    if not isinstance(v, int) or isinstance(v, bool):
                        ok = False
                        if logger and debug:
                            logger.error(
                                f'Value "{bound_name}" for scale_param "{key}" '
                                f"must be an integer, got {type(v)}."
                            )
                        break
                if not ok:
                    break

        if not ok:
            invalid_examples.append(example)
            continue

        # 7. 检查 output_type
        output_type = obj.get("output_type")
        if not isinstance(output_type, str):
            ok = False
            if logger and debug:
                logger.error('"output_type" must be a string.')
        elif output_type not in allowed_output_types:
            ok = False
            if logger and debug:
                logger.error(
                    f'"output_type" must be one of {allowed_output_types}, got "{output_type}".'
                )

        if not ok:
            invalid_examples.append(example)
            continue
        
        # 7. 检查 is_output_unique
        is_output_unique = obj.get("is_output_unique")
        if not isinstance(is_output_unique, bool):
            ok = False
            if logger and debug:
                logger.error('"is_output_unique" must be boolean.')
                
        if not ok:
            invalid_examples.append(example)
            continue


        example["parsed_json"] = code
        valid_examples.append(example)

    return valid_examples,invalid_examples

import random


def new_random_get_json_object(json_obj_form,max_number = 10000,min_number = 0):
    if isinstance(json_obj_form,str):
        json_obj_form = json.loads(json_obj_form)
        
    example_json_obj = dict()
    for k,v in json_obj_form['scale_params'].items():
        v_min = max(min_number,int(v["min"]))
        v_max = min(max_number,int(v["max"]))
        
        v_min = min(v_min,v_max)
        v_max = max(v_min,v_max)
        
        v_true = random.randint(v_min,v_max)
        example_json_obj[k] = v_true
    
    return example_json_obj

def new_get_json_object(json_obj_form,get_type="min"):
    if isinstance(json_obj_form,str):
        json_obj_form = json.loads(json_obj_form)
        
    example_json_obj = dict()
    for k,v in json_obj_form['scale_params'].items():
        if get_type=="min":
            example_json_obj[k] = int(v["min"])
        elif get_type =="max":
            example_json_obj[k] = int(v["max"])
    
    return example_json_obj


def to_float_or_none(x):
    """把任意对象转成 float，失败返回 None。"""
    try:
        return float(str(x).strip())
    except (ValueError, TypeError):
        return None


def str_to_float_list_or_none(x):
    """
    把字符串按空白字符分割成数字列表，失败返回 None。
    例如："1  2\t3.5\n4" -> [1.0, 2.0, 3.5, 4.0]
    """
    tokens = str(x).split()
    if not tokens:
        return []
    arr = []
    for t in tokens:
        try:
            arr.append(float(t))
        except ValueError:
            return None
    return arr

def complie_program_output(output_str,output_type):
    if output_type=="number":
        return to_float_or_none(output_str)
    elif output_type=="array":
        return str_to_float_list_or_none(output_str)
    elif output_type=="string":
        now_s = str(output_str)
        now_tokens = now_s.split()
        if len(now_tokens) != 1:
            return None
        else:
            return now_tokens[0]
    else:
        return None


def _exec_generator_for_environment(code,cpp_py_solution_list,output_type,test_obj,sandboxfusion_url,logger,debug):
    sandbox_code = code+f'''
if __name__ == "__main__":
    print(generate_testcase({test_obj}))
'''
    ret = sandboxfusion_run(sandboxfusion_url, sandbox_code,logger=logger,
                                                language='python',stdin="")
    if ret["ok"]:
        test_case_input,problem_detail = parse_testcase(ret['run_result']["stdout"],logger,debug)
        if test_case_input == None or problem_detail == None:
            logger.error(f" Error in parse test case.")
            return False,None
            
    else:
        if logger and debug:
            logger.error(f"Error in exec generator with SandboxFusion! {ret}")
        return False,None

    # step2：跑多组solution，并获取每一组solution的输出
    compare_to_check_list = []
    for cpp_py_solution_item in cpp_py_solution_list:
        checked_list = run_reference_solution(inputs=[test_case_input],
                                            sol_code=cpp_py_solution_item['sol_code'],
                                            sol_bin = cpp_py_solution_item['sol_bin'],
                                            logger=logger,
                                            debug=debug,
                                            lang=cpp_py_solution_item['language'])
        if checked_list[0]['flag']==False:
            if logger and debug:
                logger.error(f"Error in run reference solution!")
            return False,None
        
        checked_list[0]['output']=complie_program_output(checked_list[0]['output'],output_type)
        if checked_list[0]['output'] == None:
            if logger and debug:
                logger.error(f"Error in complie output!")
            return False,None
        compare_to_check_list.append(checked_list[0]) # {input:xxx,output:xxx,flag:ture/false}
                    
                    
    for k, checked_list in enumerate(compare_to_check_list[:-1]):
        now_submission_output = checked_list['output']
        nxt_submission_output = compare_to_check_list[k + 1]['output']
        eps=1e-7

        if output_type == "number":
            if now_submission_output is None or nxt_submission_output is None:
                logger.error("Output cannot be parsed as number.")
                return False, None
            if abs(now_submission_output - nxt_submission_output) > eps:
                logger.error("Verifying and find two output not equal (number).")
                return False, None
        elif output_type == "array":
            if now_submission_output is None or nxt_submission_output is None:
                logger.error("Output cannot be parsed as numeric array.")
                return False, None
            if len(now_submission_output) != len(nxt_submission_output):
                logger.error("Verifying and find two output not equal (array length).")
                return False, None
            for a, b in zip(now_submission_output, nxt_submission_output):
                if abs(a - b) > eps:
                    logger.error("Verifying and find two output not equal (array element).")
                    return False, None
        elif output_type == "string":
            if now_submission_output is None or nxt_submission_output is None:
                logger.error("Output cannot be parsed as string.")
                return False, None
            if now_submission_output != nxt_submission_output:
                logger.error("Verifying and find two output not equal (string).")
                return False, None
        else:
            logger.error("Unknown output type.")
            return False, None

    return True, compare_to_check_list[0]

def output_diversity_check(compare_to_check_list,different_output_limit,max_output_times,logger,debug):
    output_count = defaultdict(int)

    # 统计每个output的个数
    for checked_list in compare_to_check_list:
        output_count[str(checked_list['output']).rstrip("\n")] += 1

    # 计算族的数量
    num_of_groups = len(output_count)

    # 找到最大族的大小
    max_group_size = max(output_count.values())
    if debug:
        logger.info(f"Num of Groups: {num_of_groups}, Max Group Size: {max_group_size}")
    
    return num_of_groups >= different_output_limit and max_group_size <= max_output_times

    
def verify_and_exec_generator_for_environment_combined(
    examples,
    logger,
    debug=False,
    check_number=3,
    deep_test_times=100,
    breadth_test_times=100,
    sandboxfusion_url=None,
    different_output_limit=10,
    max_output_rate=0.3,
):
    _logger = logger
    success_problems = []
    left_problems = []

    cwd = Path.cwd()
    src_dir = (cwd / ".." / "testlib").resolve()
    bin_dir = (cwd / "testlib").resolve()
    src_dir.mkdir(parents=True, exist_ok=True)
    bin_dir.mkdir(parents=True, exist_ok=True)

    # 外层进度条：遍历所有 problem
    outer_bar = tqdm(
        examples,
        desc="Processing problems",
        unit="problem",
        ncols=100,
        ascii=True,
    )

    for problem_idx, example in enumerate(outer_bar, start=1):
        code, lang = extract_last_code_block(example["answer"])
        todo_flag = True
        code = fix_newlines_in_python_strings(import_needed_module_for_python(code))

        if lang and lang == "python":
            try:
                cpp_py_solution_list = []
                for solution_code, solution_lang in zip(
                    example["solutions"]["solution"], example["solutions"]["language"]
                ):
                    if solution_lang == 2:
                        cpp_py_solution_list.append(
                            {"code": solution_code, "language": "cpp"}
                        )
                    elif solution_lang == 3:
                        cpp_py_solution_list.append(
                            {"code": solution_code, "language": "python"}
                        )
                    if len(cpp_py_solution_list) >= check_number:
                        break

                if len(cpp_py_solution_list) < check_number:
                    # correct_submissions 不够，不用加入 todo
                    if logger and debug:
                        logger.error(
                            "Not enough submission to check test case correctness."
                        )
                    continue

                # 编译 / 写入参考解（不经过 sandbox）
                error_write_and_build_flag = False
                for idx, cpp_py_solution_item in enumerate(cpp_py_solution_list):
                    solution_code = cpp_py_solution_item["code"]
                    sol_lang = cpp_py_solution_item["language"]
                    if sol_lang == "cpp":
                        sol_code = src_dir / f"solution_{idx}.cpp"
                        sol_bin = bin_dir / f"sol_{idx}"
                    elif sol_lang == "python":
                        sol_code = bin_dir / f"sol_{idx}.py"
                        sol_bin = None

                    success_flag = write_and_build_referenece_solution(
                        solution_code_str=solution_code,
                        lang=sol_lang,
                        sol_code=sol_code,
                        sol_bin=sol_bin,
                        debug=debug,
                        logger=logger,
                    )

                    cpp_py_solution_item["sol_code"] = sol_code
                    cpp_py_solution_item["sol_bin"] = sol_bin

                    if not success_flag:
                        error_write_and_build_flag = True
                        break

                if error_write_and_build_flag:
                    if logger and debug:
                        logger.error("Error during write_and_build.")
                    continue

                error_cnt = 0
                json_obj_form = json.loads(example["parsed_json"])
                output_type = json_obj_form["output_type"]

                # ============ 1. 广度测试 ============
                if todo_flag and breadth_test_times > 0:
                    if debug:
                        breadth_bar = tqdm(
                            range(breadth_test_times),
                            desc=f"[{problem_idx}] Breadth",
                            unit="run",
                            leave=False,          # 结束后自动清掉这条进度条
                            ncols=80,
                            ascii=True,
                            position=1,           # 放在外层进度条下面一行
                        )
                    else:
                        breadth_bar = range(breadth_test_times)

                    for _ in breadth_bar:
                        if debug:
                            breadth_bar.set_postfix(errors=error_cnt, refresh=True)

                        test_obj = new_random_get_json_object(json_obj_form,max_number=1000)
                        success_flag, _ = _exec_generator_for_environment(
                            code=code,
                            cpp_py_solution_list=cpp_py_solution_list,
                            output_type=output_type,
                            test_obj=test_obj,
                            sandboxfusion_url=sandboxfusion_url,
                            logger=logger,
                            debug=debug,
                        )
                        if not success_flag:
                            error_cnt += 1
                            break

                    if error_cnt:
                        todo_flag = False
                        if logger and debug:
                            logger.error("Error in Breadth Test.")

                # ============ 2. 深度测试（同一 test_obj，多次生成看多样性） ============
                if todo_flag and deep_test_times > 0:
                    test_obj = new_random_get_json_object(json_obj_form, min_number=100)
                    checked_output_list = []

                    if debug:
                        deep_bar = tqdm(
                            range(deep_test_times),
                            desc=f"[{problem_idx}] Deep",
                            unit="run",
                            leave=False,
                            ncols=80,
                            ascii=True,
                            position=1,
                        )
                    else:
                        deep_bar = range(deep_test_times)

                    for _ in deep_bar:
                        if debug:
                            deep_bar.set_postfix(errors=error_cnt, refresh=True)

                        success_flag, checked_output = _exec_generator_for_environment(
                            code=code,
                            cpp_py_solution_list=cpp_py_solution_list[:1],
                            output_type=output_type,
                            test_obj=test_obj,
                            sandboxfusion_url=sandboxfusion_url,
                            logger=logger,
                            debug=debug,
                        )
                        if not success_flag:
                            error_cnt += 1
                            break
                        checked_output_list.append(checked_output)

                    if error_cnt or (
                        not output_diversity_check(
                            checked_output_list,
                            different_output_limit=different_output_limit,
                            max_output_times=max_output_rate * deep_test_times,
                            logger=logger,
                            debug=debug,
                        )
                    ):
                        todo_flag = False
                        if logger and debug:
                            logger.error("Error in Deep Test.")

                # ============ 3. 压力测试：min ============
                if todo_flag:
                    test_obj = new_get_json_object(json_obj_form, get_type="min")
                    success_flag, _ = _exec_generator_for_environment(
                        code=code,
                        cpp_py_solution_list=cpp_py_solution_list,
                        output_type=output_type,
                        test_obj=test_obj,
                        sandboxfusion_url=sandboxfusion_url,
                        logger=logger,
                        debug=debug,
                    )
                    if not success_flag:
                        todo_flag = False
                        if logger and debug:
                            logger.error("Error in Pressure Test in min.")

                # # ============ 4. 压力测试：max ============
                # if todo_flag:
                #     test_obj = new_get_json_object(json_obj_form, get_type="max")
                #     success_flag, _ = _exec_generator_for_environment(
                #         code=code,
                #         cpp_py_solution_list=cpp_py_solution_list,
                #         output_type=output_type,
                #         test_obj=test_obj,
                #         sandboxfusion_url=sandboxfusion_url,
                #         logger=logger,
                #         debug=debug,
                #     )
                #     if not success_flag:
                #         todo_flag = False
                #         if logger and debug:
                #             logger.error("Error in Pressure Test in max.")

            except Exception as e:
                _logger.error(f"Error in loading Python Function Generator: {e}")
        else:
            _logger.error("No Python code.")

        if todo_flag:
            success_problems.append({**example, "generate_testcase": code})
        else:
            left_problems.append(example)

    return success_problems, left_problems


def find_max_difficulty(examples, logger, debug=False, sandboxfusion_url=None, max_prompt_length=2048):
    _logger = logger
    success_problems = []
    cwd = Path.cwd()
    src_dir = (cwd / ".." / "testlib").resolve()
    bin_dir = (cwd / "testlib").resolve()
    src_dir.mkdir(parents=True, exist_ok=True)
    bin_dir.mkdir(parents=True, exist_ok=True)

    for example in tqdm(examples, desc="Processing test cases", unit="problem", ncols=100, ascii=True):
        code, lang = extract_last_code_block(example['answer'])
        code = fix_newlines_in_python_strings(import_needed_module_for_python(code))
        todo_flag = True
        if lang and lang == "python":
            try:
                # 二分查找测试输入的最大长度

                
                json_object = json.loads(example['parsed_json'])
                
                min_range = 10000
                for k,v in json_object['scale_params'].items():
                    min_range = min(min_range,v['max']-v['min'])
                
                if min_range <= 0:
                    logger.error("v_max == v_min,skip.")
                    continue
                
                base_list = [ (min(v['max']-v['min'],10000))/min_range for k,v in json_object['scale_params'].items()]
                low, high = 0, min_range 

                while low <= high:
                    mid = (low + high) // 2
                    
                    # 直接将mid作为v_max代入
                    test_obj = {k: v['min']+int(base*mid) for (k,v),base in zip(json_object['scale_params'].items(),base_list)}

                    sandbox_code = code + f'''
if __name__ == "__main__":
    print(generate_testcase({test_obj} ))
'''
                    ret = sandboxfusion_run(sandboxfusion_url, sandbox_code, logger=logger,
                                             language='python', stdin="")
                    # print(ret)
                    if ret["ok"]:
                        test_case_input,problem_detail = parse_testcase(ret['run_result']["stdout"],logger,debug)
                        if test_case_input == None or problem_detail == None:
                            logger.error(f" Error in parse test case.")
                            logger.error(ret['run_result']["stderr"])
                            logger.error(len(ret['run_result']["stdout"]))
                            todo_flag = False
                            break

                        # 如果输入的长度小于等于最大长度，尝试增大v_max
                        if len(test_case_input) <= max_prompt_length:
                            low = mid + 1
                        else:
                            # 否则减小v_max
                            high = mid - 1
                    else:
                        # print()
                        logger.error(ret['run_result']["stderr"])
                        logger.error(len(ret['run_result']["stdout"]))
                        todo_flag = False
                        break
                    
                if todo_flag:
                    # 最终的最大v_max
                    final_range = high
                    logger.info(f"Max range found: {final_range}")
                    logger.info(test_case_input)
                    if final_range <= 0:
                        logger.error("MAX v_max == v_min,skip.")
                        continue
                    
                    new_json_object = {
                        k: {
                            "min":v['min'] ,
                            "max":v['min']+int(base*final_range),
                            "base":base
                        } for (k,v),base in zip(json_object['scale_params'].items(),base_list)
                    }
                    print(new_json_object)
                    example['parsed_json'] = str(new_json_object)
                    example['scale_range'] = final_range
                    example['output_type']=json_object['output_type']
                    example['is_output_unique']=json_object['is_output_unique']
                    
                    success_problems.append(example)
            except Exception as e:
                _logger.error(f"Error in loading Python Function Generator: {e}")
        else:
            _logger.error("No Python code.")

    return success_problems


def generate_problem_detail_and_ground_truth(example,problem_scale,sandboxfusion_url,logger=None,max_try = 3):
    problem_detail=""
    ground_truth = ""
    test_case_input= ""
    code = fix_newlines_in_python_strings(import_needed_module_for_python(example['generate_testcase']))
    for _ in range(max_try):
        error_flag = False
        success_flag = False
        problem_detail_sandbox_code = code + f'''
if __name__ == "__main__":
    print(generate_testcase({problem_scale} ))
'''
        ret = sandboxfusion_run(sandboxfusion_url, problem_detail_sandbox_code,logger=logger,
                                                language='python', stdin="")
        
        if ret["ok"]:
            test_case_input,problem_detail = parse_testcase(ret['run_result']["stdout"])
            if test_case_input == None or problem_detail == None:
                error_flag = True 
        else:
            error_flag = True 

        if error_flag:
            continue
        
        for solution_code,lang in zip(example['solutions']['solution'],example['solutions']['language']):
            if lang == 2:
                ret = sandboxfusion_run(sandboxfusion_url, solution_code, logger=logger,
                                        language='cpp', stdin=test_case_input)
            elif lang == 3:
                ret = sandboxfusion_run(sandboxfusion_url, solution_code,logger=logger,
                                        language='python', stdin=test_case_input)
            if ret['ok']:
                success_flag = True
                ground_truth = ret['run_result']["stdout"]
                break
        if (not error_flag) and success_flag:
            break

    return str(problem_detail),ground_truth
        
if __name__ == "__main__":
    logger = setup_logger()
    examples=  load_and_prepare_dataset("./Code/FORGE",load_type="json",logger=logger,split="Train",file_glob="with_generator_after_filter.jsonl")