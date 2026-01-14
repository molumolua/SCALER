import re
from math_verify.errors import TimeoutException
from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

# def format_verify_and_extract(solution_str: str) -> tuple[float, str]:
#     """
#     支持两种格式：
#     1) <think>xxx</think><answer>yy</answer>
#     2) <think>xxx</think>yy

#     约束：
#       - 必须以 <think> 开头；
#       - 若存在 <answer> 标签，则 </think> 和 <answer> 之间只能有空白字符；
#       - 若存在 </answer>，则它必须是最后一个字符。
#     """
#     pattern = r"(?s)^<think>(.*?)</think>\s*(?:<answer>(.*?)</answer>|(.*\S.*))$"
#     m = re.match(pattern, solution_str)
#     if not m:
#         return 0.0, ""

#     # 有 <answer> 标签就用 group(2)，否则用 </think> 后面的内容 group(3)
#     answer = (m.group(2) if m.group(2) is not None else m.group(3)).strip()
#     return 1.0, answer
def format_verify_and_extract(solution_str: str) -> tuple[float, str]: 
    """ 要求： 1. 以 <think> 开头； 
    2. </think> 和 <answer> 之间只能有空白字符（或直接相连）；
    3. </answer> 必须是最后一个字符； 4. 不再强制任何换行或其它空白。 
    """ 
    pattern = r"(?s)^<think>(.*?)</think>\s*<answer>(.*?)</answer>$" 
    m = re.match(pattern,solution_str) 
    if not m: 
        return 0.0, "" 
    # m.group(1) <think>…</think> 之间的内容 
    # m.group(2) <answer>…</answer> 之间的内容 
    answer = m.group(2).strip() 
    return 1.0, answer
def compute_score(solution_str, ground_truth):
    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )
    acc = 0
    pred = ""
    format_verify = 0.0
    try:
        answer_str = solution_str
        # format_verify,answer_str = format_verify_and_extract(solution_str)
        acc,_=verify_func([ground_truth], [answer_str])
    except Exception as e:
        print(e)
    except TimeoutException:
        print("TimeoutException in math-verify.")

    reward = 1.0 if acc else -1.0

    return {
        "score": reward,
        "acc": acc,
        "answer": answer_str,
        "pred": str(pred),
        "format_verify": format_verify
    }