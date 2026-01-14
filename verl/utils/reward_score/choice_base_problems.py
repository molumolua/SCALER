import re
from math_verify.errors import TimeoutException
from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
try:
    from math_verify.errors import TimeoutException
    from math_verify.metric import math_metric
    from math_verify.parser import StringExtractionConfig
    _HAS_MV = True
except ImportError:
    TimeoutException = Exception  # 占位，保证代码可运行
    _HAS_MV = False

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

def _normalize_choice(s: str) -> str | None:
    """
    将模型输出/标准答案归一化为单个选项字母：
    - 忽略外层括号与空白，例如 "(A)" -> "a"
    - 忽略大小写，例如 "a" -> "a"
    - 优先提取形如 \boxed{...} 内的内容
    - 在文本中找“独立单字母token”（如 '答案是 A' -> 'a'）
    找不到则返回 None
    """
    if s is None:
        return None
    s = s.strip()

    # 1) 处理 \boxed{...}
    m = re.search(r'\\boxed\{([^}]*)\}', s)
    if m:
        inner = m.group(1).strip()
        # 如果 \boxed{...} 是单字母，直接取之
        if re.fullmatch(r'[A-Za-z]', inner):
            return inner.casefold()
        # 否则继续在 inner 里找单字母
        s = inner
    # else:
    #     return None

    # 2) 直接是被括号包裹的单字母，如 "(A)"、"[a]"、"{A}"
    m = re.fullmatch(r'[\s\(\[\{\<]*([A-Za-z])[\s\)\]\}\>]*', s)
    if m:
        return m.group(1).casefold()

    # 3) 在文本中寻找“独立的单字母token”，取最后一个更稳（很多模型末尾给结论）
    letters = re.findall(r'\b([A-Za-z])\b', s)
    if letters:
        if len(letters)>1:
            return None
        return letters[-1].casefold()

    return None


def compute_score(solution_str: str, ground_truth: str) -> float:
    """
    字符串通道评分：
    1) 先做选项字母的“括号无关、大小写无关”匹配；
    2) 不匹配则回退到 Math-Verify 的 StringExtractionConfig（若可用）。
    返回分数：匹配=1.0，不匹配=0.0
    """
    # 步骤1：大小写不敏感 + 忽略括号
    
    acc = 0
    pred_norm = None
    format_verify = 0.0

    answer_str = solution_str
    #format_verify,answer_str = format_verify_and_extract(solution_str)
    gt_norm = _normalize_choice(ground_truth)
    pred_norm = _normalize_choice(answer_str)
    
    if gt_norm is not None and pred_norm is not None:
        if gt_norm == pred_norm:
            acc = 1
        else:
            print(f"Not equal gt:{gt_norm}, pred:{pred_norm}")
    else:
        # 步骤2：回退到 Math-Verify 的字符串通道
        if _HAS_MV:
            verify_func = math_metric(
                gold_extraction_target=(StringExtractionConfig(),),
                pred_extraction_target=(StringExtractionConfig(),),
            )
            try:
                acc, _ = verify_func([ground_truth], [answer_str])

            except Exception as e:
                print(e)
            except TimeoutException:
                print("TimeoutException in math-verify.")
        else:
            print("Not math-verify,skip.")

        

    reward = 1.0 if acc else -1.0
    
    return {
        "score": reward,
        "acc": acc,
        "answer": answer_str,
        "pred": str(pred_norm),
        "format_verify": format_verify
    }