from .deepmath_util import extract_answer, math_equal
from math_verify import verify, parse
from typing import Union


def reward_func(
    data_source,
    solution_str: Union[bool, float, str],
    ground_truth: Union[float, str],
    extra_info,
    include_percentage: bool = True,
    tolerance: float = 1e-4,
    math_equal_timeout: float = 10.0,
    math_equal_check_antlr_version: bool = False,
    float_rounding: int = 6,
    numeric_precision: int = 15,
    strict: bool = True,
    verify_timeout_seconds: int = 3,
) -> dict:
    """
    同步版 reward_func，不再使用 Ray，也不依赖 <think></think> 标签。

    返回：
        {
            "score": 1.0/-1.0,
            "acc": bool,
            "pred": 提取出的答案（omi_pred）,
            "omi_correct": bool,
            "mathv_correct": bool
        }
    """
    # 不再区分 think / solution，直接使用完整的 solution_str
    text = str(solution_str)

    omi_pred = None
    omi_correct = False
    mathv_pred = None
    mathv_correct = False

    # OMI 答案检查
    try:
        # 直接从完整输出中提取 boxed 答案
        omi_pred = extract_answer(text, extract_from_boxed=True)
        omi_correct = math_equal(
            omi_pred,
            ground_truth,
            include_percentage,
            tolerance,
            math_equal_timeout,
            math_equal_check_antlr_version,
        )
    except Exception:
        omi_correct = False

    # math_verify 检查
    try:
        mathv_pred = parse(text)
        gold_parsed = parse(f"\\boxed{{${ground_truth}$}}")
        mathv_correct = verify(
            gold_parsed,
            mathv_pred,
            float_rounding=float_rounding,
            numeric_precision=numeric_precision,
            strict=strict,
            timeout_seconds=verify_timeout_seconds,
        )
    except Exception:
        mathv_correct = False

    acc = omi_correct or mathv_correct
    score = 1.0 if acc else -1.0

    return {
        "score": score,
        "acc": acc,
        "pred": omi_pred,
        "omi_correct": omi_correct,
        "mathv_correct": mathv_correct,
    }
