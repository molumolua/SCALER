import json
try:
    from .deepmath_util import extract_answer
except:
    from verl.utils.reward_score.deepmath_util import extract_answer
# 如果你真的有 TimeoutException，就在这里导入
# from some_module import TimeoutException 

def to_float_or_none(x):
    """把任意对象转成 float，失败返回 None。"""
    try:
        return float(str(x).strip())
    except (ValueError, TypeError):
        return None



def compute_score(solution_str, ground_truth, output_type):
    acc = 0
    pred = ""
    answer_str = ""
    format_verify = 0.0
    eps = 1e-7

    try:
        answer_str = extract_answer(solution_str)
        ground_truth_str = extract_answer(ground_truth)
        # 假设 pred 就是模型抽取出的最终答案
        pred = answer_str

        if output_type == "number":
            answer_val = to_float_or_none(answer_str)
            ground_truth_val = to_float_or_none(ground_truth_str)
            # 考虑浮点误差
            if (
                answer_val is not None
                and ground_truth_val is not None
                and abs(answer_val - ground_truth_val) <= eps
            ):
                acc = 1.0

        elif output_type == "array":
            try:
                answer_val = json.loads(answer_str)
                ground_truth_val = json.loads(ground_truth_str)
            except Exception as e:
                print("JSON 解析失败:", e)
            else:
                # 必须都是 list/tuple
                if isinstance(answer_val, (list, tuple)) and isinstance(
                    ground_truth_val, (list, tuple)
                ):
                    # 长度必须一致
                    if len(answer_val) == len(ground_truth_val):
                        acc_flag = 1
                        for a, b in zip(answer_val, ground_truth_val):
                            a_f = to_float_or_none(a)
                            b_f = to_float_or_none(b)
                            # 有一个转不成数字就直接错
                            if (
                                a_f is None
                                or b_f is None
                                or abs(a_f - b_f) > eps
                            ):
                                acc_flag = 0
                                break
                        acc = acc_flag
                    else:
                        acc = 0
                else:
                    # 不是数组，直接错
                    acc = 0

        elif output_type == "string":
            answer_val = str(answer_str)
            ground_truth_val = str(ground_truth_str)

            answer_vals = answer_val.split()
            ground_truth_vals = ground_truth_val.split()

            if (
                len(answer_vals) == 1
                and len(ground_truth_vals) == 1
                and answer_vals[0] == ground_truth_vals[0]
            ):
                acc = 1

        else:
            # 未知的 output_type，按错误处理，或者 raise 也行
            print(f"Unknown output_type: {output_type}")

    except Exception as e:
        print(e)

    reward = 1.0 if acc else -1.0

    return {
        "score": reward,
        "acc": acc,
        "answer": answer_str,
        "pred": str(pred),
        "format_verify": format_verify,
    }
def test_compute_score():
    """
    针对 compute_score 的一组测试用例。
    把这个函数放在包含 compute_score 的同一模块中，直接运行 test_compute_score()。
    """
    cases = [
        # (描述, solution_str, ground_truth, output_type, 期望 acc)
        ("number - 完全匹配（整数）",
         "\\boxed{3\n}", "\\boxed{3\n}", "number", 1),

        ("number - 小数匹配（相同）",
         "\\boxed{3.14\n\n}", "\\boxed{3.14}", "number", 1),

        ("number - 不相等",
         r"\boxed{2.999}", r"\boxed{3.0}", "number", 0),

        ("string - 完全相同单词",
         r"\boxed{hello}", r"\boxed{hello}", "string", 1),

        ("string - 不同单词",
         r"\boxed{hello}", r"\boxed{world}", "string", 0),

        ("array - JSON 格式方括号（标准情况）",
         r"\boxed{[1,2,3]}", r"\boxed{[1,2,3]}", "array", 1),

        ("array - JSON 有空格（标准情况）",
         r"\boxed{[1, 2, 3]}", r"\boxed{[1,2,3]}", "array", 1),

        ("array - 长度不一致（应为错）",
         r"\boxed{[1,2]}", r"\boxed{[1,2,3]}", "array", 0),

        ("array - 元素类型无法直接相减（字符串元素）",
         r"\boxed{[\"1\",\"2\",\"3\"]}", r"\boxed{[1,2,3]}", "array", 0),

        ("array - 非 JSON 且以空格分隔（注意：当前实现可能无法解析）",
         r"\boxed{1 2 3}", r"\boxed{[1,2,3]}", "array", 0),

        ("array - 含非法 token（解析失败）",
         r"\boxed{[1,2,foo]}", r"\boxed{[1,2,3]}", "array", 0),
    ]

    total = len(cases)
    passed = 0
    failed_cases = []

    print("开始运行 compute_score 测试用例，共 %d 个\n" % total)
    for i, (desc, sol, gt, typ, expect) in enumerate(cases, 1):
        try:
            res = compute_score(sol, gt, typ)
            acc = res.get("acc", 0)
            ok = (1 == acc) if expect == 1 else (0 == acc)
            status = "PASS" if ok else "FAIL"
            if ok:
                passed += 1
            else:
                failed_cases.append((i, desc, sol, gt, typ, expect, acc, res))
            print(f"[{status}] {i}. {desc}")
            print(f"    ground_truth: {gt}")
            print(f"    输入(solution): {sol}")
            print(f"    期望 acc: {expect}, 实际 acc: {acc}")
            print(f"    返回值: {res}\n")
        except Exception as e:
            failed_cases.append((i, desc, sol, gt, typ, expect, f"EXCEPTION: {e}"))
            print(f"[ERROR] {i}. {desc} 在执行时抛出异常: {e}\n")

    print("测试结束：%d/%d 通过" % (passed, total))
    if failed_cases:
        print("\n失败用例详情（用于调试）：")
        for item in failed_cases:
            print(item)

# 如果你想立即运行测试，请取消下面一行的注释：
# test_compute_score()


if __name__ == "__main__":
    test_compute_score()