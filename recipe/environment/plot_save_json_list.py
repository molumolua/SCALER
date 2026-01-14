import json
import os
from collections import defaultdict, OrderedDict
import matplotlib.pyplot as plt

def plot_avg_response_length_by_problem_distance(json_path: str,
                                                 out_png_path: str,
                                                 sort_desc: bool = True):
    """
    读取 JSON 列表文件，按 problem_name 和 distance 聚类，计算平均 response_length，
    统计 response_length == 16384 和 response_length > 8192 的数量，
    同时计算平均 prompt_length，并将柱状图保存为 PNG。

    参数：
        json_path: JSON 列表文件路径（如 /.../save_json_list.json）
        out_png_path: 输出图片路径（如 /.../save_json_list.png）
        sort_desc: 是否按平均值从高到低排序显示

    返回：
        OrderedDict，键为 (problem_name, distance)，
        值为 {'avg_response_length': float, 'count': int, 'count_eq_16384': int, 'count_gt_8192': int}
        和按 problem_name 聚类的 OrderedDict（同上结构）
    """
    # 1) 读取数据
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 2) 聚类并收集每个 (problem_name, distance) 的 response_length / prompt_length
    groups_distance = defaultdict(list)          # response_length
    groups_problem = defaultdict(list)           # response_length
    groups_distance_prompt = defaultdict(list)   # prompt_length
    groups_problem_prompt = defaultdict(list)    # prompt_length

    for item in data:
        if not isinstance(item, dict):
            continue
        pname = item.get("problem_name")
        distance = item.get("distance")
        rl = item.get("response_length")
        pl = item.get("prompt_length")  # 新增：读取 prompt_length

        if pname is None or distance is None or rl is None:
            continue

        # 解析 response_length
        try:
            rl_val = int(rl)  # 兼容字符串数字
        except (TypeError, ValueError):
            continue

        # 解析 prompt_length（允许缺失）
        pl_val = None
        if pl is not None:
            try:
                pl_val = int(pl)
            except (TypeError, ValueError):
                pl_val = None

        groups_distance[(pname, distance)].append(rl_val)
        groups_problem[pname].append(rl_val)

        if pl_val is not None:
            groups_distance_prompt[(pname, distance)].append(pl_val)
            groups_problem_prompt[pname].append(pl_val)

    if not groups_distance or not groups_problem:
        raise ValueError("未从文件中解析到任何包含 problem_name 和 distance 的条目。")

    # 3) 计算统计量
    stats_distance = []
    stats_problem = []

    for (pname, dist), lst in groups_distance.items():
        avg = sum(lst) / len(lst)
        cnt = len(lst)
        cnt_eq_16384 = sum(1 for v in lst if v == 16384)
        cnt_gt_8192 = sum(1 for v in lst if v >= 8192)

        # 对应的 prompt_length 列表
        pl_list = groups_distance_prompt.get((pname, dist), [])
        avg_prompt = sum(pl_list) / len(pl_list) if pl_list else None

        # 这里把 avg_prompt 一并存进去，方便后面画图用
        stats_distance.append(((pname, dist), avg, cnt, cnt_eq_16384, cnt_gt_8192, avg_prompt))

    for pname, lst in groups_problem.items():
        avg = sum(lst) / len(lst)
        cnt = len(lst)
        cnt_eq_16384 = sum(1 for v in lst if v == 16384)
        cnt_gt_8192 = sum(1 for v in lst if v >= 8192)

        pl_list = groups_problem_prompt.get(pname, [])
        avg_prompt = sum(pl_list) / len(pl_list) if pl_list else None

        stats_problem.append((pname, avg, cnt, cnt_eq_16384, cnt_gt_8192, avg_prompt))

    # 4) 排序（可选）
    stats_distance.sort(key=lambda x: x[1], reverse=True)
    stats_problem.sort(key=lambda x: x[1], reverse=True)

    labels_distance = [f"{s[0][0]} (d={s[0][1]})" for s in stats_distance]
    avgs_distance = [s[1] for s in stats_distance]
    counts_distance = [s[2] for s in stats_distance]
    counts_eq_16384_distance = [s[3] for s in stats_distance]
    counts_gt_8192_distance = [s[4] for s in stats_distance]
    prompt_avgs_distance = [s[5] for s in stats_distance]  # 新增：平均 prompt_length

    labels_problem = [s[0] for s in stats_problem]
    avgs_problem = [s[1] for s in stats_problem]
    counts_problem = [s[2] for s in stats_problem]
    counts_eq_16384_problem = [s[3] for s in stats_problem]
    counts_gt_8192_problem = [s[4] for s in stats_problem]
    prompt_avgs_problem = [s[5] for s in stats_problem]    # 新增：平均 prompt_length

    # 5) 画图保存 (按 problem_name 和 distance 聚类)
    plt.figure(figsize=(max(6, 0.35 * len(labels_distance) + 3), 5))
    bars_distance = plt.bar(range(len(labels_distance)), avgs_distance)
    plt.xticks(range(len(labels_distance)), labels_distance, rotation=45, ha="right")
    plt.ylabel("Average response_length")
    plt.title("Average response_length by problem_name and distance")

    # 在柱子上标注样本数 n, =16384, >8192，以及 avg_prompt
    for b, c, c16, c8192, pavg in zip(
        bars_distance,
        counts_distance,
        counts_eq_16384_distance,
        counts_gt_8192_distance,
        prompt_avgs_distance
    ):
        # 如果没有 prompt 数据，就不展示那一行
        if pavg is None:
            txt = f"n={c}\n16k:{c16}\n>8k:{c8192}"
        else:
            txt = f"n={c}\n16k:{c16}\n>8k:{c8192}\n{pavg:.1f}"
        plt.text(
            b.get_x() + b.get_width() / 2.0,
            b.get_height(),
            txt,
            ha="center", va="bottom", fontsize=8
        )

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png_path), exist_ok=True)
    plt.savefig(out_png_path.replace(".png", "_by_problem_distance.png"), dpi=200)
    plt.close()

    # 6) 画图保存 (按 problem_name 单独聚类)
    plt.figure(figsize=(max(6, 0.35 * len(labels_problem) + 3), 5))
    bars_problem = plt.bar(range(len(labels_problem)), avgs_problem)
    plt.xticks(range(len(labels_problem)), labels_problem, rotation=45, ha="right")
    plt.ylabel("Average response_length")
    plt.title("Average response_length by problem_name")

    for b, c, c16, c8192, pavg in zip(
        bars_problem,
        counts_problem,
        counts_eq_16384_problem,
        counts_gt_8192_problem,
        prompt_avgs_problem
    ):
        if pavg is None:
            txt = f"n={c}\n16k:{c16}\n>8k:{c8192}"
        else:
            txt = f"n={c}\n16k:{c16}\n>8k:{c8192}\n{pavg:.1f}"
        plt.text(
            b.get_x() + b.get_width() / 2.0,
            b.get_height(),
            txt,
            ha="center", va="bottom", fontsize=8
        )

    plt.tight_layout()
    plt.savefig(out_png_path.replace(".png", "_by_problem.png"), dpi=200)
    plt.close()

    # 7) 返回统计结果（保持原来的结构，不引入 prompt，避免外部调用改动）
    result_distance = OrderedDict()
    result_problem = OrderedDict()

    for (pname, dist), avg, cnt, cnt16, cnt8192, _ in stats_distance:
        result_distance[(pname, dist)] = {
            "avg_response_length": avg,
            "count": cnt,
            "count_eq_16384": cnt16,
            "count_gt_8192": cnt8192
        }

    for pname, avg, cnt, cnt16, cnt8192, _ in stats_problem:
        result_problem[pname] = {
            "avg_response_length": avg,
            "count": cnt,
            "count_eq_16384": cnt16,
            "count_gt_8192": cnt8192
        }

    return result_distance, result_problem


# === 示例调用 ===
if __name__ == "__main__":
    json_path = "/inspire/hdd/global_user/xucaijun-253108120121/Data-Select-verl/recipe/nppc/save_json_list_20251106_102902.json"
    out_png_path = "/inspire/hdd/global_user/xucaijun-253108120121/Data-Select-verl/recipe/nppc/save_json_20251106_102902.png"
    stats_distance, stats_problem = plot_avg_response_length_by_problem_distance(json_path, out_png_path)
    # 打印简要汇总
    print("按 problem_name 和 distance 聚类的统计：")
    for k, v in stats_distance.items():
        pname, dist = k
        print(f"{pname} (d={dist}): avg={v['avg_response_length']:.2f}, n={v['count']}, =16384:{v['count_eq_16384']}, >8192:{v['count_gt_8192']}")
    
    print("\n按 problem_name 单独聚类的统计：")
    for k, v in stats_problem.items():
        print(f"{k}: avg={v['avg_response_length']:.2f}, n={v['count']}, =16384:{v['count_eq_16384']}, >8192:{v['count_gt_8192']}")
    
    print(f"图已保存至: {out_png_path.replace('.png', '_by_problem_distance.png')}")
    print(f"图已保存至: {out_png_path.replace('.png', '_by_problem.png')}")
