import os
import json
import random
import json
import os
import numpy as np
from pathlib import Path
from typing import Iterable, Union, Any
from collections import defaultdict
import re


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def load_jsonl(file: Union[str, Path]) -> Iterable[Any]:
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except:
                print("Error in loading:", line)
                exit()


def save_jsonl(samples, save_path):
    # ensure path
    folder = os.path.dirname(save_path)
    os.makedirs(folder, exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print("Saved to", save_path)

def save_json(samples, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(samples, f, ensure_ascii=False, indent=4)
    print("Saved to", save_path)

def lower_keys(example):
    new_example = {}
    for key, value in example.items():
        if key != key.lower():
            new_key = key.lower()
            new_example[new_key] = value
        else:
            new_example[key] = value
    return new_example


EXAMPLES = get_examples()


def load_prompt(data_name, prompt_type, num_shots):
    if not num_shots:
        return []

    if data_name in ["gsm_hard", "svamp", "tabmwp", "asdiv", "mawps"]:
        data_name = "gsm8k"
    if data_name in ["math_oai", "hungarian_exam", "math-oai", "aime24", "amc23"]:
        data_name = "math"
    if data_name in ["sat_math"]:
        data_name = "mmlu_stem"
    if data_name in [
        "gaokao2024_I",
        "gaokao2024_II",
        "gaokao_math_qa",
        "gaokao2024_mix",
        "cn_middle_school",
    ]:
        data_name = "gaokao"

    if prompt_type in ["tool-integrated"]:
        prompt_type = "tora"

    return EXAMPLES[data_name][:num_shots]


PROMPT_TEMPLATES = {
    "direct": ("Question: {input}\nAnswer: ", "{output}", "\n\n"),
    "cot": ("Question: {input}\nAnswer: ", "{output}", "\n\n\n"),
    "pal": ("Question: {input}\n\n", "{output}", "\n---\n"),
    "tool-integrated": ("Question: {input}\n\nSolution:\n", "{output}", "\n---\n"),
    "self-instruct": ("<|user|>\n{input}\n<|assistant|>\n", "{output}", "\n"),
    "tora": ("<|user|>\n{input}\n<|assistant|>\n", "{output}", "\n"),
    "wizard_zs": (
        "### Instruction:\n{input}\n\n### Response: Let's think step by step.",
        "{output}",
        "\n\n\n",
    ),
    "platypus_fs": (
        "### Instruction:\n{input}\n\n### Response:\n",
        "{output}",
        "\n\n\n",
    ),
    "deepseek-math": (
        "User: {input}\nPlease reason step by step, "
        "and put your final answer within \\boxed{{}}.\n\nAssistant:",
        "{output}",
        "\n\n\n",
    ),
    "kpmath": (
        "User: Please reason step by step and put your final answer at the end "
        'with "The answer is: ".\n\n{input}\n\nAssistant:',
        "{output}",
    ),
    "jiuzhang": (
        "## Question\n{input}\n\n## Solution\n",
        "{output}",
        "\n\n\n",
    ),
    "jiuzhang_tora": (
        "## Question\n{input}\n\n## Code Solution\n",
        "{output}",
        "\n\n\n",
    ),
    "jiuzhang_nl": (
        "## Question\n{input}\n\n## Natural Language Solution\n",
        "{output}",
        "\n\n\n",
    ),
    "mmiqc": (
        'Please solve the following problem and put your answer at the end with "The answer is: ".\n\n{input}\n\n',
        "{output}",
        "\n\n\n",
    ),
    "abel": (
        "Question:\n{input}\nAnswer:\nLet's think step by step.\n",
        "{output}",
        "\n\n",
    ),
    "shepherd": ("{input}\n", "{output}", "\n\n\n"),
    "qwen-boxed": (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n{input}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
        "<|im_start|>assistant\n",
        "{output}",
        "\n\n",
    ),
    "qwen25-math-cot": (
        "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
        "<|im_start|>user\n{input}<|im_end|>\n"
        "<|im_start|>assistant\n",
        "{output}",
        "\n\n",
    ),
    "mathstral": (
        "{input}\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
        "{output}",
        "\n\n",
    ),
    "internlm-math-fs": ("Question:{input}\nAnswer:", "{output}", "\n"),
    "internlm-math-chat": (
        "<|im_start|>user\n{input}<|im_end|>\n" "<|im_start|>assistant\n",
        "{output}",
        "\n\n",
    ),
    "mistral": (
        "[INST] {input}[/INST]",
        "{output}",
        "\n\n",
    ),
    "numina": ("### Problem: {input}\n### Solution:", " {output}", "\n\n"),
    "think": (
        "<｜begin▁of▁sentence｜>A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer, and put your final answer within \\boxed{{}} . The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.<｜User｜>{input}\n<｜Assistant｜><think> ",
        "{output}",
        "\n\n\n",
    ),
    "qwen_think": (
        "<|im_start|>system\nA conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer, and put your final answer within \\boxed{{}} . The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.<|im_end|>\n<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n",
        "{output}",
        "\n\n",
    ),
}


def construct_prompt(example, data_name, args):
    if args.adapt_few_shot and data_name in [
        "gaokao2024_I",
        "gaokao2024_II",
        "gaokao_math_qa",
        "gaokao2024_mix",
        "cn_middle_school",
    ]:
        demos = load_prompt(data_name, args.prompt_type, 5)
    else:
        demos = load_prompt(data_name, args.prompt_type, args.num_shots)
    prompt_type = args.prompt_type
    if prompt_type == "platypus_fs":
        prompt_type = "cot"
    if prompt_type == "tool-integrated":
        prompt_type = "tora"

    prompt_temp = PROMPT_TEMPLATES[args.prompt_type]

    splitter = prompt_temp[2]
    input_template, output_template, splitter = (
        prompt_temp[0],
        prompt_temp[1],
        prompt_temp[2],
    )
    if args.prompt_type == "qwen25-math-cot":
        # Hotfix to support putting all demos into a single turn
        demo_prompt = splitter.join([q + "\n" + a for q, a in demos])
    else:
        demo_prompt = splitter.join(
            [
                input_template.format(input=q) + output_template.format(output=a)
                for q, a in demos
            ]
        )
    context = input_template.format(input=example["question"])
    if len(demo_prompt) == 0 or (
        args.adapt_few_shot and example["gt_ans"] not in ["A", "B", "C", "D", "E"]
    ):
        full_prompt = context
    else:
        if args.prompt_type == "qwen25-math-cot":
            # Hotfix to supportting put all demos into a single turn
            full_prompt = demo_prompt + splitter + example["question"]
            full_prompt = input_template.format(input=full_prompt)
        else:
            full_prompt = demo_prompt + splitter + context

    if args.prompt_type == "platypus_fs":
        full_prompt_temp = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:\n"
        )
        full_prompt = full_prompt_temp.format(instruction=full_prompt)

    if prompt_type == "tora":
        full_prompt = (
            """Integrate step-by-step reasoning and Python code to solve math problems using the following guidelines:

- Analyze the question and write functions to solve the problem; the function should not take any arguments.
- Present the final result in LaTeX using a `\boxed{}` without any units.
- Utilize the `pi` symbol and `Rational`` from Sympy for $\pi$ and fractions, and simplify all fractions and square roots without converting them to decimal values.

Here are some examples you may refer to:

---

"""
            + full_prompt
        )

    return full_prompt.strip(" ")  # important!


key_map = {
    "gt": "Ground Truth",
    "pred": "Prediction",
    "gt_cot": "Reference CoT",
    "score": "Score",
}


def show_sample(sample, print_all_preds=False):
    print("==" * 20)
    for key in ["idx", "type", "level", "dataset"]:
        if key in sample:
            # capitalize
            print("{}: {}".format(key[0].upper() + key[1:], sample[key]))
    print("Question:", repr(sample["question"]))
    if "code" in sample:
        if print_all_preds:
            for code in sample["code"]:
                print("-" * 20)
                print("code:", code)
            print("Execution:", sample["report"])
        else:
            print("Solution:\n", sample["code"][0])
            print("Execution:", sample["report"][0])
    if "pred" in sample:
        print("Prediction:", repr(sample["pred"][0]))
    for key in ["gt", "score", "unit", "gt_cot"]:
        if key in sample:
            _key = key_map.get(key, key)
            print("{}: {}".format(_key, repr(sample[key])))
    print()

def find_position(section, next_section, section_list, title_list, matches, answer_len, logger, begin=0):
    try:
        logger.debug(f"Finding position for section '{section}' and next_section '{next_section}' starting from index {begin}")
        start = -1
        end = -1
        # 查找当前章节的位置
        for i in range(len(title_list)-1,-1,-1):
            if title_list[i] == section_list[section]:
                start = matches[i].end()
                logger.debug(f"Found start of section '{section_list[section]}' at position {start}")
                break
        if start == -1:
            logger.error(f"Section '{section_list[section]}' not found in the title list.")
            return start, end, begin

        # 查找下一个章节的位置（如果有的话）
        if next_section < len(section_list):
            for i in range(len(title_list)-1,-1,-1):
                if title_list[i] == section_list[next_section]:
                    end = matches[i].start()
                    logger.debug(f"Found end of section '{section_list[section]}' at position {end}")
                    break
        else:
            end = answer_len  # 如果没有下一个章节，则使用提供的 `answer_len` 作为结束位置。
            logger.debug(f"No next section. Using answer_len {answer_len} as end position.")

        # # 查找当前章节的位置
        # for i in range(begin, len(title_list)):
        #     if title_list[i] == section_list[section]:
        #         begin = i + 1
        #         start = matches[i].end()
        #         logger.debug(f"Found start of section '{section_list[section]}' at position {start}")
        #         break
        # if start == -1:
        #     logger.error(f"Section '{section_list[section]}' not found in the title list.")
        #     return start, end, begin

        # # 查找下一个章节的位置（如果有的话）
        # if next_section < len(section_list):
        #     for i in range(begin, len(title_list)):
        #         if title_list[i] == section_list[next_section]:
        #             begin = i
        #             end = matches[i].start()
        #             logger.debug(f"Found end of section '{section_list[section]}' at position {end}")
        #             break
        # else:
        #     end = answer_len  # 如果没有下一个章节，则使用提供的 `answer_len` 作为结束位置。
        #     logger.debug(f"No next section. Using answer_len {answer_len} as end position.")

        if end == -1 and next_section < len(section_list):
            logger.error(f"Next section '{section_list[next_section]}' not found in the title list.")
        return start, end, begin
    except Exception as e:
        logger.error(f"Error in find_position: {e}")
        return -1, -1, begin

def clear_string(s):
    return s[:-1] if s and s[-1] == ':' else s

def parse_answer(answer_text, sections, logger):
    try:
        logger.debug("Parsing answer text.")
        extracted = {section: "" for section in sections}

        # 匹配标题部分的正则表达式（支持以###或更多#开头的格式）
        pattern = re.compile(
            r'^\s*(?:\d+\.\s*|\d+\s*|\*+\s*|\-+\s*|#+\s*)?\*\*(?:\d+\.\s*|\d+\s*|\*+\s*|\-+\s*|#+\s*)?(.*?)\*\*:?', re.MULTILINE
        )
        matches = list(pattern.finditer(answer_text))

        # 如果没有找到匹配，再尝试另一种格式的正则表达式
        if not matches:
            pattern = re.compile(
                r'^\s*(?:\d+\.\s*|\d+\s*|\*+\s*|\-+\s*|#+\s*)?\*\*(?:\d+\.\s*|\d+\s*|\*+\s*|\-+\s*|#+\s*)?(.*?):\*\*', re.MULTILINE
            )
            matches = list(pattern.finditer(answer_text))

        # 如果仍没有匹配，返回空内容
        if not matches:
            logger.warning("No section headers matched in the answer text.")
            logger.warning(answer_text)
            return ("" for _ in sections)

        logger.debug(f"Found {len(matches)} section headers.")
        begin = 0
        title_list = [clear_string(match.group(1).strip()) for match in matches]
        for idx, section in enumerate(sections):
            start, end, begin = find_position(idx, idx + 1, sections, title_list, matches, len(answer_text), logger, begin)
            if start == -1 or end == -1:
                logger.warning(f"Could not extract section '{section}'.")
                continue
            # 提取内容并去除前后空白字符
            content = answer_text[start:end].strip()
            extracted[section] = content
            logger.debug(f"Extracted content for section '{section}': {content[:50]}...")  # 只显示前50个字符
        return (extracted[section] for section in sections)
    except Exception as e:
        logger.error(f"Error in parse_answer: {e}")
        return ("" for _ in sections)

def extract_think_and_after(text):
    """
    提取字符串中 <think> 标签内部的内容，以及 </think> 之后的文本。

    参数：
        text (str): 包含 <think> 标签的完整字符串。

    返回：
        tuple: (think_content, after_think)
               think_content 为 <think>...</think> 中的文本（若没匹配到返回 None）。
               after_think 为 </think> 后的文本（若没匹配到返回 None）。
    """
    # 使用正则表达式，启用 DOTALL (re.DOTALL) 使 '.' 能匹配换行符
    if "<think>" not in text:
        text = "<think>" + text
    pattern = r"<think>(.*?)</think>(.*)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        think_content = match.group(1).strip()
        after_think = match.group(2).strip()
        return think_content, after_think
    else:
        # 如果没有匹配到，就返回 (None, None)
        return None, None
    



def process_output_data(data_list):
    # 使用 defaultdict 来聚合
    grouped = defaultdict(list)

    # 遍历数据，将相同 original_problem 的 dict 聚集在一起
    for item in data_list:
        grouped[item['original_problem']].append(item)

    # 转换成二维 list
    result = list(grouped.values())
    return result
