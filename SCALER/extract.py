import re
from typing import Optional, Tuple
from typing import Any, Dict, List, Optional
# from process_dataset import load_and_prepare_dataset
try:
    from logger import setup_logger
except:
    from SCALER.logger import setup_logger
    
def show_literal_newlines(s: str) -> str:
    # 只把真实控制字符替换成可见转义；已存在的 "\n" 文本不受影响
    return (s
            .replace('\r\n', '\\r\\n')
            .replace('\n', '\\n')
            .replace('\t', '\\t')
            .replace('\r', '\\r'))
    
CANDIDATE_CODE_KEYS = [
    "code", "program", "source_code", "final_code", "solution", "answer", "submission",
]

def _extract_code_from_row(row: Dict[str, Any]) -> Optional[str]:
    """尽量从一条样本中拿到可用的代码字符串，兼容多种字段结构。"""
    # 1) 直接可用的字符串字段
    for k in CANDIDATE_CODE_KEYS:
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v

    # 2) solutions 可能是 list[str] 或 list[dict]
    v = row.get("solutions") or row.get("solution_set")
    if isinstance(v, list) and v:
        # 优先取第一个非空字符串
        for item in v:
            if isinstance(item, str) and item.strip():
                return item
            if isinstance(item, dict):
                # dict 场景再走一次候选键
                for k2 in CANDIDATE_CODE_KEYS + ["code_text", "content", "data"]:
                    vv = item.get(k2)
                    if isinstance(vv, str) and vv.strip():
                        return vv
    # 3) dict(list)
    if isinstance(v,dict) and v:
        for k in CANDIDATE_CODE_KEYS + ["code_text", "content", "data"]:
            vv = v.get(k)
            if isinstance(vv, str) and vv.strip():
                return vv
            if isinstance(vv,list) and vv:
                for item in vv:
                    if isinstance(item, str) and item.strip():
                        # code_contest
                        return item
    return None


def extract_last_code_block(answer: str) -> Tuple[Optional[str], Optional[str]]:
    """
    从文本中提取“最后一个”```fenced code block```。
    支持形如:
        ```python
        ...code...
        ```
        ``` cpp
        ...code...
        ```
        ```
        ...code...
        ```

    返回: (code:str|None, lang:str|None)，若未找到返回 (None, None)
    """
    if not answer:
        return None, None

    # 匹配：``` [可选空格][可选语言] 换行 代码 ```   —— 非贪婪捕获代码
    fence_pat = re.compile(
        r"```[ \t]*([A-Za-z0-9_+\-\.]*)[ \t]*\r?\n(.*?)```",
        re.DOTALL
    )

    matches = list(fence_pat.finditer(answer))
    if matches:
        m = matches[-1]  # 取最后一个
        lang = (m.group(1) or "").strip().lower() or None
        code = m.group(2).strip()
        return code, lang

    # 兜底：如果有开头的 ``` 但缺少收尾 ```，取到文本末尾
    open_pat = re.compile(
        r"```[ \t]*([A-Za-z0-9_+\-\.]*)[ \t]*\r?\n(.*)$",
        re.DOTALL
    )
    m = open_pat.search(answer)
    if m:
        lang = (m.group(1) or "").strip().lower() or None
        code = m.group(2).strip()
        return code, lang

    return None, None


# 定义下一节的边界标题（大小写不敏感、允许可选的 "Format"）
SECTION_BOUNDARY = re.compile(
    r"(?im)^\s*(output(?:\s*format)?|examples?|sample(?:\s+input|\s+tests?)?|notes?|constraints?)\s*:?\s*$"
)

def split_with_input_section(text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    将题面分割为：
      - before_text: Input 之前的内容
      - input_text:  Input 小节（从 Input 标题行之后到下一个小节标题之前）
      - after_text:  下一个小节标题开始直到文末
    若没有找到 Input，小节：返回 (text, None, None)
    """
    if not isinstance(text, str):
        return None, None, None

    # 1) 找 Input / Input Format 标题行（大小写不敏感，允许冒号）
    m_start = re.search(r"(?im)^\s*input(?:\s*format)?\s*:?\s*$", text)
    if not m_start:
        # 未找到 Input，全部作为 before 返回
        return text, None, None

    # 计算三段的边界
    before_text = text[:m_start.start()]
    input_body_start = m_start.end()

    # 2) 从 Input 后往下找下一章节标题（Output / Examples / Sample / Notes / Constraints）
    m_end = SECTION_BOUNDARY.search(text, pos=input_body_start)
    if m_end:
        input_body_end = m_end.start()
        after_text = text[m_end.start():]
    else:
        input_body_end = len(text)
        after_text = ""

    # 3) 提取并做轻微清理：去掉 Input 内容开头的多余空行、末尾空白
    input_text_raw = text[input_body_start:input_body_end]
    input_text = re.sub(r"^\s*\n", "", input_text_raw)  # 去掉最前面的连串空行
    input_text = input_text.rstrip()

    # 4) 统一去掉 before 的末尾多余空白（不破坏中间结构）
    before_text = before_text.rstrip("\n")

    # 5) 若 after 存在，保留原貌；如果是空字符串则返回 ""（而不是 None）
    return before_text, input_text if input_text else "", after_text


def parse_gen_script(text: str) -> List[Dict[str, Any]]:
    """
    解析包含多组 '# ----- Group k: ... -----' 标题与 './gen ...' 命令的脚本文本。
    返回列表，每个元素是一个字典：
        {
          "note": str,            # 原始备注
          "commands": List[str],  # 本组所有 ./gen 命令（按出现顺序）
        }
    """
    lines = text.splitlines()
    groups: List[Dict[str, Any]] = []
    current = {
                "note":"first_line",
                "commands":[]
    }


    for line in lines:
        s = line.strip()
        if  s  and s.startswith("#"):
            if len(current["commands"])>0:
                groups.append(current)

            current = {
                "note":s,
                "commands":[]
            }
        elif s and s.startswith("./gen "): 
            current["commands"].append(s)

    # 文件结束后把最后一组加入
    if current:
        groups.append(current)

    return groups

def parse_one_gen_script(text: str) -> Dict[str, Any]:
    """
    解析包含 './gen ...' 命令的脚本文本。
    返回一个字典：
        {
          "note": str,            # 原始备注
          "commands": List[str],  # 本组所有 ./gen 命令（按出现顺序）
        }
    """
    lines = text.splitlines()
    current = {
                "note":"first_line",
                "commands":[]
    }


    for line in lines:
        s = line.strip()
        if s and s.startswith("./gen "): 
            current["commands"].append(s)

    return current

import re

_PLACEHOLDER_RE = re.compile(r"\{([a-zA-Z_]\w*)\}")

def safe_format_template(raw_template: str, values: dict) -> str:
    """
    安全渲染：保留 {name} 占位符；转义其他所有 { } 为 {{ }}，再 .format(values)。
    """

    # 1) 暂存合法占位符的位置，用哨兵标记，避免它们被转义
    sentinels = []
    def _mark(m):
        name = m.group(1)
        token = f"\x00PH_{len(sentinels)}\x00"  # 哨兵
        sentinels.append((token, name))
        return token

    marked = _PLACEHOLDER_RE.sub(_mark, raw_template)

    # 2) 全文转义剩余的大括号
    escaped = marked.replace("{", "{{").replace("}", "}}")

    # 3) 恢复占位符（把哨兵替换回 {name}）
    for token, name in sentinels:
        escaped = escaped.replace(token, "{%s}" % name)

    # 4) 最终渲染
    return escaped.format(**values)

def get_function_code_from_str(code_str, function_name):
    # 查找函数定义的位置
    start_index = code_str.find(f"def {function_name}")
    if start_index == -1:
        return None  # 如果没找到函数定义，返回 None
    
    # 从函数定义的位置开始，找到函数体的结束
    def_indent_level = None  # 用来记录函数的缩进级别
    indent_level = None
    function_code = code_str[start_index:]  # 从函数定义开始提取代码
    in_multiline_comment = False  # 标志是否在多行注释中
    lines = function_code.splitlines()
    function_lines = []
    
    for line in lines:
        # 跳过单行注释
        if line.strip().startswith("#"):
            continue
        
        # 处理多行注释
        if '"""' in line or "'''" in line:
            # 判断是否是多行注释的开始或结束
            if not in_multiline_comment:
                in_multiline_comment = True
                continue  # 跳过注释开始行
            else:
                in_multiline_comment = False
                continue  # 跳过注释结束行
        
        if in_multiline_comment:
            continue  # 如果在多行注释中，跳过该行
        
        
        if def_indent_level is None:
            # 首次确定函数的缩进级别
            def_indent_level = len(line) - len(line.lstrip())
            function_lines.append(line)
        else:
            if indent_level is None:
                indent_level = len(line) - len(line.lstrip())
            if indent_level:
                if len(line) - len(line.lstrip()) >= indent_level:
                    function_lines.append(line)
                else:
                    break  # 当缩进不符合时，表示函数结束
    
    return '\n'.join(function_lines)

    
    
# if __name__ == "__main__":
#     logger=setup_logger()
#     dataset = load_and_prepare_dataset("./Dataset/hf_datasets/code_contest_upgrade_1",file_glob="output_problems.jsonl",logger=logger,load_type="json")
#     for item in dataset:
#         code = extract_last_code_block(item['answer'])
#         print(code)