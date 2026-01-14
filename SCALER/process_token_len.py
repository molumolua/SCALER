from typing import List, Dict, Any, Literal, Union
from transformers import AutoTokenizer
from typing import Any, Dict, Iterable, List
from tqdm import tqdm
try:
    from transformers import AutoTokenizer
except Exception as e:
    raise ImportError("需要安装 transformers：pip install transformers") from e




def _detect_style(tokenizer) -> str:
    """
    检测常见模板风格：
      - llama3: 存在 <|start_header_id|>, <|eot_id|>
      - llama2/mistral: 词表含 [INST] 或 <<SYS>>
      - chatml/qwen: 有 <|im_start|> / <|im_end|> 或 <|system|> 这类 token
      - 默认回退 chatml
    """
    specials = set(getattr(tokenizer, "all_special_tokens", []) or [])
    vocab = {}
    try:
        vocab = tokenizer.get_vocab() or {}
    except Exception:
        pass

    if {"<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"} & specials:
        return "llama3"
    if "[INST]" in vocab or "<<SYS>>" in vocab or "[/INST]" in vocab:
        return "llama2"
    if {"<|im_start|>", "<|im_end|>", "<|system|>", "<|assistant|>", "<|user|>"} & specials:
        return "chatml"
    return "chatml"


def _format_chatml(messages: List[Dict[str, str]], add_generation_prompt: bool) -> str:
    """ChatML/Qwen 风格."""
    out = []
    for m in messages:
        out.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>")
    if add_generation_prompt and (not messages or messages[-1]["role"] != "assistant"):
        out.append("<|im_start|>assistant\n")
    return "".join(out)


def _format_llama3(messages: List[Dict[str, str]], add_generation_prompt: bool) -> str:
    """Llama-3 风格（headers + eot）."""
    def block(role, content):
        return (
            "<|start_header_id|>" + role + "<|end_header_id|>\n\n" +
            content + "<|eot_id|>"
        )
    out = ["<|begin_of_text|>"]
    for m in messages:
        out.append(block(m["role"], m["content"]))
    if add_generation_prompt and (not messages or messages[-1]["role"] != "assistant"):
        out.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
    return "".join(out)


def _format_llama2(messages: List[Dict[str, str]], add_generation_prompt: bool) -> str:
    """
    Llama-2/Mistral 的 [INST] 风格（简化实现）：
    - 将 system 合并到首条 user 前的 <<SYS>> 区域
    - user/assistant 交替；若多段 system 则并入开头系统块
    """
    sys_buf = []
    conv: List[Dict[str, str]] = []
    for m in messages:
        if m["role"] == "system":
            sys_buf.append(m["content"])
        else:
            conv.append(m)
    sys_text = "\n".join(sys_buf).strip()
    sys_part = f"<<SYS>>\n{sys_text}\n<</SYS>>\n\n" if sys_text else ""

    # 把第一条 user 前置上 system 区
    out = []
    i = 0
    while i < len(conv):
        if conv[i]["role"] == "user":
            user_text = sys_part + conv[i]["content"] if sys_part else conv[i]["content"]
            out.append(f"[INST] {user_text} [/INST]")
            # 紧随的 assistant 直接拼接其内容
            if i + 1 < len(conv) and conv[i + 1]["role"] == "assistant":
                out.append(conv[i + 1]["content"])
                i += 2
            else:
                i += 1
        else:
            # 若遇到开头就是 assistant，则直接原样附加（少见，但保证鲁棒）
            out.append(conv[i]["content"])
            i += 1
    text = "<s>" + "".join(out).strip()
    if add_generation_prompt and (not messages or messages[-1]["role"] != "assistant"):
        text += " [INST] "
    return text


def pack_chat_prompt(
    messages: List[Dict[str, Any]],
    model_path: str,
    add_generation_prompt: bool = False,
    tokenize: bool = False,
    trust_remote_code: bool = True,
) -> Union[str, Dict[str, Any]]:
    """
    将 [{role, content}] 消息列表打包成“带特殊 token 的聊天串”或“input_ids”。
    优先使用 tokenizer.apply_chat_template；若无模板则按常见家族智能回退。

    参数
    - messages: 列表，元素形如 {"role": "system|user|assistant", "content": "..."}
    - model_path: HF 名称或本地路径
    - add_generation_prompt: 若需要在末尾加上“等待下一条 assistant 输出”的前缀，则为 True
    - tokenize: 返回 input_ids 而非字符串
    - trust_remote_code: 某些模型（如 Qwen）需要 True

    返回
    - tokenize=False: str
    - tokenize=True : dict，包含 {"input_ids": List[int], "attention_mask": List[int]}
    """
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    msgs = messages

    if getattr(tok, "chat_template", None):
        return len(tok.apply_chat_template(
            msgs, tokenize=tokenize, add_generation_prompt=add_generation_prompt
        ))

    style = _detect_style(tok)
    if style == "llama3":
        text = _format_llama3(msgs, add_generation_prompt)
    elif style == "llama2":
        text = _format_llama2(msgs, add_generation_prompt)
    else:  # chatml/qwen
        text = _format_chatml(msgs, add_generation_prompt)


    enc = tok(text, add_special_tokens=False)
    return len(enc['input_ids'])


def filter_examples_by_token_budget(
    examples: Iterable[Dict[str, Any]],
    model_path: str,
    max_token: int,
    trust_remote_code: bool = True,
    logger = None
) -> List[Dict[str, Any]]:
    """
    根据 tokenizer 统计每个 example[prompt_key] 中所有 `content` 文本的 token 和，
    仅保留 sum_tokens <= max_token 的样本。
    参数:
      - examples: 可迭代的样本，每个样本为 dict，包含 prompt_key 字段
      - model_path: 传给 transformers.AutoTokenizer.from_pretrained 的路径或名称
      - max_token: token 上限（<= 该值才会保留）
      - prompt_key: 提示字段名，默认 "prompt"
      - trust_remote_code: 对部分模型（如 Qwen 系列）需要 True

    返回:
      - 过滤后的样本列表（保持原有顺序）
    """
    filtered = []
    for ex in tqdm(examples):
        token_len = pack_chat_prompt(ex['prompt'],model_path=model_path,add_generation_prompt=True,tokenize=True,trust_remote_code=trust_remote_code)
        if token_len <= max_token:
            filtered.append(ex)
        else:
            logger.error(f"len {token_len} exceed max_token {max_token}.")

    return filtered
