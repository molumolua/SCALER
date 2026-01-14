from openai import OpenAI
import openai
import time
import os
import time
import logging
from functools import partial
from multiprocessing import Pool
from tqdm import tqdm
from logger import setup_logger
from zai import ZhipuAiClient
MODEL="glm-4.6"
client = ZhipuAiClient(api_key="")


# BASE_URL="https://api.openai.com/v1"
# API_KEY=None
# MODEL="gpt-5"
# client = OpenAI(base_url=BASE_URL,api_key=API_KEY)
def get_oai_completion(prompt,model,temperature,think=False,stream=False):
    try:
        # print(prompt)
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                # {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            stream=stream
        )
        if stream:
            answer=stream_get_answer(response)
            # print(answer)
            return answer
        # print(response)
        answer = response.choices[0].message.content
        # print(f"answer:{answer}")
        if think:
            think_answer=response.choices[0].message.reasoning_content
            return answer,think_answer
        return answer
    except Exception as e:
        print(f"Error fetching answer: {e}")
        return None
def stream_get_answer(stream):
    reasoning_content=""
    answer_content=""
    is_answering=False
    for chunk in stream:
        # 处理usage信息
        if not getattr(chunk, 'choices', None):
            print("\n" + "=" * 20 + "Token 使用情况" + "=" * 20 + "\n")
            print(chunk.usage)
            continue

        delta = chunk.choices[0].delta

        # 检查是否有reasoning_content属性
        if not hasattr(delta, 'reasoning_content'):
            continue

        # 处理空内容情况
        if not getattr(delta, 'reasoning_content', None) and not getattr(delta, 'content', None):
            continue

        # 处理开始回答的情况
        if not getattr(delta, 'reasoning_content', None) and not is_answering:
            is_answering = True

        # 处理思考过程
        if getattr(delta, 'reasoning_content', None):
            # print(delta.reasoning_content, end='', flush=True)
            reasoning_content += delta.reasoning_content
        # 处理回复内容
        elif getattr(delta, 'content', None):
            # print(delta.content, end='', flush=True)
            answer_content += delta.content
    return "<think>\n"+reasoning_content+"\n</think>\n\n"+answer_content
def call_chatgpt(prompt,model):
    success = False
    re_try_count = 10
    ans = ''
    while not success and re_try_count >= 0:
        re_try_count -= 1
        try:
            ans = get_oai_completion(prompt,model)
            success = True
        except:
            time.sleep(5)
            print('retry for sample:', prompt)
    return ans


def get_answer_from_chat_model(prompt, logger=None, eng='gpt-3.5-turbo', temperature=0.0, timeout=20, max_try=3,think=False):
    """
    向聊天模型发送单个请求，并返回回答。

    Args:
        prompt (str): 提示词。
        logger (logging.Logger): 日志记录器。
        eng (str): 使用的模型名称。
        temperature (float): 温度参数。
        timeout (int): 请求超时时间（秒）。
        max_try (int): 最大重试次数。

    Returns:
        str: 模型的回答。
    """
    # if eng not in [
    #     "gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-3.5-turbo-0613", "gpt-3.5-turbo-16k-0613","deepseek-reasoner",
    #     "gpt-4", "gpt-4-0613", "gpt-4-32k", "gpt-4-32k-0613", "gpt-3.5-turbo-1106","gpt-4o","deepseek-ai/DeepSeek-R1","deepseek-r1",
    #     "glm-4-plus"
    # ]:
    #     raise ValueError(f"Unsupported model: {eng}")

    is_success = False
    num_exception = 0

    while not is_success:
        if max_try > 0 and num_exception >= max_try:
            logger.error(f"Max retries reached for question: {prompt}") if logger else None
            return ""
        try:
            return  get_oai_completion(prompt,eng,temperature,think)
        except Exception as e:
            num_exception += 1
            sleep_time = min(num_exception, 2)
            if logger:
                is_print_exc = num_exception % 10 == 0
                logger.error(f"Exception for question '{prompt}': {e}", exc_info=is_print_exc)
                logger.info(f"Retry {num_exception}/{max_try} after sleeping for {sleep_time} seconds.")
            time.sleep(sleep_time)
            is_success = False

def wrapper(idx_args, func):
    """
    包装函数，用于多进程返回索引和结果。

    Args:
        idx_args (tuple): (索引, 参数)
        func (callable): 要调用的函数。

    Returns:
        tuple: (索引, 结果)
    """
    idx, args = idx_args
    res = func(args)
    return idx, res
def get_answer_from_model(prompt,tokenizer,llm,param):
    return NotImplementedError
def batch_get_chat_api(examples, eng, pre_fun, post_fun,
                       logger=None, n_processes=4, temperature=0.7, timeout=20, max_try=3,think=False, **kwargs):
    """
    批量处理聊天模型的 API 请求。

    Args:
        examples (list): 示例数据列表，每个元素是包含 'question' 键的字典。
        eng (str): 使用的模型名称。
        pre_fun (callable): 前处理函数。
        post_fun (callable): 后处理函数。
        logger (logging.Logger): 日志记录器。
        n_processes (int): 使用的进程数。
        temperature (float): 温度参数。
        timeout (int): 请求超时时间（秒）。
        max_try (int): 最大重试次数。
        **kwargs: 其他可选参数。

    Returns:
        None
    """ 
    get_answer_func = partial(
        get_answer_from_chat_model,
        logger=logger,
        eng=eng,
        temperature=temperature,
        timeout=timeout,
        max_try=max_try,
        think=think,
        **kwargs
    )

    prompts = [f"{pre_fun(example)}" for example in examples]

    idx2res = {}
    with Pool(n_processes) as pool:
        tasks = enumerate(prompts)
        wrapped_func = partial(wrapper, func=get_answer_func)
        for idx, response in tqdm(pool.imap_unordered(wrapped_func, tasks), total=len(prompts), desc="Processing"):
            idx2res[idx] = response

    for idx, example in enumerate(examples):
        post_fun(example, idx2res.get(idx, ""))
            
if __name__ == "__main__":
    # answer=get_oai_completion("你是什么模型？",model='qwen3-235b',temperature=0.6,stream=False)
    # print(answer)
    logger =setup_logger()
    problem=dict()
    problem['prompt']="what your name"
    def pre_fun(example):
        return example['prompt']
    def post_fun(example,reply):
        example['response']=reply
    problems=[problem]
    batch_get_chat_api(
            examples=problems,
            eng=MODEL,
            pre_fun=pre_fun,  # simplified
            post_fun=post_fun,
            logger=None,
            n_processes=1,
            temperature=1,
            timeout=20,
            max_try=3,
            think=False
    )
    print(problem['response'])