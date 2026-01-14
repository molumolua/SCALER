import logging
def setup_logger():
    logger = logging.getLogger("BatchOpenAI")
    logger.setLevel(logging.INFO)
    # 创建控制台处理器
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # 创建格式器
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    # 添加处理器到日志记录器
    if not logger.handlers:
        logger.addHandler(ch)
    return logger