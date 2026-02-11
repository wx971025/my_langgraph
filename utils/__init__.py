import os
import logging
import hashlib
from langgraph.graph.state import CompiledStateGraph
from typing import Literal

class ColoredFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[36m',      # 青色
        'INFO': '\033[32m',       # 绿色
        'WARNING': '\033[33m',    # 黄色
        'ERROR': '\033[31m',      # 红色
        'CRITICAL': '\033[35m',   # 紫色
        'PROCESS': '\033[34m',    # 蓝色
    }
    RESET = '\033[0m'
    
    def format(self, record):
        log_message = super().format(record)
        levelname = record.levelname
        process_id = record.process

        pid_pattern = f'PID:{process_id}'
        colored_process_id = f"{self.COLORS['PROCESS']}PID:{process_id}{self.RESET}"
        log_message = log_message.replace(pid_pattern, colored_process_id)

        if levelname in self.COLORS:
            colored_level = f"{self.COLORS[levelname]}{levelname}{self.RESET}"
            log_message = log_message.replace(levelname, colored_level)
        return log_message

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False        # 防止日志传到根记录器

stream_formatter = ColoredFormatter('[PID:%(process)d][TID:%(thread)d] %(asctime)s [%(levelname)s] %(pathname)s:%(lineno)d - %(message)s')
stream_formatter.datefmt = '%m-%d %H:%M:%S'

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(stream_formatter)
logger.addHandler(stream_handler)


def calculate_md5(file_path: str, input_type: Literal["string", "file"] = "file") -> str:
    def _calculate_file_md5(path):
        hash_md5 = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    if input_type == "file":
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"{file_path} is not a valid file")
        return _calculate_file_md5(file_path)
    elif input_type == "string":
        return hashlib.md5(file_path.encode()).hexdigest()
    else:
        raise NotImplementedError(f"MD5 calculation for {input_type} is not implemented")


def draw_graph_image(graph: CompiledStateGraph, xray: bool = True):
    save_dir = "temp"
    filename: str = "graph.png"
    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"Saving graph image to {save_dir}")
    image_data = graph.get_graph(xray=xray).draw_mermaid_png()
    save_path = os.path.join(save_dir, filename)
    with open(save_path, "wb") as f:
        f.write(image_data)
    logger.info(f"Graph image saved to {save_path}")
    return save_path


def detect_language(s: str) -> Literal["zh", "en"]:
    s = s[:100]

    counts = {
        "zh": 0,   # Chinese
    }

    for ch in s:
        code = ord(ch)
        if 0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF or 0x20000 <= code <= 0x2A6DF:
            counts["zh"] += 1

    lang, cnt = max(counts.items(), key=lambda item: item[1])
    if cnt > 0 and cnt >= 0.10 * len(s):
        return lang
    return "en"
