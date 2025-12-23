import os
import logging
from langgraph.graph.state import CompiledStateGraph

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


def draw_graph_image(graph: CompiledStateGraph):
    save_dir = "temp/graph_images"
    filename: str = "graph.png"
    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"Saving graph image to {save_dir}")
    image_data = graph.get_graph().draw_mermaid_png()
    save_path = os.path.join(save_dir, filename)
    with open(save_path, "wb") as f:
        f.write(image_data)
    logger.info(f"Graph image saved to {save_path}")
    return save_path
