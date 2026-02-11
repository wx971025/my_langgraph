from langchain_core.tools import tool
from datetime import datetime

from utils.decorator import time_record


@tool("system_time")
@time_record
def system_time_tool() -> str:
    """Get the system time
    Returns:
        str: The system time, China Standard Time
    """
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"The system time is {current_time}, China Standard Time"
