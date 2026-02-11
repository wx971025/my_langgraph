import time
import inspect
from functools import wraps

from utils import logger

def time_record(func):
    if inspect.iscoroutinefunction(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            result = await func(*args, **kwargs)
            end_time = time.time()
            duration = end_time - start_time
            logger.info(f"[timeit_async] Function '{func.__name__}' executed in {duration:.4f} seconds")
            return result
        return wrapper
    elif inspect.isasyncgenfunction(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            async for item in func(*args, **kwargs):
                yield item
            end_time = time.time()
            duration = end_time - start_time
            logger.info(f"[timeit_async_gen] Async generator '{func.__name__}' executed in {duration:.4f} seconds")
            return
        return wrapper
    else:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            duration = end_time - start_time
            logger.info(f"[timeit] Function '{func.__name__}' executed in {duration:.4f} seconds")
            return result
        return wrapper
