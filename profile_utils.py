import time
import logging
import psutil

def elapsed_since(start):
    return time.strftime("%H:%M:%S", time.gmtime(time.time() - start))

def get_process_memory():
    return psutil.virtual_memory().used

def profile(func):
    def wrapper(*args, **kwargs):
        mem_before = get_process_memory() / (1024 ** 3)
        start = time.time()
        result = func(*args, **kwargs)
        elapsed_time = elapsed_since(start)
        mem_after = get_process_memory() / (1024 ** 3)
        logging.info(f"{func.__name__}: Memory Before: {mem_before:3.3f} GB, After: {mem_after:3.3f} GB")
        logging.info(f"Memory Consumed: {mem_after - mem_before:3.3f} GB; Exec time (HH:MM:SS): {elapsed_time}")
        return result
    return wrapper
