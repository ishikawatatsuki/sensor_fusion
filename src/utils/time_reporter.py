import time

def time_reporter(func):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        runtime = end_time - start_time
        return value, runtime
    return wrapper
  