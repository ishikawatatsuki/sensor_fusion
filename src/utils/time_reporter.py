import time

def time_measurer(func):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        runtime = end_time - start_time
        try:
          runtime = runtime / kwargs['data'].N
        except:
          pass
        return value, runtime
    return wrapper
  

if __name__ == "__main__":
  
  @time_measurer
  def long_task(num_times=100):
      for _ in range(num_times):
          sum([number**2 for number in range(10_000)])
      return "value to return"

  result, processing_time = long_task(num_times=1000)
  
  print(f"Result: {result}")
  print(f"Inference time: {processing_time:.3f} seconds")