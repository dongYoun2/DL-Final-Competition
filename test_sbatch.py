import time
import torch


print(torch.cuda.is_available())

for i in range(10):
    print(f'{i}: {time.time()} hello world!')
    time.sleep(1)
