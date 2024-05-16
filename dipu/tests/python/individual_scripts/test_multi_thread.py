import threading
import torch
import torch_dipu


def worker(num):
    print(f"Worker {num} started")
    index = num % torch.cuda.device_count()
    torch.cuda.set_device(index)
    print(torch.randn(2, device=f"cuda:{index}"))
    print(f"Worker {num} finished")


for i in range(20):
    t = threading.Thread(target=worker, args=(i,))
    t.start()

main_thread = threading.current_thread()
for t in threading.enumerate():
    if t is main_thread:
        continue
    t.join()

print("All threads finished")
