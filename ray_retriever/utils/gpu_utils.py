from typing import List, Dict, Any
import torch

def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device(f"cuda:0")
    else:
        device = torch.device("cpu")
    return device    

def get_gpu_memory(max_gpus=None) -> List[float]:
    gpu_memory = []
    num_gpus = (
        torch.cuda.device_count()
        if max_gpus is None
        else min(max_gpus, torch.cuda.device_count())
    )

    for gpu_id in range(num_gpus):
        with torch.cuda.device(gpu_id):
            device = torch.cuda.current_device()
            gpu_properties = torch.cuda.get_device_properties(device)
            total_memory = gpu_properties.total_memory / (1024**3)
            allocated_memory = torch.cuda.memory_allocated() / (1024**3)
            available_memory = total_memory - allocated_memory
            gpu_memory.append(available_memory)
    return gpu_memory
