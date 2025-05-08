import torch
import gc
import os

def print_details():
    for slurm_var in ["SLURMD_NODENAME","SBATCH_CLUSTERS", 
                      "SBATCH_PARTITION","SLURM_JOB_PARTITION",
                      "SLURM_NODEID","SLURM_MEM_PER_GPU",
                      "SLURM_MEM_PER_CPU","SLURM_MEM_PER_NODE","SLURM_JOB_ID"]:
        try:
            print(slurm_var, os.environ[slurm_var])
        except:
            print(slurm_var, "doesnt exist")
    try:
        print('torch.cuda.get_device_name()',torch.cuda.get_device_name())
        print('torch.cuda.get_device_capability()',torch.cuda.get_device_capability())
        current_device = torch.cuda.current_device()
        gpu = torch.cuda.get_device_properties(current_device)
        print(f"GPU Name: {gpu.name}")
        print(f"GPU Memory Total: {gpu.total_memory / 1024**2} MB")
        print(f"GPU Memory Free: {torch.cuda.memory_allocated(current_device) / 1024**2} MB")
        print(f"GPU Memory Used: {torch.cuda.memory_reserved(current_device) / 1024**2} MB")
    except:
        print("couldnt print cuda details")

def get_gpu_memory_usage():
    if torch.cuda.is_available():
        # Get the current device
        device = torch.cuda.current_device()
        
        # Get allocated memory in bytes
        allocated = torch.cuda.memory_allocated(device)
        
        # Get cached memory in bytes (allocated + cached = total reserved)
        reserved = torch.cuda.memory_reserved(device)
        
        # Convert to more readable format (MB)
        allocated_mb = allocated / 1024 / 1024
        reserved_mb = reserved / 1024 / 1024
        
        return {
            "device": device,
            "allocated_bytes": allocated,
            "allocated_mb": allocated_mb,
            "reserved_bytes": reserved,
            "reserved_mb": reserved_mb
        }
    else:
        return {}

def find_cuda_objects():
    cuda_objects = []
    for obj in gc.get_objects():
        try:
            if isinstance(obj, torch.Tensor) and obj.is_cuda:
                cuda_objects.append(obj)
            elif isinstance(obj, torch.nn.Module) and any(p.is_cuda for p in obj.parameters()):
                cuda_objects.append(obj)
        except:
            pass  # Avoid issues with objects that can't be inspected
    return cuda_objects

def find_cuda_tensors_with_grads():
    tensors_with_grads = []
    for obj in gc.get_objects():
        try:
            if isinstance(obj, torch.Tensor) and obj.is_cuda and obj.grad is not None:
                tensors_with_grads.append(obj)
        except:
            pass  # Ignore inaccessible objects
    return tensors_with_grads

def delete_unique_objects(list1, list2):
    """Delete objects that are only in one of the two lists."""
    set1 = {id(obj) for obj in list1}
    set2 = {id(obj) for obj in list2}
    
    unique_in_list1 = [obj for obj in list1 if id(obj) not in set2]
    unique_in_list2 = [obj for obj in list2 if id(obj) not in set1]

    # Delete tensors and modules
    for obj in unique_in_list1 + unique_in_list2:
        
        if isinstance(obj, torch.Tensor):
            if obj.grad is not None:
                if obj.grad.grad_fn is not None:
                    obj.grad.detach_()
            obj.grad = None  # Clear gradients first
        #obj.detach_()
        obj.to("cpu")
        del obj  # Delete the object

    # Force garbage collection and free CUDA memory
    gc.collect()
    torch.cuda.empty_cache()