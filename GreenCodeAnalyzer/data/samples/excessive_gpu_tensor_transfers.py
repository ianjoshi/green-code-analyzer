import torch

def inefficient_tensor_transfers():
    # Sample tensor initialized on GPU
    tensor = torch.randn(1000, 1000).cuda()
    
    # First violation: unnecessary transfer to CPU and back to GPU
    cpu_tensor = tensor.cpu()  # Transfer to CPU
    intermediate = cpu_tensor + 1
    gpu_tensor = intermediate.cuda()  # Transfer back to GPU
    
    # Some processing on GPU
    result = gpu_tensor * 2
    
    # Second violation: another unnecessary round trip
    cpu_result = result.cpu()  # Transfer to CPU again
    final_tensor = cpu_result - 1
    gpu_final = final_tensor.cuda()  # Transfer back to GPU
    
    return gpu_final