import torch

def reduction_operations_pytorch():
    """
    This function demonstrates the reduction operation code smell in PyTorch:
    1. Using loops to compute sums instead of torch.sum.
    2. Using loops to compute means instead of torch.mean.
    3. Using loops to compute min and max instead of torch.min and torch.max.
    """
    # Create a sample tensor
    tensor = torch.rand(100)
    
    # Violation 1: Element-wise sum instead of torch.sum(tensor)
    total = 0
    for i in range(len(tensor)):
        total += tensor[i]
    
    # Violation 2: Element-wise mean instead of torch.mean(tensor)
    total_mean = 0
    for i in range(len(tensor)):
        total_mean += tensor[i]
    mean = total_mean / len(tensor)
    
    # Violation 3: Element-wise min instead of torch.min(tensor)
    min_value = tensor[0]
    for i in range(1, len(tensor)):
        if tensor[i] < min_value:
            min_value = tensor[i]
    
    # Violation 4: Element-wise max instead of torch.max(tensor)
    max_value = tensor[0]
    for i in range(1, len(tensor)):
        if tensor[i] > max_value:
            max_value = tensor[i]
    
    return total, mean, min_value, max_value
