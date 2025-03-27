import torch

def conditional_operation_pytorch():
    """
    This function demonstrates the conditional operation code smell in PyTorch.
    """
    # Create a sample tensor
    tensor = torch.rand(100)

    # Violation: Using loops to conditionally modify elements instead of using vectorized operations
    for i in range(len(tensor)):
        if tensor[i] > 0.5:
            tensor[i] = tensor[i] + 1  
        else:
            tensor[i] = tensor[i] - 1  
    
    return tensor 