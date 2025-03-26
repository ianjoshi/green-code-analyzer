import torch

def element_wise_operations_pytorch():
    """
    This function demonstrates the element-wise code smell in PyTorch:
    1. When iterating over tensor elements to perform element-wise simple operations instead of using vectorized operations.
    2. When iterating over tensor elements to perform functions on elements instead of mapping.
    """
    # Create a sample tensor
    random_tensor = torch.rand(100)
    tensor = torch.zeros_like(random_tensor)
    
    # Violation 1: Element-wise addition instead of tensor-wise addition (tensor + 1)
    for i in range(len(tensor)):
        tensor[i] = tensor[i] + 1  

    # Violation 2: Element-wise subtraction instead of tensor-wise subtraction (tensor - 2)
    for i in range(len(tensor)):
        tensor[i] = tensor[i] - 2

    # Violation 3: Element-wise multiplication instead of tensor-wise multiplication (tensor * 3)
    for i in range(len(tensor)):
        tensor[i] = tensor[i] * 3

    # Violation 4: Element-wise division instead of tensor-wise division (tensor / 4)   
    for i in range(len(tensor)):
        tensor[i] = tensor[i] / 4

    # Violation 5: Element-wise power operation instead of tensor-wise power operation (tensor ** 2)
    for i in range(len(tensor)):
        tensor[i] = tensor[i] ** 2

    # Violation 6: Element-wise square root operation instead of tensor-wise square root operation (torch.sqrt(tensor))
    for i in range(len(tensor)):
        tensor[i] = torch.sqrt(tensor[i])

    # Violation 7: Element-wise exponential operation instead of tensor-wise exponential operation (torch.exp(tensor))
    for i in range(len(tensor)):
        tensor[i] = torch.exp(tensor[i])

    # Violation 8: Element-wise logarithm operation instead of tensor-wise logarithm operation (torch.log(tensor))
    for i in range(len(tensor)):
        tensor[i] = torch.log(tensor[i])

    # Violation 9: Element-wise trigonometric operation instead of tensor-wise trigonometric operation (torch.sin(tensor))
    for i in range(len(tensor)):
        tensor[i] = torch.sin(tensor[i])
        
    return tensor
