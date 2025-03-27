import torch 

def filter_operations_torch():
    """
    This function demonstrates the filer operation code smell in PyTorch.
    """
    # Create a sample tensor
    tensor = torch.rand(100)
    
    # Violation: Using loops to filter elements instead of using boolean indexing (tensor[tensor > 0.5])
    filtered_elements = []
    for i in range(len(tensor)):
        if tensor[i] > 0.5:
            filtered_elements.append(tensor[i])

    return filtered_elements
