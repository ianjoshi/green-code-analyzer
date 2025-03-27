import torch

def process_tensor_matrices(tensor_list_a, tensor_list_b):
    """
    Process pairs of tensor matrices by multiplying them sequentially instead of batching.
    This method contains two separate violations of the BatchMatrixMultiplicationRule for PyTorch.
    """
    # Violation 1: Sequential matrix multiplication in a loop
    results1 = []
    for i in range(len(tensor_list_a)):
        # Inefficient: Repeated individual calls to torch.bmm
        result = torch.bmm(tensor_list_a[i].unsqueeze(0), tensor_list_b[i].unsqueeze(0))
        results1.append(result.squeeze(0))
    
    # Some intermediate processing (to separate the violations)
    intermediate = [r + 1 for r in results1]
    
    # Violation 2: Another loop with sequential matrix multiplications
    results2 = []
    for i in range(len(tensor_list_a)):
        # Inefficient: Another set of individual torch.bmm calls
        doubled = torch.bmm(tensor_list_a[i].unsqueeze(0), tensor_list_b[i].unsqueeze(0)) * 2
        results2.append(doubled.squeeze(0))
    
    return results1, results2

# Example usage
if __name__ == "__main__":
    # Sample 2x2 tensors for demonstration (batch dimension added/removed as needed)
    a_tensors = [torch.tensor([[1.0, 2.0], [3.0, 4.0]]) for _ in range(5)]
    b_tensors = [torch.tensor([[5.0, 6.0], [7.0, 8.0]]) for _ in range(5)]
    r1, r2 = process_tensor_matrices(a_tensors, b_tensors)
    print("Results1:", r1[0])
    print("Results2:", r2[0])