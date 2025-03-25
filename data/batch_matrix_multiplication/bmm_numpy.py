import numpy as np

def inefficient_bmm_numpy(matrix_list_a, matrix_list_b):
    """
    Process pairs of matrices by multiplying them sequentially instead of batching.
    This method contains two separate violations of the BatchMatrixMultiplicationRule.
    """
    # Violation 1: Sequential matrix multiplication in a loop
    results1 = []
    for i in range(len(matrix_list_a)):
        # Inefficient: Repeated individual calls to np.matmul
        result = np.matmul(matrix_list_a[i], matrix_list_b[i])
        results1.append(result)
    
    # Some intermediate processing (to separate the violations)
    intermediate = [r + 1 for r in results1]
    
    # Violation 2: Another loop with sequential matrix multiplications
    results2 = []
    for i in range(len(matrix_list_a)):
        # Inefficient: Another set of individual np.matmul calls
        doubled = np.matmul(matrix_list_a[i], matrix_list_b[i]) * 2
        results2.append(doubled)
    
    return results1, results2