import numpy as np

def conditional_operations_numpy():
    """
    This function demonstrates the conditional operation code smell in NumPy.
    """
    # Create a sample array
    array = np.random.rand(100)
    
    # Violation: Using loops to conditionally modify elements instead of using vectorized operations
    modified_elements = []
    for i in range(len(array)):
        if array[i] > 0.5:
            modified_elements.append(array[i] + 1) 
        else:
            modified_elements.append(array[i] - 1)  

    return modified_elements
