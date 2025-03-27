import numpy as np

def reduction_operations_numpy():
    """
    This function demonstrates the reduction operation code smell in NumPy:
    1. Using loops to compute sums instead of np.sum.
    2. Using loops to compute means instead of np.mean.
    3. Using loops to compute min and max instead of np.min and np.max.
    """
    # Create a sample array
    array = np.random.rand(100)
    
    # Violation 1: Element-wise sum instead of np.sum(array)
    total = 0
    for i in range(len(array)):
        total += array[i]
    
    # Violation 2: Element-wise mean instead of np.mean(array)
    total_mean = 0
    for i in range(len(array)):
        total_mean += array[i]
    mean = total_mean / len(array)
    
    # Violation 3: Element-wise min instead of np.min(array)
    min_value = array[0]
    for i in range(1, len(array)):
        if array[i] < min_value:
            min_value = array[i]
    
    # Violation 4: Element-wise max instead of np.max(array)
    max_value = array[0]
    for i in range(1, len(array)):
        if array[i] > max_value:
            max_value = array[i]
    
    return total, mean, min_value, max_value
