import numpy as np

def filter_operations_numpy():
    """
    This function demonstrates the filer operation code smell in NumPy.
    """
    # Create a sample array
    array = np.random.rand(100)
    
    # Violation: Using loops to filter elements instead of using array[array > 0.5]
    filtered_elements = []
    for i in range(len(array)):
        if array[i] > 0.5:
            filtered_elements.append(array[i])

    return filtered_elements
