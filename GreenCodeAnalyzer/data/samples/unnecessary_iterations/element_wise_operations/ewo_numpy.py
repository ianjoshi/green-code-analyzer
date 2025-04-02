import numpy as np

def element_wise_operations_numpy():
    """
    This function demonstrates the element-wise code smell in NumPy:
    1. When iterating over array elements to perform element-wise simple operations instead of using vectorized operations.
    2. When iterating over array elements to perform functions on elements instead of mapping.
    """
    # Create a sample array
    random = np.random.rand(100)
    array = np.zeros_like(random)
    
    # Violation 1: Element-wise addition instead of array-wise addition (array + 1)
    for i in range(len(array)):
        array[i] = array[i] + 1  

    # Violation 2: Element-wise subtraction instead of array-wise subtraction (array - 2)
    for i in range(len(array)):
        array[i] = array[i] - 2

    # Violation 3: Element-wise multiplication instead of array-wise multiplication (array * 3)
    for i in range(len(array)):
        array[i] = array[i] * 3

    # Violation 4: Element-wise division instead of array-wise division (array / 4)   
    for i in range(len(array)):
        array[i] = array[i] / 4

    # Violation 5: Element-wise power operation instead of array-wise power operation (array ** 2 or np.power(array, 2))
    for i in range(len(array)):
        array[i] = array[i] ** 2

    # Violation 6: Element-wise square root operation instead of array-wise square root operation (np.sqrt(array))
    for i in range(len(array)):
        array[i] = np.sqrt(array[i])

    # Violation 7: Element-wise exponential operation instead of array-wise exponential operation (np.exp(array))
    for i in range(len(array)):
        array[i] = np.exp(array[i])

    # Violation 8: Element-wise logarithm operation instead of array-wise logarithm operation (np.log(array))
    for i in range(len(array)):
        array[i] = np.log(array[i])

    # Violation 9: Element-wise trigonometric operation instead of array-wise trigonometric operation (np.sin(array))
    for i in range(len(array)):
        array[i] = np.sin(array[i])
        
    return array
