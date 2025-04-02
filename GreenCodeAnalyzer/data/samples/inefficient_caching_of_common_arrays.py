import numpy as np

def inefficient_caching_of_common_arrays():
    # First violation: recreating the same array inside a loop
    for i in range(100):
        arr1 = np.arange(0, 10)  # Loop-invariant array creation
        result = arr1[i % 10] * 2  # Use the array in some computation
        print(result)

    # Some intermediate processing
    total = 0

    # Second violation: recreating another identical array inside a loop
    for j in range(50):
        arr2 = np.zeros(5)  # Loop-invariant array creation
        total += arr2[j % 5] + j  # Inefficient repeated creation

    return total