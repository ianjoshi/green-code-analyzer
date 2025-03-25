import tensorflow as tf

def inefficient_bmm_tensorflow(tensor_list_a, tensor_list_b):
    """
    Process pairs of TensorFlow matrices by multiplying them sequentially instead of batching.
    This method contains two separate violations of the BatchMatrixMultiplicationRule for TensorFlow.
    """
    # Violation 1: Sequential matrix multiplication in a loop
    results1 = []
    for i in range(len(tensor_list_a)):
        # Inefficient: Repeated individual calls to tf.linalg.matmul
        result = tf.linalg.matmul(tensor_list_a[i], tensor_list_b[i])
        results1.append(result)
    
    # Some intermediate processing (to separate the violations)
    _ = [r + 1 for r in results1]
    
    # Violation 2: Another loop with sequential matrix multiplications
    results2 = []
    for i in range(len(tensor_list_a)):
        # Inefficient: Another set of individual tf.linalg.matmul calls
        doubled = tf.linalg.matmul(tensor_list_a[i], tensor_list_b[i]) * 2
        results2.append(doubled)
    
    return results1, results2