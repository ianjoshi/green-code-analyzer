import tensorflow as tf

def process_tf_matrices(tensor_list_a, tensor_list_b):
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
    intermediate = [r + 1 for r in results1]
    
    # Violation 2: Another loop with sequential matrix multiplications
    results2 = []
    for i in range(len(tensor_list_a)):
        # Inefficient: Another set of individual tf.linalg.matmul calls
        doubled = tf.linalg.matmul(tensor_list_a[i], tensor_list_b[i]) * 2
        results2.append(doubled)
    
    return results1, results2

# Example usage
if __name__ == "__main__":
    # Sample 2x2 tensors for demonstration
    a_tensors = [tf.constant([[1.0, 2.0], [3.0, 4.0]]) for _ in range(5)]
    b_tensors = [tf.constant([[5.0, 6.0], [7.0, 8.0]]) for _ in range(5)]
    r1, r2 = process_tf_matrices(a_tensors, b_tensors)
    print("Results1:", r1[0])
    print("Results2:", r2[0])