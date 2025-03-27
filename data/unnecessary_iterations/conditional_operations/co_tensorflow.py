import tensorflow as tf

def conditional_operation_tensorflow():
    """
    This function demonstrates the conditional operation code smell in TensorFlow.
    """
    # Create a sample tensor
    tensor = tf.random.uniform(shape=(100,), dtype=tf.float32)
    
    # Violation: Using loops to conditionally modify elements instead of using vectorized operations
    for i in range(len(tensor)):
        if tensor[i] > 0.5:
            tensor[i] = tensor[i] + 1  
        else:
            tensor[i] = tensor[i] - 1  
    
    return tensor