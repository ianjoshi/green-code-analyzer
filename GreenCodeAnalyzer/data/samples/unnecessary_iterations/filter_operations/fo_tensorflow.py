import tensorflow as tf

def filter_operations_tensorflow():
    """
    This function demonstrates the filer operation code smell in TensorFlow.
    """
    # Create a sample tensor
    tensor = tf.random.uniform(shape=(100,), dtype=tf.float32)
    
    # Violation: Using loops to filter elements instead of using boolean indexing (tensor[tensor > 0.5])
    filtered_elements = []
    for i in range(len(tensor)):
        if tensor[i] > 0.5:
            filtered_elements.append(tensor[i])

    return filtered_elements
