import tensorflow as tf

def reduction_operations_tensorflow():
    """
    This function demonstrates the reduction operation code smell in TensorFlow:
    1. Using loops to compute sums instead of tf.reduce_sum.
    2. Using loops to compute means instead of tf.reduce_mean.
    3. Using loops to compute min and max instead of tf.reduce_min and tf.reduce_max.
    """
    # Create a sample tensor
    tensor = tf.random.uniform(shape=(100,), dtype=tf.float32)
    
    # Violation 1: Element-wise sum instead of tf.reduce_sum(tensor)
    total = 0
    for i in range(len(tensor)):
        total += tensor[i]
    
    # Violation 2: Element-wise mean instead of tf.reduce_mean(tensor)
    total_mean = 0
    for i in range(len(tensor)):
        total_mean += tensor[i]
    mean = total_mean / len(tensor)
    
    # Violation 3: Element-wise min instead of tf.reduce_min(tensor)
    min_value = tensor[0]
    for i in range(1, len(tensor)):
        if tensor[i] < min_value:
            min_value = tensor[i]
    
    # Violation 4: Element-wise max instead of tf.reduce_max(tensor)
    max_value = tensor[0]
    for i in range(1, len(tensor)):
        if tensor[i] > max_value:
            max_value = tensor[i]
    
    return total, mean, min_value, max_value
