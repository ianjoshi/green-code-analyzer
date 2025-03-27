import tensorflow as tf

def inefficient_broadcasting():
    # Define some sample tensors
    a = tf.constant([[1., 2.], [3., 4.]])  # Shape: (2, 2)
    b = tf.constant([[1.], [2.]])           # Shape: (2, 1)
    c = tf.constant([5., 6.])               # Shape: (2,)

    # Violation 1: Using tf.tile where broadcasting would suffice
    result1 = a + tf.tile(b, [1, 2])  # tf.tile expands b to (2, 2), but broadcasting could handle this
    print(result1)

    # Some intermediate computation
    d = result1 * 2

    # Violation 2: Another unnecessary tf.tile in a different operation
    result2 = d - tf.tile(c, [2, 1])  # tf.tile expands c to (2, 2), but broadcasting could be used
    print(result2)