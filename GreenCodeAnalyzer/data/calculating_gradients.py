import torch.nn as nn
import tensorflow as tf

def inefficient_gradient_calculations(input_data, tf_input_data):
    # PyTorch model setup
    pytorch_model = nn.Linear(10, 2)
    
    # Violation 1: PyTorch model call without torch.no_grad() during inference
    pytorch_output = pytorch_model(input_data)
    
    # TensorFlow model setup
    tf_model = tf.keras.models.Sequential([tf.keras.layers.Dense(2)])
    
    # Violation 2: TensorFlow model call inside tf.GradientTape() during inference
    with tf.GradientTape() as tape:
        tf_output = tf_model(tf_input_data)
    
    # Combine outputs (no gradient computation involved)
    return pytorch_output, tf_output