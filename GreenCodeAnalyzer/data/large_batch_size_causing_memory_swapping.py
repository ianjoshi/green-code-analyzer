from torch.utils.data import DataLoader
import tensorflow as tf

def inefficient_large_batch_size():
    # Define a dummy dataset for PyTorch
    pytorch_dataset = range(1000)  # Placeholder dataset
    
    # Violation 1: Large batch size in PyTorch DataLoader (keyword argument)
    train_loader = DataLoader(pytorch_dataset, batch_size=2048)
    for batch in train_loader:
        print(f"Processing PyTorch batch: {batch}")
    
    # Define a dummy TensorFlow dataset
    tf_dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5])
    
    # Violation 2: Large batch size in TensorFlow Dataset.batch (positional argument)
    batched_dataset = tf_dataset.batch(4096)
    for batch in batched_dataset:
        print(f"Processing TensorFlow batch: {batch.numpy()}")