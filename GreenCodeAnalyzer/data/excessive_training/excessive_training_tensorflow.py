import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

def create_model():
    """Create a simple neural network model"""
    model = Sequential([
        Dense(20, activation='relu', input_shape=(10,)),
        Dense(2, activation='softmax')
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_inefficiently(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
    """
    Train TensorFlow model inefficiently without early stopping.
    """
    # Training loop violation - no early stopping mechanism
    history = {"val_loss": [], "val_accuracy": []}
    
    for epoch in range(epochs):
        # Train the model for one epoch
        model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=1,
            verbose=0
        )
        
        # Evaluate on validation data
        val_results = model.evaluate(X_val, y_val, verbose=0)
        val_loss, val_accuracy = val_results
        
        # Store metrics
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        
        print(f'Epoch {epoch+1}/{epochs} - val_loss: {val_loss:.4f} - val_accuracy: {val_accuracy:.4f}')
    
    return model, history

if __name__ == "__main__":
    # Generate synthetic data
    X_train = np.random.randn(1000, 10).astype(np.float32)
    y_train = np.random.randint(0, 2, size=1000).astype(np.int32)
    
    X_val = np.random.randn(200, 10).astype(np.float32)
    y_val = np.random.randint(0, 2, size=200).astype(np.int32)
    
    # Create and compile the model
    model = create_model()
    
    # Violation: Train without early stopping
    # This wastes energy by continuing to train after the model has converged
    model, history = train_inefficiently(model, X_train, y_train, X_val, y_val, epochs=100)
    
    print("Training complete!")
