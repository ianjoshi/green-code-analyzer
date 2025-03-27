import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, log_loss

# Set random seed for reproducibility
np.random.seed(42)

def train_inefficiently(X_train, y_train, X_val, y_val, max_iter=200, hidden_layer_size=20):
    """
    Train a scikit-learn neural network inefficiently by manually implementing training
    with too many iterations and without proper early stopping.
    """
    # Create a neural network model
    mlp = MLPClassifier(
        hidden_layer_sizes=(hidden_layer_size,),
        activation='relu',
        solver='adam',
        learning_rate_init=0.001,
        max_iter=1,
        warm_start=True
    )
    
    # Training history
    history = {
        'val_loss': [],
        'val_accuracy': []
    }
    
    # Violation: Manual training loop without proper early stopping
    for i in range(max_iter):
        # Partial fit on one iteration
        mlp.fit(X_train, y_train)
        
        # Compute validation metrics
        y_val_pred = mlp.predict(X_val)
        y_val_prob = mlp.predict_proba(X_val)
        
        val_accuracy = accuracy_score(y_val, y_val_pred)
        val_loss = log_loss(y_val, y_val_prob)
        
        # Store metrics
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
        
        print(f"Iteration {i+1}/{max_iter} - val_loss: {val_loss:.4f} - val_accuracy: {val_accuracy:.4f}")
    
    return mlp, history

if __name__ == "__main__":
    # Generate synthetic data for demonstration
    X_train = np.random.randn(1000, 10).astype(np.float32)
    y_train = np.random.randint(0, 2, size=1000)
    
    X_val = np.random.randn(200, 10).astype(np.float32)
    y_val = np.random.randint(0, 2, size=200)
    
    # Violation: Train without early stopping
    model, history = train_inefficiently(X_train, y_train, X_val, y_val, max_iter=200)
    
    print("Training complete!")
