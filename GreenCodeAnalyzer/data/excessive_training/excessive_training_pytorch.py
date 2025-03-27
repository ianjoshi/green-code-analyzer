import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class SimpleNN(nn.Module):
    """A simple neural network."""
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def train_inefficiently(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    """
    Train a model inefficiently by not implementing early stopping and continuing to train even when
    validation loss plateaus.
    """
    # Training loop violating energy efficiency - no early stopping mechanism
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        for inputs, targets in train_loader:
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Validation phase
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        val_accuracy = 100 * correct / total
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {val_loss/len(val_loader):.4f}, Val Accuracy: {val_accuracy:.2f}%')
    
    return model


if __name__ == "__main__":
    # Generate synthetic data for demonstration
    X_train = np.random.randn(1000, 10).astype(np.float32)
    y_train = np.random.randint(0, 2, size=1000).astype(np.int64)
    
    X_val = np.random.randn(200, 10).astype(np.float32)
    y_val = np.random.randint(0, 2, size=200).astype(np.int64)
    
    # Convert to PyTorch tensors
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Initialize the model, loss function, and optimizer
    model = SimpleNN(input_size=10, hidden_size=20, output_size=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Violation: Training without early stopping
    # A high number of epochs is set with no mechanism to stop early
    model = train_inefficiently(model, train_loader, val_loader, criterion, optimizer, num_epochs=100)
    
    print("Training complete!")