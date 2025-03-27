import torch.nn as nn
class DataParallel():
    def __init__(self, model):
        self.model = model

def inefficient_data_parallelization():
    
    # Create a simple neural network
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 2)
    )
    
    # Inefficient: Using DataParallel without proper batch size consideration
    model = nn.DataParallel(model)
    
    dont_highlight = DataParallel(model)
    
    return model

