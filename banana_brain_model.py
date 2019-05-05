import torch
import torch.nn as nn
import torch.nn.functional as F

class BananaBrainQNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, hidden_layers, drop_p=0.5):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(BananaBrainQNetwork, self).__init__()
        # this sets the random seed for the torch library
        self.seed = torch.manual_seed(seed)
        
        # Add the first layer, input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(state_size, hidden_layers[0])])
        
        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        
        self.action = nn.Linear(hidden_layers[-1], action_size)
        
        self.dropout = nn.Dropout(p=drop_p)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = state
        # Forward through each layer in `hidden_layers`, with ReLU activation and dropout
        for linear in self.hidden_layers:
            x = F.relu(linear(x))
            x = self.dropout(x)
        return self.action(x)