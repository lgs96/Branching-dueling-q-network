import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from agents.common.utils import identity


class Network(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, layer_num):
        """Initialization."""
        super(Network, self).__init__()

        # set common feature layer
        self.feature_layer = nn.Sequential(
            nn.Linear(in_dim, layer_num), 
            nn.ReLU(),
        )
        
        # set advantage layer
        self.advantage_layer = nn.Sequential(
            nn.Linear(layer_num, layer_num),
            nn.ReLU(),
            nn.Linear(layer_num, out_dim),
        )

        # set value layer
        self.value_layer = nn.Sequential(
            nn.Linear(layer_num, layer_num),
            nn.ReLU(),
            nn.Linear(layer_num, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        feature = self.feature_layer(x)
        
        value = self.value_layer(feature)
        advantage = self.advantage_layer(feature)

        q = value + advantage - advantage.mean(dim=-1, keepdim=True)
        
        return q    
    
class LSTM_Network(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, layer_num):
        """Initialization."""
        super(LSTM_Network, self).__init__()

        hidden_size = 64
        self.lstm_layer = nn.Sequential(
            nn.LSTM(in_dim, hidden_size)
        )
        
        # set common feature layer
        self.feature_layer = nn.Sequential(
            nn.Linear(hidden_size, layer_num), 
            nn.ReLU(),
        )
        
        # set advantage layer
        self.advantage_layer = nn.Sequential(
            nn.Linear(layer_num, layer_num),
            nn.ReLU(),
            nn.Linear(layer_num, out_dim),
        )

        # set value layer
        self.value_layer = nn.Sequential(
            nn.Linear(layer_num, layer_num),
            nn.ReLU(),
            nn.Linear(layer_num, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        state = self.lstm_layer(x)
        
        feature = self.feature_layer(state)
        
        value = self.value_layer(feature)
        advantage = self.advantage_layer(feature)

        q = value + advantage - advantage.mean(dim=-1, keepdim=True)
        
        return q  