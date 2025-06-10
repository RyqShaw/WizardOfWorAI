import torch
import torch.nn as nn


class DQN(nn.Module):
  def __init__(self, input_dimension, output_dimension):
    super.__init__()
    self.stack = nn.Sequential(
      nn.Linear(input_dimension, 128), # Observation layer
      nn.ReLU(),
      nn.Linear(128, 128), # Hidden Layer
      nn.ReLU(),
      nn.Linear(128, output_dimension) # Action Layer
    )
    
  def forward(self, x):
    return self.stack(x)
