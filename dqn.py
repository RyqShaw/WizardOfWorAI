import torch
import torch.nn as nn


class DQN(nn.Module):
  def __init__(self, input_dimension, output_dimension):
    super.__init__()
    self.stack = nn.Sequential(
      nn.Linear(input_dimension, 128),
      nn.ReLU(),
      # start with 128 (smaller side)
      nn.Linear(128, output_dimension)
      
    )
    
