import torch
import torch.nn as nn


class DQN(nn.Module):
  def __init__(self, input_dimension, output_dimension, device):
    super().__init__()
    print("dimension", input_dimension)
    self.device = device
    # Convolutional layers (observation layer)
    self.conv_layers = nn.Sequential(
      nn.Conv2d(input_dimension, 16, kernel_size=8, stride=2, padding=0),
      nn.ReLU(),
            
      # nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
      # nn.ReLU(),
          
      # nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=0),
      # nn.ReLU(),
      
      nn.Conv2d(16, 4, kernel_size=4, stride=1, padding=0),
      nn.ReLU(),
      nn.Flatten()
    )

    #can amke this a function but i decided to hardcode it the math is here
    # flattened size: 64 * 7 * 7 = 3136
    # layer 1: (84-8)/4+1 = 20
    # layer 2: (20-4)/2+1 = 9  
    # layer 3: (9-3)/1+1 = 7
    # final: 64 channels * 7 height * 7 width = 3136
      
    # Linear Layers (hidden and action layers)
    self.fc_layers = nn.Sequential(
      nn.Linear(5336 + 0, 1024),  
      nn.ReLU(),
      nn.Linear(1024, 512),   
      # nn.ReLU(),
      # nn.Linear(512, 512),  
      nn.ReLU(),
      nn.Linear(512, output_dimension) 
    )

    #self.stack = nn.Sequential(
    #  nn.Linear(input_dimension, 128), # Observation layer
    #  nn.ReLU(),
    #  nn.Linear(128, 128), # Hidden Layer
    #  nn.ReLU(),
    #  nn.Linear(128, output_dimension) # Action Layer
    #)

    #i seperated the observation layer from the hidden and action layer just bc it looks
    #cleaner and helped me undertsand it better

    
  def forward(self, x
              # , fc_data
              ):
    features = self.conv_layers(x).to(self.device)
    # features = torch.cat(features, fc_data).to(self.device)
    return self.fc_layers(features).to(self.device)

