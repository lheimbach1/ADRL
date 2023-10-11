import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch

class CNN_Net(nn.Module):
    def __init__(self, state_shape, action_shape = None, device='cpu'):
        super(CNN_Net, self).__init__()
        self.input_dim  = int(np.prod(state_shape))
        self.device = device
        self.hidden_sizes = []

        self.conv1 = nn.Conv1d(1, 32, 5) # 1 input channel, 32 output channels, kernel size of 5
        self.pool1 = nn.MaxPool1d(2)  # Let's use a pooling kernel of size 2 for simplicity. You can adjust.
        self.conv2 = nn.Conv1d(32, 64, 5) # 32 input channels, 64 output channels, kernel size of 5
        self.pool2 = nn.AdaptiveAvgPool1d(32)
        self.output_dim = 64 * 32  # Depending on the pooling operation, this may need adjustments.

    def forward(self, obs, state = None, info = {}):
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float().to(self.device)
        if len(obs.shape) == 2:
            obs = obs.unsqueeze(1)

        obs = F.relu(self.conv1(obs))
        obs = self.pool1(obs)
        obs = F.relu(self.conv2(obs))
        obs = self.pool2(obs)
        logits = torch.flatten(obs, 1) # flatten all dimensions except batch
        return logits, state
