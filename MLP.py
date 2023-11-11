import torch.nn as nn
import torch


class NetRelu(torch.nn.Module):
    def __init__(self, n_hidden_neurons):
        super().__init__()
        self.fc1 = nn.Linear(1, n_hidden_neurons)
        self.act_relu = nn.ReLU()
        self.fc2 = nn.Linear(n_hidden_neurons, 1)
        pass

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_relu(x)
        x = self.fc2(x)
        return x


