"""Q-network for the Banana navigation task."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Maps a 37-dim state to one Q-value per action.

    With dueling=True the head is split into a state-value stream V(s) and an
    advantage stream A(s, a), recombined as Q = V + (A - mean(A)).
    """

    def __init__(self, state_size, action_size, seed=0,
                 fc1_units=64, fc2_units=64, dueling=False):
        super(QNetwork, self).__init__()
        torch.manual_seed(seed)
        self.dueling = dueling
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        if dueling:
            self.value = nn.Linear(fc2_units, 1)
            self.advantage = nn.Linear(fc2_units, action_size)
        else:
            self.out = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        if self.dueling:
            v = self.value(x)
            a = self.advantage(x)
            return v + a - a.mean(dim=1, keepdim=True)
        return self.out(x)
