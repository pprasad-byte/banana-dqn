"""DQN agent with experience replay, a soft-updated target network, and
optional Double DQN / Dueling DQN."""
import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from model import QNetwork

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # soft update rate for the target network
LR = 5e-4               # learning rate
UPDATE_EVERY = 4        # learn every N environment steps

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent(object):
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed=0,
                 double=False, dueling=False):
        self.state_size = state_size
        self.action_size = action_size
        self.double = double
        self.dueling = dueling
        random.seed(seed)

        self.qnetwork_local = QNetwork(state_size, action_size, seed,
                                       dueling=dueling).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed,
                                        dueling=dueling).to(device)
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, seed)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        """Store a transition and learn every UPDATE_EVERY steps."""
        self.memory.add(state, action, reward, next_state, done)
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0 and len(self.memory) > BATCH_SIZE:
            self.learn(self.memory.sample(), GAMMA)

    def act(self, state, eps=0.0):
        """Epsilon-greedy action for the given state."""
        if random.random() < eps:
            return random.randrange(self.action_size)
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        return int(action_values.argmax(dim=1).item())

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        with torch.no_grad():
            if self.double:
                # Double DQN: local net selects the action, target net evaluates it
                best_actions = self.qnetwork_local(next_states).argmax(1, keepdim=True)
                q_next = self.qnetwork_target(next_states).gather(1, best_actions)
            else:
                q_next = self.qnetwork_target(next_states).max(1, keepdim=True)[0]
            q_targets = rewards + gamma * q_next * (1 - dones)

        q_expected = self.qnetwork_local(states).gather(1, actions)
        loss = F.mse_loss(q_expected, q_targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    @staticmethod
    def soft_update(local_model, target_model, tau):
        """theta_target = tau * theta_local + (1 - tau) * theta_target"""
        for target_param, local_param in zip(target_model.parameters(),
                                             local_model.parameters()):
            target_param.data.copy_(tau * local_param.data +
                                    (1.0 - tau) * target_param.data)

    def save(self, path):
        torch.save({"state_dict": self.qnetwork_local.state_dict(),
                    "double": self.double,
                    "dueling": self.dueling}, path)


class ReplayBuffer(object):
    """Fixed-size buffer of experience tuples, sampled uniformly."""

    def __init__(self, buffer_size, batch_size, seed):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple(
            "Experience", ["state", "action", "reward", "next_state", "done"])
        random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        self.memory.append(self.experience(state, action, reward, next_state, done))

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8)).float().to(device)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)
