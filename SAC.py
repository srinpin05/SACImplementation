import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

#first time using torch, so READ THE DOCS!
alpha = 0.2  
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.std = nn.Linear(hidden_dim, action_dim)
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        std = self.std(x)
        return mean, std
    def sample(self, state):
        mean, std = self.forward(state)
        normal = Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob
    def loss_function(self, state, action, q1, q2, log_prob):
        return (min(q1.value(state, action), q2(state, action)) - alpha*log_prob).mean()

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def value(self, state, action):
        return self.forward(state, action)

class SAC:
    def __init__(self, state_dim, action_dim, hidden_dim, lr):
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim)
        self.q1 = QNetwork(state_dim, action_dim, hidden_dim)
        self.q2 = QNetwork(state_dim, action_dim, hidden_dim)
        self.target_q1 = QNetwork(state_dim, action_dim, hidden_dim)
        self.target_q2 = QNetwork(state_dim, action_dim, hidden_dim)
        #set optimizers for each network for gradient descent/ascent

        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_opitmizer = optim.Adam(self.q2.parameters(), lr=lr)
        
        #initialize target networks with same parameters as main networks
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())
    def compute_Qtargets(self, next_state, reward, done, discount):
        with torch.no_grad():
            next_action, next_action_log_prob = self.policy.sample(next_state)
            target_q1_value = self.target_q1.value(next_state,next_action)
            target_q2_value = self.target_q2.value(next_state, next_action)
            target_q_value = reward + discount * (1 - done)*(torch.min(target_q1_value, target_q2_value) - alpha * next_action_log_prob)
        return target_q_value
    def compute_qloss()