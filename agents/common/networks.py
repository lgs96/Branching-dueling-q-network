import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from agents.common.utils import identity


"""
DQN, DDQN, A2C critic, VPG critic, TRPO critic, PPO critic, DDPG actor, TD3 actor
"""
class MLP(nn.Module):
    def __init__(self, 
                 input_size, 
                 output_size, 
                 hidden_sizes=(32,32), 
                 activation=F.relu, 
                 output_activation=identity,
                 use_output_layer=True,
    ):
        super(MLP, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.output_activation = output_activation
        self.use_output_layer = use_output_layer

        # Set hidden layers
        self.hidden_layers = nn.ModuleList()
        in_size = self.input_size
        for next_size in self.hidden_sizes:
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            self.hidden_layers.append(fc)

        # Set output layers
        if self.use_output_layer:
            self.output_layer = nn.Linear(in_size, self.output_size)
        else:
            self.output_layer = identity

    def forward(self, x):
        for hidden_layer in self.hidden_layers:
            x = self.activation(hidden_layer(x))
        x = self.output_activation(self.output_layer(x))
        return x

class Network(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        """Initialization."""
        super(Network, self).__init__()

        # set common feature layer
        self.feature_layer = nn.Sequential(
            nn.Linear(in_dim, 128), 
            nn.ReLU(),
        )
        
        # set advantage layer
        self.advantage_layer = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
        )

        # set value layer
        self.value_layer = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        feature = self.feature_layer(x)
        
        value = self.value_layer(feature)
        advantage = self.advantage_layer(feature)

        q = value + advantage - advantage.mean(dim=-1, keepdim=True)
        
        return q    
    
"""
A2C actor
"""
class CategoricalPolicy(MLP):
    def forward(self, x):
        x = super(CategoricalPolicy, self).forward(x)
        pi = F.softmax(x, dim=-1)

        dist = Categorical(pi)
        action = dist.sample()
        log_pi = dist.log_prob(action)
        return action, pi, log_pi


"""
DDPG critic, TD3 critic, SAC qf, TAC qf
"""
class FlattenMLP(MLP):
    def forward(self, x, a):
        q = torch.cat([x,a], dim=-1)
        return super(FlattenMLP, self).forward(q)


"""
VPG actor, TRPO actor, PPO actor
"""
class GaussianPolicy(MLP):
    def __init__(self, 
                 input_size, 
                 output_size, 
                 hidden_sizes=(64,64),
                 activation=torch.tanh,
    ):
        super(GaussianPolicy, self).__init__(
            input_size=input_size,
            output_size=output_size,
            hidden_sizes=hidden_sizes,
            activation=activation,
        )

    def forward(self, x, pi=None):
        mu = super(GaussianPolicy, self).forward(x)
        log_std = torch.zeros_like(mu)
        std = torch.exp(log_std)
        
        dist = Normal(mu, std)
        if pi == None:
            pi = dist.sample()
        log_pi = dist.log_prob(pi).sum(dim=-1)
        return mu, std, pi, log_pi


"""
SAC actor, TAC actor
"""
LOG_STD_MAX = 2
LOG_STD_MIN = -20

class ReparamGaussianPolicy(MLP):
    def __init__(self, 
                 input_size, 
                 output_size,
                 hidden_sizes=(64,64),
                 activation=F.relu,
                 action_scale=1.0,
                 log_type='log',
                 q=1.5,
                 device=None, 
    ):
        super(ReparamGaussianPolicy, self).__init__(
            input_size=input_size,
            output_size=output_size,
            hidden_sizes=hidden_sizes,
            activation=activation,
            use_output_layer=False,
        )

        in_size = hidden_sizes[-1]
        self.action_scale = action_scale
        self.log_type = log_type
        self.q = 2.0 - q
        self.device = device

        # Set output layers
        self.mu_layer = nn.Linear(in_size, output_size)
        self.log_std_layer = nn.Linear(in_size, output_size)        

    def clip_but_pass_gradient(self, x, l=-1., u=1.):
        clip_up = (x > u).float()
        clip_low = (x < l).float()
        clip_value = (u - x)*clip_up + (l - x)*clip_low
        return x + clip_value.detach()

    def apply_squashing_func(self, mu, pi, log_pi):
        mu = torch.tanh(mu)
        pi = torch.tanh(pi)
        # To avoid evil machine precision error, strictly clip 1-pi**2 to [0,1] range.
        if self.log_type == 'log':
            log_pi -= torch.sum(torch.log(self.clip_but_pass_gradient(1 - pi.pow(2), l=0., u=1.) + 1e-6), dim=-1)
        elif self.log_type == 'log-q':
            log_pi -= torch.log(self.clip_but_pass_gradient(1 - pi.pow(2), l=0., u=1.) + 1e-6)
        return mu, pi, log_pi

    def tsallis_entropy_log_q(self, x, q):
        safe_x = torch.max(x, torch.Tensor([1e-6]).to(self.device))
        log_q_x = torch.log(safe_x) if q==1. else (safe_x.pow(1-q)-1)/(1-q)
        return log_q_x.sum(dim=-1)
        
    def forward(self, x):
        x = super(ReparamGaussianPolicy, self).forward(x)
        
        mu = self.mu_layer(x)
        log_std = torch.tanh(self.log_std_layer(x))
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        std = torch.exp(log_std)
        
        # https://pytorch.org/docs/stable/distributions.html#normal
        dist = Normal(mu, std)
        pi = dist.rsample() # reparameterization trick (mean + std * N(0,1))

        if self.log_type == 'log':
            log_pi = dist.log_prob(pi).sum(dim=-1)
            mu, pi, log_pi = self.apply_squashing_func(mu, pi, log_pi)
        elif self.log_type == 'log-q':
            log_pi = dist.log_prob(pi)
            mu, pi, log_pi = self.apply_squashing_func(mu, pi, log_pi)
            exp_log_pi = torch.exp(log_pi)
            log_pi = self.tsallis_entropy_log_q(exp_log_pi, self.q)
        
        # make sure actions are in correct range
        mu = mu * self.action_scale
        pi = pi * self.action_scale
        return mu, pi, log_pi