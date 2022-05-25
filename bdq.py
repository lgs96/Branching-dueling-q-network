import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import random
from torch.optim import lr_scheduler

from agents.common.utils import *
from agents.common.buffers import *
from agents.common.networks import *
from agents.common.dueling import *

class BranchingQNetwork(nn.Module):

    def __init__(self, obs: int, ac_dim: int, n: int): 

        super().__init__()

        self.ac_dim = ac_dim
        self.n = n 

        self.model = nn.Sequential(nn.Linear(obs, 512), 
                                   nn.ReLU(),
                                   nn.Linear(512, 256), 
                                   nn.ReLU())

        self.value_head = nn.Sequential(nn.Linear(256, 128),
                                        nn.ReLU(),
                                        nn.Linear(128, 1))
        
        self.adv_heads = nn.ModuleList([nn.Sequential(nn.Linear(256, 128),
                                        nn.ReLU(),
                                        nn.Linear(128, n)) for i in range(ac_dim)])

    def forward(self, x): 

        out = self.model(x)
        value = self.value_head(out)
        advs = torch.stack([l(out) for l in self.adv_heads], dim = 1)
        
        q_val = value.unsqueeze(2) + advs - advs.mean(2, keepdim = True )

        return q_val

class Agent(object):
   """An implementation of the Deep Q-Network (DQN), Double DQN agents."""

   def __init__(self,
                env,
                args,
                device,
                obs_dim,
                act_dim,
                act_num,
                steps=0,
                gamma=0.99,
                epsilon=1.0,
                epsilon_decay=0.995,
                buffer_size=int(1e4),
                batch_size=64,
                target_update_step=100,
                eval_mode=False,
                q_losses=list(),
                logger=dict(),
   ):

      self.env = env
      self.args = args
      self.device = device
      self.obs_dim = obs_dim
      self.act_dim = act_dim
      self.act_num = act_num
      self.steps = steps
      self.gamma = gamma
      self.epsilon = epsilon
      self.epsilon_decay = epsilon_decay
      self.buffer_size = buffer_size
      self.batch_size = batch_size
      self.target_update_step = target_update_step
      self.eval_mode = eval_mode
      self.q_losses = q_losses
      self.logger = logger

      # Main network
      self.qf = BranchingQNetwork(self.obs_dim, self.act_dim, self.act_num).to(self.device)
      # Target network
      self.qf_target = MLP(self.obs_dim, self.act_dim, self.act_num).to(self.device)
      
      # Initialize target parameters to match main parameters
      hard_target_update(self.qf, self.qf_target)

      # Create an optimizer
      self.qf_optimizer = optim.Adam(self.qf.parameters(), lr=1e-3)

      # Experience buffer
      self.replay_buffer = ReplayBuffer(self.obs_dim, self.act_dim, self.buffer_size, self.device)

   def select_action(self, obs):
      """Select an action from the set of available actions."""
      # Decaying epsilon
      self.epsilon *= self.epsilon_decay
      self.epsilon = max(self.epsilon, 0.01)

      if np.random.rand() <= self.epsilon:
         # Choose a random action with probability epsilon
         return np.random.randint(self.act_num)
      else:
        # Choose the action with highest Q-value at the current state
         q_value = self.qf(obs)
         q_argmax = torch.argmax(q_value.squeeze(0), dim = 1)
         return q_argmax.detach().cpu().numpy()

   def train_model(self):
      batch = self.replay_buffer.sample(self.batch_size)
      obs1 = batch['obs1']
      obs2 = batch['obs2']
      acts = batch['acts']
      rews = batch['rews']
      done = batch['done']

      if 0: # Check shape of experiences
         print("obs1", obs1.shape)
         print("obs2", obs2.shape)
         print("acts", acts.shape)
         print("rews", rews.shape)
         print("done", done.shape)

      # Prediction Q(s)
      q = self.qf(obs1).gather(self.act_dim, acts.long()).squeeze(1)
      
      # Target for Q regression
      if self.args.algo == 'dqn':      # DQN
         q_target = self.qf_target(obs2)
      elif self.args.algo == 'ddqn':   # Double DQN
         q2 = self.qf(obs2)
         q_target = self.qf_target(obs2)
         q_target = q_target.gather(1, q2.max(1)[1].unsqueeze(1))
      q_backup = rews + self.gamma*(1-done)*q_target.max(1)[0]
      q_backup.to(self.device)

      if 0: # Check shape of prediction and target
         print("q", q.shape)
         print("q_backup", q_backup.shape)

      # Update perdiction network parameter
      qf_loss = F.mse_loss(q, q_backup.detach())
      self.qf_optimizer.zero_grad()
      qf_loss.backward()
      for param in self.qf.parameters():
        param.grad *= 1/(self.act_dim + 1)
        param.grad.data.clamp_(-1, 1)
      self.qf_optimizer.step()

      # Synchronize target parameters 𝜃‾ as 𝜃 every C steps
      if self.steps % self.target_update_step == 0:
         hard_target_update(self.qf, self.qf_target)
      
      # Save loss
      self.q_losses.append(qf_loss.item())

   def run(self, max_step):
      step_number = 0
      total_reward = 0.

      obs = self.env.reset()
      done = False

      # Keep interacting until agent reaches a terminal state.
      while not (done or step_number == max_step):
         if self.args.render:
            self.env.render()       
      
         if self.eval_mode:
            q_value = self.qf(torch.Tensor(obs).to(self.device))
            q_argmax = torch.argmax(q_value.squeeze(0), dim = 1)
            action = q_argmax.detach().cpu().numpy()    
            next_obs, reward, done, _ = self.env.step(action)
         else:
            self.steps += 1

            # Collect experience (s, a, r, s') using some policy
            action = self.select_action(torch.Tensor(obs).to(self.device))
            next_obs, reward, done, _ = self.env.step(action)
         
            # Add experience to replay buffer
            self.replay_buffer.add(obs, action, reward, next_obs, done)
            
            # Start training when the number of experience is greater than batch_size
            if self.steps > self.batch_size:
               self.train_model()

         total_reward += reward
         step_number += 1
         obs = next_obs
      
      # Save logs
      self.logger['LossQ'] = round(np.mean(self.q_losses), 5)
      return step_number, total_reward
