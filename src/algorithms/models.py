import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical
    
class ActorCritic(nn.Module):
    def __init__(self, config, input_size, output_size):
        super(ActorCritic, self).__init__()
        self.config = config
        nn_number = self.config.nn_number
        
        if self.config.env == "LunarLander":
            self.actor = nn.Sequential(
                nn.Linear(input_size, nn_number),
                nn.Tanh(),
                nn.Linear(nn_number, nn_number),
                nn.Tanh(),
                nn.Linear(nn_number, output_size),
                nn.Softmax(dim=-1)
                )
        elif self.config.env == "RocketLeague":
            self.actor = nn.Sequential(
                nn.Linear(input_size, nn_number),
                nn.Tanh(),
                nn.Linear(nn_number, nn_number),
                nn.Tanh(),
                )
            self.action_heads = []
            for i in range(5):
                self.action_heads.append(nn.Linear(nn_number, 3))
            for i in range(3):
                self.action_heads.append(nn.Linear(nn_number, 2))
                
            with torch.no_grad():
                """
                act[0] = throttle #[-1, 1] continuous
                act[1] = steer #[-1, 1] continuous
                act[2] = pitch #[-1, 1] continuous
                act[3] = yaw #[-1, 1] continuous
                act[4] = roll #[-1, 1] continuous
                act[5] = jump #{0, 1} discrete
                act[6] = boost #{0, 1} discrete
                act[7] = handbrake #{0, 1} discrete
                """
                self.action_heads[0].bias = torch.nn.Parameter(torch.Tensor([0., 0., 1.]))
                self.action_heads[1].bias = torch.nn.Parameter(torch.Tensor([0., 1., 0.]))
                self.action_heads[2].bias = torch.nn.Parameter(torch.Tensor([0., 1., 0.]))
                self.action_heads[3].bias = torch.nn.Parameter(torch.Tensor([0., 1., 0.]))
                self.action_heads[4].bias = torch.nn.Parameter(torch.Tensor([0., 1., 0.]))
                self.action_heads[5].bias = torch.nn.Parameter(torch.Tensor([1., 0.]))
                self.action_heads[6].bias = torch.nn.Parameter(torch.Tensor([1., 0.]))
                self.action_heads[7].bias = torch.nn.Parameter(torch.Tensor([1., 0.]))
        
        self.critic = nn.Sequential(
            nn.Linear(input_size, nn_number),
            nn.Tanh(),
            nn.Linear(nn_number, nn_number),
            nn.Tanh(),
            nn.Linear(nn_number, 1)
            )

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        if self.config.env == "LunarLander":
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
            action = dist.sample()
            action_logprob = dist.log_prob(action)
                                           
        elif self.config.env == "RocketLeague":
            action = []
            action_logprob = []
            for action_head in self.action_heads:
                probs = F.softmax(action_head(self.actor(state)), dim=-1)
                dist = Categorical(probs)
                act = dist.sample()
                act_logprob = dist.log_prob(act)
                action.append(act)
                action_logprob.append(act_logprob)
                
            action = torch.stack(action)
            action_logprob = torch.stack(action_logprob)
        
        return action, action_logprob

    def evaluate(self, state, action):        
        if self.config.env == "LunarLander":            
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
            action_logprobs = dist.log_prob(action)                                           
            dist_entropy = dist.entropy()
            
        elif self.config.env == "RocketLeague":
            action_logprobs = []
            dist_entropy = []
            for i, action_head in enumerate(self.action_heads):
                probs = F.softmax(action_head(self.actor(state)), dim=-1)
                dist = Categorical(probs)
                act_logprobs = dist.log_prob(action[:,i])
                action_logprobs.append(act_logprobs)
                dist_entropy.append(dist.entropy())
                
            action_logprobs = torch.transpose(torch.stack(action_logprobs), 0, 1)
            dist_entropy = torch.transpose(torch.stack(dist_entropy), 0, 1)
                
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy
        
        
            