import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import torch.nn.functional as F

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class AgentRL_attention(nn.Module):
    def __init__(self, dim_obs, dim_action):
        super(AgentRL_attention, self).__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(dim_obs, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        
        self.actor = nn.Sequential(
            layer_init(nn.Linear(dim_obs, 64)),
            nn.Tanh(),
            multi_head(),
            nn.Tanh(),
            multi_head(),
            nn.Tanh(),
            layer_init(nn.Linear(64,64)),
            nn.Tanh(),
            layer_init(nn.Linear(64,dim_action),std=0.01),
        )
        

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        if torch.isnan(logits).sum().item() > 0:
            print('he')
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)



class multi_head(nn.Module):
    def __init__(self):
        super(multi_head, self).__init__()
        self.A = nn.ModuleList([nn.Linear(32, 1) for _ in range(4)])
        self.weight_init()

    def weight_init(self):
        for i in range(2):
            nn.init.xavier_normal_(self.A[i].weight)
            self.A[i].bias.data.fill_(0.0)
    
    def attn_summary(self, features):
        features_attn = []
        for i in range(2):
            features_attn.append((self.A[i](features[i].squeeze())))
        features_attn = F.softmax(torch.cat(features_attn), dim=-1).unsqueeze(1)
        features = torch.cat(features)
        features = (features * features)
        features = features.reshape(-1,64)
        return features, features_attn
    
    def forward(self,x):
        # x1,x2,x3,x4 = x[:,:16], x[:,16:32], x[:,32:48], x[:,48:]

        x1,x2 = x[:,:32], x[:,32:]

        # results, _ = self.attn_summary([x1,x2,x3,x4])
        results, _ = self.attn_summary([x1,x2])

        return results
        
    