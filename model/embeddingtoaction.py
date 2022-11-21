import torch.nn as nn

import gym

class MLP(nn.Module):
    def __init__(self,env:gym.Env,embed_dim) -> None:
        super(MLP,self).__init__()
        self.actiondim = len(env.action_space.sample())
        self.embeddim = embed_dim
        self.MLP = nn.Sequential(
            nn.Linear(self.embeddim,32),
            nn.ReLU(),
            nn.Linear(32,self.actiondim),
            nn.Tanh()
        )
    
    def forward(self,stateembedding):
        return self.MLP(stateembedding)
    