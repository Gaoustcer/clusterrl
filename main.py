from torch.utils.data import DataLoader
from dataset.returntogo import Trajdataset
import torch
from clusteragent.clusteragent import Agent
torch.autograd.set_detect_anomaly(True)
if __name__ == "__main__":
    agent = Agent()
    EPOCH = 32
    for epoch in range(EPOCH):
        agent.train()
    # loader = DataLoader(Trajdataset(16),batch_size=32)
    # for obs,actions,reward,timesteps in loader:
    #     print(obs)
    #     print(len(obs))
    #     obs = torch.stack(obs,dim=-1)
    #     print(obs.shape)
    #     actions = torch.stack(actions,dim=-1)
    #     print(actions.shape)
    #     exit()
