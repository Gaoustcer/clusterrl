import torch

class clusterloss(object):
    def __init__(self,emb_dim = 2,epoch = 32) -> None:
        self.embeddim = emb_dim
        self.center = torch.rand(2,self.embeddim).cuda()
        self.center = self.center/torch.norm(self.center,p = 2,dim = 1,keepdim = True)
        self.epoch = epoch
        self.alpha = 1

    def loss(self,inputs:torch.Tensor):
        '''
        inputs [N,embeddim]
        '''
        dot = torch.mm(inputs,self.center.T)
        losswithincluster = -torch.mean(torch.max(dot,dim = 1).values)
        # print("losswithincluster",losswithincluster)
        lossbetweencluster = self.center[0].dot(self.center[1])
        return losswithincluster + self.alpha * lossbetweencluster

    def norm(self):
        self.center /= torch.norm(self.center, p = 2,dim = 1,keepdim = True).detach()

    def update(self,inputs:torch.Tensor):
        self.center = inputs[:2].detach()
        '''
        inputs [N,embedding]
        '''
        for epoch in range(self.epoch):
            distance = torch.mm(inputs,self.center.T)
            maxindices = torch.max(distance,dim=1).indices
            nonzeros = maxindices.nonzero().squeeze() # with center 1
            # nonzeros = inputs[nonzeros]
            self.center[1] = torch.mean(inputs[nonzeros],dim=0).detach()
            zerosindices = (maxindices == 0).nonzero().squeeze()
            self.center[0] = torch.mean(inputs[zerosindices],dim=0).detach()
            self.norm()
        self.center[1] = torch.mean(inputs[nonzeros],dim=0)
        self.center[0] = torch.mean(inputs[zerosindices],dim = 0)
        self.norm()
            # self.loss(inputs)
        # print(self.center)
        return self.loss(inputs)
import torch.nn as nn
class Network(nn.Module):
    def __init__(self) -> None:
        super(Network,self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(2,32),
            nn.ReLU(),
            nn.Linear(32,8),
            nn.ReLU(),
            nn.Linear(8,2)
        )
    
    def forward(self,data):
        data = torch.from_numpy(data).to(torch.float32)
        return self.linear(data)
def norm(data):
    return data/torch.norm(data,p=2,dim = 1,keepdim = True)


if __name__ == "__main__":
    cluster = clusterloss()
    center1 = []
    N = 1024
    from random import random
    from math import sin,cos 
    for i in range(N):
        theta = random()/10
        center1.append((sin(theta),cos(theta)))
    center2 = []
    for i in range(N):
        theta = random()/10 + 0.2
        center2.append((sin(theta),cos(theta)))
    import numpy as np
    center1 = np.array(center1)
    center2 = np.array(center2)
    data = np.concatenate((center1,center2),axis =0)
    net = Network()
    data = net(data)
    data = norm(data)
    import matplotlib.pyplot as plt
    print(data.shape)
    plt.scatter(x = data.detach().numpy()[:,0],y = data.detach().numpy()[:,1],s = 1)
    plt.savefig("distribution.png")
    # data = torch.from_numpy(data).cuda()
    data = data.cuda()
    loss = cluster.update(data)
    loss.backward()



            
