from clusterloss.cluster import clusterloss
from dataset.returntogo import Trajdataset
from torch.utils.data import DataLoader
from model.embeddingtoaction import MLP
from model.squencemodel import SequenceModel
import torch
import torch.nn as nn
import d4rl
import gym
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import torch.nn.functional as F

class Agent(object):
    def __init__(self,envname = "hopper-medium-v2",embed_dim = 2,Traj_length = 32) -> None:
        self.env = gym.make(envname)
        self.embeddim = embed_dim
        self.Trajlength = Traj_length
        self.averagecost = self.getaveragereturn()
        self.Seq2embedding = SequenceModel(self.env,embed_dim=self.embeddim).cuda()
        self.actiondecoder = MLP(self.env,embed_dim=self.embeddim).cuda()
        self.clusterloss = clusterloss(emb_dim=self.embeddim)
        self.index = 0
        self.writer = SummaryWriter("./logs/clustertransformer{}".format(envname))
        self.optimSeq = torch.optim.Adam(self.Seq2embedding.parameters(),lr = 0.0001)
        self.optimdecoder = torch.optim.Adam(self.actiondecoder.parameters(),lr = 0.0001)
        self.loader = DataLoader(Trajdataset(trajlengh=self.Trajlength),batch_size=1024)

    def getaveragereturn(self):
        dataset = d4rl.qlearning_dataset(self.env)
        terminalcount = sum(dataset['terminals'])
        totalcost = sum(dataset['rewards'])
        return totalcost/terminalcount

    def normal(self,input_tensor):
        norm = torch.norm(input_tensor,p=2,dim=-1,keepdim=True)
        return input_tensor/norm
    def train(self):
        from tqdm import tqdm
        for obs,actions,rewards,timesteps in tqdm(self.loader):
            self.optimdecoder.zero_grad()
            self.optimSeq.zero_grad()
            # print('obs0 shape',obs[0].shape)

            obs = torch.stack(obs,dim = 1).cuda()
            actions = torch.stack(actions,dim=1).cuda()
            # print("obs shape is",obs.shape)
            # print('action shape is',actions.shape)
            # exit()
            timesteps = timesteps.cuda()
            clusterembedding = self.Seq2embedding(obs,actions,timesteps)
            normalembedding = self.normal(clusterembedding)
            clusterloss = self.clusterloss.update(normalembedding)
            
            self.plot(normalembedding)
            predactions = self.actiondecoder(clusterembedding)
            currentactions = actions[:,-1,:]
            actionloss = F.mse_loss(currentactions,predactions)
            # loss = clusterloss + actionloss
            loss = actionloss + clusterloss
            print(loss)
            loss.backward()
            self.optimdecoder.step()
            self.optimSeq.step()
            self.writer.add_scalar("clusterloss",clusterloss,self.index)
            self.writer.add_scalar("actionloss",actionloss,self.index)

            self.index += 1

    def plot(self,emebedding:torch.Tensor):
        numpyembedding = emebedding.cpu().detach().numpy()
        center = self.clusterloss.center.cpu().detach().numpy()
        plt.scatter(x = numpyembedding[:,0],y = numpyembedding[:,1],s = 1)
        plt.scatter(x = center[:,0],y = center[:,1],s = 1,c='r')
        plt.savefig("./logs/latentspacedistributions/{}.png".format(self.index))
        plt.clf()

    