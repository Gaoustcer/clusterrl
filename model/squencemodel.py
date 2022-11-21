import torch.nn as nn
import gym
import torch



class SequenceModel(nn.Module):
    def __init__(self,env:gym.Env,embed_dim = 2,attention_dim = 32) -> None:
        super(SequenceModel,self).__init__()
        self.statedim = len(env.observation_space.sample())
        self.actiondim = len(env.action_space.sample())
        self.embeddim = attention_dim
        self.outputdim = embed_dim
        self.embeddingnet = nn.Embedding(
            num_embeddings=1024,
            embedding_dim=self.embeddim
        )
        self.stateembedding = nn.Sequential(
            nn.Linear(self.statedim,32),
            nn.ReLU(),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Linear(16,self.embeddim)
        )

        self.actionembedding = nn.Sequential(
            nn.Linear(self.actiondim,32),
            nn.ReLU(),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Linear(16,self.embeddim)
        )

        self.attention = nn.MultiheadAttention(
            embed_dim = self.embeddim,
            num_heads = 4,
            batch_first = True
        )
        self.MLP = nn.Sequential(
            nn.Linear(self.embeddim,32),
            nn.ReLU(),
            nn.Linear(32,self.outputdim)
        )
        # self.attention.forward()
    
    def forward(self,states,actions,timesteps):
        n_batch = states.shape[0]
        n_squence = states.shape[1]
        # print("state shape is",states.shape)
        # print("timesteps shape is",timesteps.shape)
        timestepembedding = self.embeddingnet(timesteps)
        # print("timestep embedding shape is",timestepembedding.shape)
        statesembedding = self.stateembedding(states) + timestepembedding
        actionsembedding = self.actionembedding(actions) + timestepembedding
        embedding = torch.concat((statesembedding,actionsembedding),dim = -1)
        embedding = embedding.reshape(n_batch,-1,self.embeddim)
        attention = self.attention.forward(query=embedding,key=embedding,value=embedding,need_weights=False)[0]
        # self.attention.forward()
        # print(attention)
        # print(attention.shape)
        attention = attention[:,-1,:]
        attention = self.MLP(attention)
        return attention
        # return attention[:,-1,:]

if __name__ == "__main__":
    import d4rl
    env = gym.make("hopper-medium-v2")

    Sequence = SequenceModel(env).cuda()
    states = torch.rand(3,17,11).cuda()
    actions = torch.rand(3,17,3).cuda()
    timesteps = torch.randint(0,17,(3,17)).cuda()
    embed_actions = Sequence(states,actions,timesteps)
    print(embed_actions.shape)

        
    
