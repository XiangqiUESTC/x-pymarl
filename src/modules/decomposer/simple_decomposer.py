import numpy as np
from torch import nn
import torch as th
import torch.nn.functional as F


class SimpleDecomposer(nn.Module):
    def __init__(self, args):
        super(SimpleDecomposer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))

        self.embed_dim = args.decompose_embed_dim

        # 此处沿用qmix的代码尝试效果
        # 获取生成超网络所用的参数hypernet_layers是超网络的层数
        hypernet_layers = getattr(args, "hypernet_layers", 1)


        if hypernet_layers == 1:
            self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim)
            self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim * self.n_agents)
        elif hypernet_layers == 2:
            hypernet_embed = self.args.hypernet_embed
            self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.embed_dim))
            self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.embed_dim * self.n_agents))
        elif hypernet_layers > 2:
            raise Exception("Sorry >2 hypernet layers is not implemented!")
        else:
            raise Exception("Error setting number of hypernet layers.")

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, self.n_agents))

    def forward(self, rewards, states):
        # 保存batch_size用于还原
        bs = rewards.size(0)
        # 为了便于用批量矩阵乘法，把变量都转化为batch_size × m × p和batch_size × p × n的形式
        rewards = rewards.reshape(-1, 1, 1)
        # states不变，因其要放入超网络中生成参数
        states = states.reshape(-1, self.state_dim)

        # 第一层
        w1 = th.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)

        # 变形
        w1 = w1.reshape(-1, 1, self.embed_dim)
        b1 = b1.reshape(-1, 1, self.embed_dim)

        hidden = F.elu(th.bmm(rewards, w1) + b1)

        # 第二层
        w_final = th.abs(self.hyper_w_final(states))
        v = self.V(states)

        # 变形
        w_final = w_final.reshape(-1, self.embed_dim, self.n_agents)
        v = v.reshape(-1, 1, self.n_agents)

        # 最后结果
        y = th.bmm(hidden, w_final) + v

        # 还原形状
        agent_rewards = y.reshape(bs, -1, self.n_agents)
        return agent_rewards

    # def cuda(self, device=None):
    #     super(SimpleDecomposer, self).cuda(device)
    #     self.hyper_w_1.cuda()
    #     self.hyper_b_1.cuda()
    #     self.hyper_w_final.cuda()
    #     self.V.cuda()

