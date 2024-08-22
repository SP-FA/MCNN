from copy import copy

import torch.nn as nn
import torch
import numpy as np


np.set_printoptions(linewidth=95)


class MLPMCNN(nn.Module):
    def __init__(self):
        super(MLPMCNN, self).__init__()
        self.fc1 = nn.Linear(15 * 15, 4, bias=False).cuda()
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4, 10, bias=False).cuda()
        self.x_list = []

    def forward(self, x):
        # return self.simulate_mc(x)

        x = x.reshape(-1, 15 * 15)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def simulate_mc(self, x):
        """模拟游戏中的计算过程
        """
        sx = copy(x).reshape(1, 225)
        sfc1 = copy(self.fc1.weight.data)
        sfc1_ = sfc1 * 100
        sfc1_ = sfc1_.to(int)
        sfc1_minus = sfc1_ < 0
        sfc1_[sfc1_minus] = 16384 + sfc1_[sfc1_minus]
        # print(sfc1_)

        self.fc1.weight.data = sfc1_.to(torch.float)
        sx = self.fc1(sx)
        sx = sx.to(int) % 16384
        sx_minus = sx >= 8192
        sx[sx_minus] = 0
        # print("layer 1: ", sx)
        self.fc1.weight.data = sfc1

        sfc2 = copy(self.fc2.weight.data)
        sfc2_ = sfc2 * 100
        sfc2_ = sfc2_.to(int)
        # print("weight 2", sfc2_)

        sx = sx * sfc2_
        # print("layer 2", sx)
        sx_minus = sx < 0
        sx[sx_minus] = 4194304 + sx[sx_minus]
        # print("layer 2 1", sx)
        sx = torch.sum(sx, dim=1)
        # print("layer 2 2", sx)
        sx = sx % 4194304
        # print("layer 2 3", sx)
        sx_minus = sx >= 2097152
        sx[sx_minus] = sx[sx_minus] - 4194304
        return sx.view(1, -1)

