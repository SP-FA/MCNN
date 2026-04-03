from copy import copy

import torch.nn as nn
import torch
import numpy as np


np.set_printoptions(linewidth=95)


class MCNN_MLP(nn.Module):
    def __init__(self):
        super(MCNN_MLP, self).__init__()
        self.fc1 = nn.Linear(15 * 15, 4, bias=False).cuda()
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4, 10, bias=False).cuda()

    def forward(self, x):
        return self.simulate_mc(x)

        # x = x.reshape(-1, 15 * 15)
        # x = self.fc1(x)
        # x = self.relu(x)
        # x = self.fc2(x)
        # return x

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

    def command(self, weight, i, x, y ,z):
        with open(f"./weights_command_{i}.mcfunction", "w") as f:
            w = weight[i].view(15, 15).flip(dims=[1]).numpy()
            for j in range(15):
                for k in range(15):
                    if i % 2 == 0:
                        f.write(f"data merge block {x + k * 2 + (1 - j % 2)} {y - (j // 2) * 4} {z} {{SuccessCount:{w[j][k]}}}\n")
                    else:
                        f.write(f"data merge block {x + k * 2 +      j % 2 } {y - (j // 2) * 4} {z} {{SuccessCount:{w[j][k]}}}\n")


class MCNN_CNN(nn.Module):
    def __init__(self):
        super(MCNN_CNN, self).__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=5, stride=3, bias=True).cuda()
        self.relu = nn.ReLU()
        self.linear = nn.Linear(2 * 4 * 4, 10, bias=True).cuda()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = x.view(-1, 2 * 4 * 4)
        x = self.linear(x)
        return x
