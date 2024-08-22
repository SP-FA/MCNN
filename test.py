import torch

from torchvision import datasets, transforms
from torch.utils import data
from tqdm import tqdm

from model import MLPMCNN
import numpy as np
import matplotlib.pyplot as plt


class RoundTransform:
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def __call__(self, x):
        # 使用阈值进行四舍五入
        return torch.where(x >= self.threshold, torch.ceil(x), torch.floor(x))


test_transformer = transforms.Compose([
    transforms.Resize(15),
    transforms.ToTensor(),
    RoundTransform(0.25)
])

test_loader = data.DataLoader(
        datasets.MNIST('data', train=False, download=True, transform=test_transformer),
        batch_size=1, shuffle=True)


model = MLPMCNN()
model.load_state_dict(torch.load("MCNN_parameters.pkl"))

# 单个样例
x = torch.tensor([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, ],
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, ],
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, ],
    [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, ],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, ],
    [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, ],
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, ],
    [0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, ],
    [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, ],
    [0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, ],
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, ],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, ],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
]).to(torch.float)
y = torch.tensor(6)

with torch.no_grad():
    correct = 0
    totalY = 0
    for x, y in tqdm(test_loader):
        x = x.cuda()
        y = y.cuda()
        pred = model(x)
        pred = pred.max(1, keepdim=True)[1]
        correct += 1 if pred == y else 0
        totalY += 1

        # if pred == y:
        #     img = np.reshape(x.cpu().numpy(), (15, 15))
        #     plt.matshow(img, cmap=plt.get_cmap('gray'))
        #     plt.show()
    print(f"acc: {correct / totalY}")

