import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils import data
from tqdm import tqdm
from model import MLPMCNN


torch.set_printoptions(profile="full")


class RoundTransform:
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def __call__(self, x):
        # 使用阈值进行四舍五入
        return torch.where(x >= self.threshold, torch.ceil(x), torch.floor(x))


if __name__ == '__main__':
    batch_size = 256
    epochs = 20

    train_transformer = transforms.Compose([
        transforms.Resize(15),
        transforms.ToTensor(),
    ])

    test_transformer = transforms.Compose([
        transforms.Resize(15),
        transforms.ToTensor(),
        RoundTransform(0.25)
    ])

    # data loading
    train_loader = data.DataLoader(
        datasets.MNIST('data', train=True, download=True, transform=train_transformer),
        batch_size=batch_size, shuffle=True)

    test_loader = data.DataLoader(
        datasets.MNIST('data', train=False, download=True, transform=test_transformer),
        batch_size=batch_size, shuffle=True)

    model = MLPMCNN()
    lossF = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        pbar = tqdm(train_loader)
        correct = 0
        totalY = 0
        for x, y in pbar:
            x = x.cuda()
            y = y.cuda()
            optimizer.zero_grad()
            pred = model(x)
            loss = lossF(pred, y)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                for param in model.parameters():
                    param.data = torch.round(param.data * 100) / 100
                    param.data[param.data == 0] = 0.005

            pred = pred.max(1, keepdim=True)[1]
            correct += pred.eq(y.view_as(pred)).sum().item()
            totalY += len(y)
            pbar.set_description(f"[Train] Epoch: {epoch} Accuracy: {correct / totalY:.3}")

        with torch.no_grad():
            pbar = tqdm(test_loader)
            correct = 0
            totalY = 0
            for x, y in pbar:
                x = x.cuda()
                y = y.cuda()
                pred = model(x)
                pred = pred.max(1, keepdim=True)[1]
                correct += pred.eq(y.view_as(pred)).sum().item()
                totalY += len(y)
                pbar.set_description(f"[Test ] Epoch: {epoch} Accuracy: {correct / totalY:.3}")

    torch.save(model.state_dict(), "./MCNN_parameters.pkl")
