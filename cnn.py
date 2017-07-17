import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.utils.data as data
import torch.optim as optim
import torch.nn.functional as F

batch_size = 50
learning_rate = 0.01
hidden_size = 500
num_classes = 10
epochs = 5
download = False

train_dataset = dsets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=download)
train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

test_dataset = dsets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=download)
test_loader = data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=2)


class MyCnn(nn.Module):
    def __init__(self):
        super(MyCnn, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.out1 = nn.Linear(7*7*20, 50)
        self.out2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.out1(x)
        x = self.out2(x)
        return x

loss_fn = nn.CrossEntropyLoss()
model = MyCnn()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))


def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss =test_loss + loss_fn(output, target).data[0] # sum up batch loss
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


for epoch in range(1, epochs + 1):
    train(epoch)
    test()