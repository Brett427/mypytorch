import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data.dataloader as dataloader
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

download = False
batch_size = 50
learning_rate = 0.01
trainset = torchvision.datasets.CIFAR10(root='./cifar10', train=True, transform=transform, download=download)
testset = torchvision.datasets.CIFAR10(root='./cifar10', train=False, transform=transform, download=download)

trainloader = dataloader.DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, num_workers=2)
testloader = dataloader.DataLoader(dataset=testset, batch_size=batch_size, shuffle=True, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class mynet(nn.Module):
    def __init__(self):
        super(mynet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = mynet()
use_gpu = torch.cuda.is_available()
if use_gpu:
    net = net.cuda()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(params=net.parameters(), lr=learning_rate)


def train(epoch):
    net.train()
    training_loss = 0
    for i, (data, target) in enumerate(trainloader):
        data, target = Variable(data).cuda(), Variable(target).cuda()
        output = net(data)
        optimizer.zero_grad()
        training_loss = loss_fn(output, target)
        training_loss.backward()
        optimizer.step()
        if i % 50 == 0:
            print epoch+1, i/50+1, training_loss.data[0]


def test():
    net.eval()
    test_loss = 0
    correct = 0
    for data, target in testloader:
        data, target = Variable(data).cuda(), Variable(target).cuda()
        output = net(data)
        test_loss = loss_fn(output, target).data[0]+test_loss
        pred = output.data.max(1)[1]
        correct = correct + pred.eq(target.data).cpu().sum()
    test_loss /= len(testloader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(testloader.dataset),
        100. * correct / len(testloader.dataset)))

for epoch in range(60):
    train(epoch)
    test()
