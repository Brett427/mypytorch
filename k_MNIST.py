# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def loadtraindata(path):
    data = pd.read_csv(path)
    label = data['label'].values
    data_image = data.loc[:, 'pixel0':'pixel783']
    r, c = data_image.values.shape
    image = []
    for i in range(r):
        image.append(data_image.loc[i, :].values.reshape(28, -1))
    image = np.array(image)
    # print image.shape
    return image, label


def loadtestdata(path):
    data = pd.read_csv(path)
    image = data.loc[:, 'pixel0':'pixel783']
    #print image
    r, c = image.values.shape
    images = []
    for i in range(r):
        images.append(image.loc[i, :].values.reshape(28, -1))
    images = np.array(images)
    return images


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.c1 = nn.Conv2d(1, 6, 3, padding=1)
        self.c2 = nn.Conv2d(6, 16, 3, padding=1)
        self.fc1 = nn.Linear(16 * 7 * 7, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        s1 = F.max_pool2d(F.relu(self.c1(x)), 2)
        s2 = F.max_pool2d(F.relu(self.c2(s1)), 2)
        s2 = s2.view(-1, 16 * 7 * 7)
        s3 = F.relu(self.fc1(s2))
        x = F.relu(self.fc2(s3))
        x = self.fc3(x)
        return F.log_softmax(x)


def train(net, train_all_data, train_label):
    sgd = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    product_loss = nn.CrossEntropyLoss()
    for epoch in range(120):
        for i in range(0, train_label.shape[0], 2000):
            train_data = train_all_data[i:i + 2000]
            train_data = torch.from_numpy(train_data[:, np.newaxis, :, :])
            # print train_data.size()
            train_data = train_data.type(torch.FloatTensor)
            j = train_label[i:i + 2000]

            image = Variable(train_data)
            label = Variable(torch.from_numpy(j))
            label = label.type(torch.LongTensor)
            output = net(image)
            sgd.zero_grad()
            loss = product_loss(output, label)
            loss.backward()
            sgd.step()
            print('[%d, %5d] loss:%.3f' %
                  (epoch + 1, i + 2000, loss.data[0]))


def test(net, test_all_data):
    k = 0
    res =[]
    for i in range(0, test_all_data.shape[0], 100):
        test_data = test_all_data[i:i + 100]
        test_data = torch.from_numpy(test_data[:, np.newaxis, :, :])
        test_data = test_data.type(torch.FloatTensor)
        # j = test_label[i:i + 100]
        image = Variable(test_data)
        output = net(image)
        a = output.data.max(1)[1].numpy()
        for i in a:
            res.append(i[0])
    return res

if __name__ == '__main__':
    traindata, label = loadtraindata('mnist_train.csv')
    net = Net()
    train(net, traindata, label)
    # torch.save(net, 'net.pkl',)
    #net = torch.load('net.pkl')
    testdata = loadtestdata('mnist_test.csv')
    res = []
    res=test(net, testdata)
    print len(res)

    ids =[]
    for i in range(1, 28001):
        ids.append(i)
    print len(ids)
    gender_submission = pd.DataFrame({'ImageId': ids, 'Label': res})
    gender_submission.to_csv('/home/brett/Desktop/111.csv', index=None)


