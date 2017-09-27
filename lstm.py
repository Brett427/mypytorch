# -*- coding:utf-8 -*-
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

torch.manual_seed(1)

lstm = nn.LSTM(3, 3)
# lstm 里的输入是三维矩阵,第一维体现序列结构就是一个句子几个单词，第二维小块结构，第三维度体现输入的元素。
inputs = [autograd.Variable(torch.randn((1, 3))) for _ in range(5)]


hidden = (autograd.Variable(torch.randn(1, 1, 3)), autograd.Variable(torch.randn((1, 1, 3))))

for i in inputs:
    out, hidden = lstm(i.view(1, 1, -1), hidden)
#print out

inputs = torch.cat(inputs).view(len(inputs), 1, 3)
out, hidden = lstm(inputs, hidden)
print out