import numpy as np

import torch
# import torch.tensor as tensor
from torch.autograd import Variable

import AutoGrad as ag


def l2_norm(x):
    X = ag.input_node(x)

    out = ag.power(2, X)
    out = ag.sum(X)

    return out


def svm_torch(data, label, epoch, lr):
    # initiate w, b all zero
    w = torch.zeros(2).requires_grad_()
    b = torch.zeros(1).requires_grad_()

    # calculate loss
    for epoch in range(10):
        for x, y in zip(data, label):
            x = Variable(x, requires_grad=True)
            y = Variable(y, requires_grad=True)

            loss = -y * (torch.dot(x, w) + b) / torch.norm(w)

            w.zero_grad()
            b.zero_grad()
            loss.backward()

            # parameter update
            w -= w.grad * lr
            b -= b.grad * lr
