import numpy as np

import torch


class Node(object):
    def __init__(self, *input, require_grad=True):
        self.require_grad = require_grad

        self.grad = None
        self.output = None

        # build the compute graph
        self.prior = input
        self.next = None

        # print(self.prior)
        # set the input next
        for p in self.prior:
            #print(p)
            p.next = self

    def func(self):
        raise NotImplementedError()

    def diff_func(self):
        raise NotImplementedError()

    def forward(self):
        if self.prior is not None:
            inpt = [in_node.forward() for in_node in self.prior]
        else:
            inpt = self.value

        out = self.func(*inpt)
        if self.require_grad:
            self.output = out

        return out

    def backward(self, x):
        if self.next is None:
            return self.diff_func(x)
        return self.next.backward(self.output) * self.diff_func(x)


class add(Node):
    def func(self, *x):
        return x[0] + x[1]

    def diff_func(self, x):
        return 1


class scale(Node):
    def func(self, x, scalar):
        self.scalar = scalar
        return scalar * x

    def diff_func(self, x):
        return self.scalar


class mml(Node):
    def func(self, *x):
        self.tmp = x
        return np.dot(x[0], x[1])

    def diff_func(self, x):
        print(self.tmp)
        if (x == self.tmp[0]).all():
            return self.tmp[1]
        else:
            return self.tmp[0]


class power(Node):
    def __init__(self, p, input, require_grad=True):
        super(power, self).__init__(input)
        self.p = p

    def func(self, *x):
        return np.power(x, self.p)

    def diff_func(self, x):
        return self.p * np.power(x, self.p - 1)


class exp(Node):
    def func(self, *x):
        return np.exp(x)

    def diff_func(self, x):
        return self.output


class log(Node):
    def func(self, *x):
        return np.log(x)

    def diff_func(self, x):
        return 1 / x


class sum(Node):
    def func(self, *x):
        return np.sum(x)

    def diff_func(self, x):
        return np.ones_like(x)


class input_node(Node):
    def __init__(self, value):
        super(input_node, self).__init__()
        self.prior = None
        self.value = value

    def func(self, *x):
        return np.array(x)

    def diff_func(self, x):
        return 1


a = np.array([0.1, 1, 1])
b = np.array([0.2, 2, 1])

X = input_node(*[a])
w = input_node(*[b])

#test = sum(X)
#print(test.forward())
#print(X.backward(X.value))
#head.forward()
#print(head.prior)
#print(head.value)
#print(head.require_grad)
#print(head.forward())
test = mml(X, w)
#print(test.forward())
test2 = power(2, test)
#test2 = exp(test)
print(test2.forward())
#print(test.output)

#print(head.value)
print(X.backward(X.value))

#b = np.array([0.1, 1, 1])
#print((a == b).all())

tenser_a = torch.tensor(a).to(torch.float32).requires_grad_()
tenser_b = torch.tensor(b).to(torch.float32).requires_grad_()

z = (torch.dot(tenser_a, tenser_b))**2
z.backward()
print(z)
print(tenser_a.grad)
