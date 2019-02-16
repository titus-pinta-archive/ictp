#!/usr/bin/env python
from __future__ import print_function
from itertools import count
import optim.argd1 as optim
import torch
import torch.nn.functional as F
import numpy as np

W_target = torch.Tensor([[1], [2], [3], [4]])
b_target = torch.Tensor([0.3])


def make_features(x):
    """Builds features i.e. a matrix with columns [x, x^2, x^3, x^4]."""
    x = x.unsqueeze(1)
    return torch.cat([x ** i for i in range(1, 4 + 1)], 1)


def f(x):
    """Approximated function."""
    return x.mm(W_target) + b_target.item()


def poly_desc(W, b):
    """Creates a string description of a polynomial."""
    result = 'y = '
    for i, w in enumerate(W):
        result += '{:+.2f} x^{} '.format(w, len(W) - i)
    result += '{:+.2f}'.format(b[0])
    return result


def get_batch(batch_size=32):
    """Builds a batch i.e. (x, f(x)) pair."""
    random = torch.Tensor(np.linspace(0, 1, 32))
    x = make_features(random)
    y = f(x)
    return x, y


# Define model
fc = torch.nn.Linear(W_target.size(0), 1)
fc.load_state_dict(torch.load('fc.sv'))

batch_x, batch_y = get_batch()
op = optim.ARGD1(fc.parameters(), lr=0.1)

loss = 0.0

def closure():
    global loss
    global output
    fc.zero_grad()

    output = F.smooth_l1_loss(fc(batch_x), batch_y)
    loss = output.item()

    output.backward()

while True:
    closure()

    op.step(closure)

    if op.state['n_iter'] % 1000 == 0:
        print(loss)

    if loss < 1e-5:
        break

print('==> Iterations {}'.format(op.state['n_iter']))


print('==> Learned function:\t' + poly_desc(fc.weight.view(-1), fc.bias))
print('==> Actual function:\t' + poly_desc(W_target.view(-1), b_target))
