import torch
from .optimizer import Optimizer, required


class A3(Optimizer):

    def __init__(self, params, lr=required, k=5):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr, k=k)
        super(A3, self).__init__(params, defaults)

        self.state['n_iter'] = 0

        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['u'] = p.data
                self.state[p]['v'] = torch.zeros_like(p.data)


    def __setstate__(self, state):
        super(A3, self).__setstate__(state)

    def step(self, closure):
        self.state['n_iter'] += 1

        for group in self.param_groups:
            k = group['k']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                lr = group['lr']
                d_p = p.grad
                beta = self.state['n_iter'] / (self.state['n_iter'] + 3)
                state['v'] = state['v'].mul(beta ** k * (1 - lr)) - d_p.mul(beta * lr)
                state['u'] = p.data.add(p.data.sub(state['u']).mul(beta * (1 - lr * beta))).sub(d_p.mul(lr * lr))
                p.data = state['u'] + state['v'].mul(lr * beta)

        return closure
