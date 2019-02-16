import torch
from .optimizer import Optimizer, required


class ARGD1(Optimizer):

    def __init__(self, params, lr=required):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr)
        super(ARGD1, self).__init__(params, defaults)

        self.state['n_iter'] = 0

        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['x_{n-1}'] = p.data
                self.state[p]['nabla(z_{n})'] = None


    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)

    def step(self, closure):
        self.state['n_iter'] += 1

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]

                d_p = p.grad.data
                p.data.sub_(d_p.mul(group['lr'] + group['lr'] ** 2))

        closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]

                d_p = p.grad.data
                state['nabla(z_{n})'] = d_p
                p.data.add_(p.data.sub(state['x_{n-1}']).mul(self.state['n_iter'] /
                                                             (self.state['n_iter'] + 3)))

        closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]

                d_p = p.grad.data
                p.data.add_(d_p.mul(-group['lr']).add(state['nabla(z_{n})'].mul(-group['lr'] ** 2)))
        return closure
