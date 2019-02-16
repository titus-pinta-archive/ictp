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
#                p.data.sub_(d_p.mul(group['lr'] + group['lr'] ** 2))
                p.data.sub_(d_p.mul(group['lr']))

        closure()

#        for group in self.param_groups:
#
#            for p in group['params']:
#                if p.grad is None:
#                    continue
#                d_p = p.grad.data
#                y = p.data - beta * (p.data - ) 


        return closure
