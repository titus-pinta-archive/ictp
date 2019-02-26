import torch
from .optimizer import Optimizer, required


class ARGD2B(Optimizer):

    def __init__(self, params, lr=required):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr)
        super(ARGD2B, self).__init__(params, defaults)

        self.state['n_iter'] = 0

        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['x_{n-1}'] = p.data
                self.state[p]['nabla(z_{n})'] = None
                self.state[p]['nabla(x_{n})'] = None
                self.state[p]['nabla(x_{n-1})'] = torch.ones_like(p.data).mul_(float('inf'))

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
                state['x_{n}'] = p.data
                state['nabla(x_{n})'] = p.grad.data
                p.data.sub_(d_p.mul(group['lr'] + group['lr'] ** 2))

        closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]

                d_p = p.grad.data
                state['nabla(z_{n})'] = d_p
                p.data.add_(p.data.sub(state['x_{n}']).mul(state['nabla(x_{n})'].div(
                                                             (state['nabla(x_{n-1})'].add(1e-5)))))
                state['nabla(x_{n-1})'] = state['nabla(x_{n})']

        closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]

                d_p = p.grad.data
                p.data.add_(d_p.mul(-group['lr']).add(state['nabla(z_{n})'].mul(-group['lr'] ** 2)))


        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue

            p.data.add_(p.grad.mul(-group['lr']))


        return closure
