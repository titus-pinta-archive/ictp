#! /bin/python

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import optim
from torchvision import datasets, transforms

import dill

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def train_stoch(args, model, device, train_loader, optimizer, epoch):


    model.train()


    for batch_idx, (data, target) in enumerate(train_loader):
        loss = None

        def closure():
            nonlocal data
            nonlocal target
            nonlocal loss

            optimizer.zero_grad()
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()

        closure()
        optimizer.step(closure)

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))



def train_non_stoch(args, model, device, train_loader, optimizer, epoch):
    def closure():
        optimizer.zero_grad()


        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
    model.train()

    closure()

    optimizer.step(closure)

def test(args, model, device, test_loader, result):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    result.append(correct)

def main():
    # Training settings


    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 30)')
    parser.add_argument('--lr', type=float, default=0.00001, metavar='LR',
                        help='learning rate (default: 0.00001)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--optim', default='SGD', help='Optimiser to use (default: SGD)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--stoch', action='store_true', default=False,
                        help='use stochastic gradient computation')
    parser.add_argument('--save-name', default=None,help='File name to save current resault.' +
                        'If None it will use the name of the optimiser. (default: None)')
    args = parser.parse_args()

    print('Gradient is computed {}'.format('stochastically' if args.stoch else 'non stochastically'))

    print('Will train for {} epochs with a batch size of {}'.format(args.epochs, args.batch_size))

    if(args.stoch):
        train = train_stoch
    else:
        train = train_non_stoch

    use_cuda = False #not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device('cuda' if use_cuda else 'cpu')

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)


    model = Net()
    model.load_state_dict(torch.load('init.model'))
    model.to(device)
    try:
        extra_params = {}

        if args.optim == 'SGD':
            extra_params = {'momentum': args.momentum}

        optim_class = getattr(optim, args.optim)
        optimizer = optim_class(model.parameters(), lr=args.lr, **extra_params)
    except Exception as e:
        print(e)
        raise ValueError('Undefined Optimiser: {}'.format(args.optim))

    result = []


    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader, result)

    with open('{}{}.result'.format(args.optim, '.stoch' if args.stoch else ''), 'wb') as f:
        dill.dump((len(test_loader.dataset), result), f)

    torch.save(model.state_dict(), '{}{}.model'.format(args.optim, '.stoch' if args.stoch else ''))

if __name__ == '__main__':
    main()
