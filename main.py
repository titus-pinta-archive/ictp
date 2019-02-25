#! /usr/python-pytorch/bin/python

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import optim
from torchvision import datasets, transforms

import dill

import signal
import sys



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
    closure_calls = 0
    def closure():
        nonlocal closure_calls
        closure_calls += 1
        print('\nNumber of closure calls: {}\n'.format(closure_calls))
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

def test(args, model, device, test_loader, result_correct, result_loss):
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

    result_correct.append(correct)
    result_loss.append(test_loss)

def main():
    #handle signals
    def exit_with_choice():
        print('Save current progress? (y)es/(n)o/(c)ancel ', end='')
        exit_choice='y'
        try:
            exit_choice = input()
        except EOFError:
            print('\nStdin error: Defaults to y')
            exit_choice = 'y'


        if exit_choice == 'y' or exit_choice == 'Y' or exit_choice == 'yes' or exit_choice == 'Yes':
            with open('./results/{}{}.result.part'.format(args.optim, '.stoch' if args.stoch else ''), 'wb') as f:
                dill.dump((len(test_loader.dataset),((args.lr, args.momentum) if args.optim ==
                                                     'SGD' else args.lr), result_correct, result_loss), f)

            torch.save(model.state_dict(), './results/{}{}.model.part'.format(args.optim, '.stoch' if args.stoch else ''))
            exit(0)

        elif exit_choice == 'n' or exit_choice == 'N' or exit_choice == 'no' or exit_choice == 'No':
            exit(0)


    def sigint_handler(sig, frame):
        exit_with_choice()



    signal.signal(signal.SIGINT, sigint_handler)



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
    parser.add_argument('--save-name', default=None, help='File name to save current resault.' +
                        'If None it will use the name of the optimiser. (default: None)')
    parser.add_argument('--load-part', default=None, help='name of the saved .part files to load' +
                        '(default: None)')
    parser.add_argument('--fash', action='store_true', default=False, help='Use MNIST fashion not MNIST')
    args = parser.parse_args()

    print('Gradient is computed {}'.format('stochastically' if args.stoch else 'non stochastically'))

    print('Will train for {} epochs with a batch size of {}'.format(args.epochs, args.batch_size))


    if(args.stoch):
        train = train_stoch
    else:
        train = train_non_stoch

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    print('Computing on {}'.format('cuda' if use_cuda else 'cpu'))

    torch.manual_seed(args.seed)

    device = torch.device('cuda' if use_cuda else 'cpu')

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    data_path = './data' if not args.fash else './data-fashion'
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(data_path, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(data_path, train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)


    model = Net()
    if args.load_part is not None:
        model.load_state_dict(torch.load('./results/{}.model.part'.format(args.load_part)))
        with open('./results/{}.result.part'.format(args.load_part), 'rb') as f:
            result = dill.load(f)
            result_correct = result[2]
            result_loss = result[3]

        print('Previous correct  {}'.format(result_correct))
        print('Previous loss {}'.format(result_loss))

    else:
        model.load_state_dict(torch.load('init.model'))
        result_correct = []
        result_loss = []



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



    for epoch in range(1, args.epochs + 1):
        try:
            train(args, model, device, train_loader, optimizer, epoch)
            test(args, model, device, test_loader, result_correct, result_loss)
        except KeyboardInterrupt:
            exit_with_choice()

    with open('./results/{}{}.result'.format(args.optim, '.stoch' if args.stoch else ''), 'wb') as f:
        dill.dump((len(test_loader.dataset),((args.lr, args.momentum) if args.optim == 'SGD' else
                                             args.lr), result_correct,result_loss), f)

    torch.save(model.state_dict(), './results/{}{}.model'.format(args.optim, '.stoch' if args.stoch else ''))

if __name__ == '__main__':
    main()
