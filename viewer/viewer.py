import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

import dill
import matplotlib.pyplot as plt
import numpy as np


import main


def help():
    print('\nThe following functions are available:\n')
    print('load_model(path)')
    print('\treturns the model stored in the specified path')
    print('\tplease provide the extension (normally .model or .model.part)')
    print('\nexample call: viewer.load_model(\'SGD.model\')\n')
    print('load result(path)')
    print('\trerturns the result vector stored in the specified path')
    print('\tplease also provide the extension')
    print('\tthe result is a tupe with the first component the number of testing images')
    print('\t\tand the second component a vector with the number of correct prediction')
    print('\t\tfor each  epoch')
    print('\nexample call: viewer.load_result(\'SGD.result\')\n')
    print('load_mnist(batch_size=64, test_batch_size=1000, use_cuda=True, seed=1)')
    print('\tthe meaning of the parameters is the same as for the torch dataloader')
    print('\nexample call: viewer.load_mnist()\n')

use_cuda = torch.cuda.is_available()


def load_model(path):
    device = torch.device('cuda' if use_cuda else 'cpu')
    net = main.Net()
    net.load_state_dict(torch.load(path))
    net.to(device)
    return net


def load_result(path):
    with open(path, 'rb') as f:
        res = dill.load(f)
    return res

def load_mnist(batch_size=64, test_batch_size=1000, use_cuda=True, seed=1):

    use_cuda = use_cuda and torch.cuda.is_available()

    torch.manual_seed(seed)

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=test_batch_size, shuffle=True, **kwargs)

    return train_loader, test_loader

def plot_result(result, plot=None, correct=True, fraction=True, show=True):

    global plt
    if plot is not None:
        plt = plot

    if correct:
        data = np.array(result[1])
    else:
        data = result[0] - np.array(result[1])

    ymax = result[0]

    if fraction:
        data = data / result[0]
        ymax = 1
    epochs = np.linspace(0, len(result[1]), len(result[1]))

    plt.ylim(0, ymax)

    plt.plot(epochs, data)
    if show:
        plt.show()

    return plt


def model_statistics(model, loader):

    device = torch.device('cuda' if use_cuda else 'cpu')

    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(loader.dataset),
        100. * correct / len(loader.dataset)))

help()
