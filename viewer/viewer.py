import torch
import dill
import main

from torchvision import datasets, transforms

def load_model(path):
    net = main.Net()
    net.load_state_dict(torch.load(path))
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
