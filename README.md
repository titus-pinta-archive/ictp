How to run
=======

## To install pytorch

* with pip

_pip3 install torch torchvision_

* with Anaconda

_conda install torch torchvision_


## To run

**If this instructions don't work for you please refer back to pytorch's webpage**

_python main.py -h_ for help

_python main.py --optim 'ARGD1'_ for different optimiser

_python main.py --lr '0.01'_ for different learning rates

_python main.py --optim ARGD1 --lr 0.000001_ to combine all

## Stop

Pressing **CTRL + c** will prompt to the following save message

_Save current proggres? (y)es/(n)o/(c)ancel_

Progress is stored in *.part files

## Help
Please refer to the help option for a more detailed look at the different parameters


## Current optimisers

*Working optimisers marked as tested*

1. ADADELTA
2. ADAGRAD
3. ADAMAX
4. ADAM
5. ARGD1 **tested**
6. ARGD1B **tested**
7. ARGD2 **tested**
8. ARGD2B **tested**
9. ASGD
10. LBFGS
11. LR_SCHEDULER
12. RMSPROP
13. RPROP
14. SGD **tested**
15. SPARSE_ADAM

## Man Page
usage: main.py [-h] [--batch-size N] [--test-batch-size N] [--epochs N]
               [--lr LR] [--momentum M] [--optim OPTIM] [--no-cuda] [--seed S]
               [--log-interval N] [--save-name SAVE_NAME]

PyTorch MNIST Example

optional arguments:

  -h, --help            show this help message and exit

  --batch-size N        input batch size for training (default: 64)

  --test-batch-size N   input batch size for testing (default: 1000)

  --epochs N            number of epochs to train (default: 30)

  --lr LR               learning rate (default: 0.00001)

  --momentum M          SGD momentum (default: 0.5)

  --optim OPTIM         Optimiser to use (default: SGD)

  --no-cuda             disables CUDA training

  --seed S              random seed (default: 1)

  --log-interval N      how many batches to wait before logging training
status

  --stoch               use stochastic gradient computation

  --save-name SAVE_NAME
File name to save current resault.If None it will use
the name of the optimiser. (default: None)

  --load-part LOAD_PART name of the saved .part files to load (default: None)'
