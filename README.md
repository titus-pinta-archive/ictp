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

## Help
Please refer to the help option for a more detailed look at the different parameters


## Current optimisers

1. ADADELTA
2. ADAGRAD
3. ADAMAX
4. ADAM
5. ARGD1
6. ASGD
7. LBFGS
8. LR_SCHEDULER
9. RMSPROP
10. RPROP
11. SGD
12. SPARSE_ADAM
