"""
:mod:`torch.optim` is a package implementing various optimization algorithms.
Most commonly used methods are already supported, and the interface is general
enough, so that more sophisticated ones can be also easily integrated in the
future.
"""

from .adadelta import Adadelta
from .adagrad import Adagrad
from .adam import Adam
from .sparse_adam import SparseAdam
from .adamax import Adamax
from .asgd import ASGD
from .sgd import SGD
from .rprop import Rprop
from .rmsprop import RMSprop
from .optimizer import Optimizer
from .lbfgs import LBFGS
from .argd1 import ARGD1
from .argd2 import ARGD2
from .argd1b import ARGD1B
from .argd2b import ARGD2B
from .a3 import A3
from . import lr_scheduler

del adadelta
del adagrad
del adam
del sparse_adam
del adamax
del asgd
del sgd
del rprop
del rmsprop
del optimizer
del lbfgs
del argd1
del argd2
del argd1b
del argd2b
del a3
