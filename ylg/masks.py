import numpy as np
import random
import math
import tensorflow as tf
from collections import defaultdict
from itertools import cycle


def allow_non_square(fn):
    '''
        Decorator: Augments nO for square masks or allows non square masks.
    '''
    def wrap(self, nI, **kwargs):
        nO = kwargs.pop('nO', None)
        if nO is None:
            nO = nI
        return fn(self, nI, nO, **kwargs)
    return wrap


def numpy(fn):
    '''
        Decorator: Converts a list to a numpy array of float32.
    '''
    def wrap(self, *args, **kwargs):
        indices = fn(self, *args, **kwargs)
        tensor = np.array(indices, dtype=np.float32)
        return tensor
    return wrap


def disallow_downsampling(fn):
    '''
        Decorator: Raises ValueError when the number of output nodes is
                   less than the number of input nodes.
    '''
    def wrap(self, nI, **kwargs):
        nO = kwargs.pop('nO', None)
        if nO is None:
            nO = nI
        if nO < nI:
            raise ValueError('Downsampling not supported.')
        return fn(self, nI, nO, **kwargs)
    return wrap


def disallow_non_square(fn):
    '''
        Decorator: Raises ValueError for non square masks.
    '''
    def wrap(self, nI, **kwargs):
        nO = kwargs.pop('nO', None)
        if nO is not None and nO != nI:
            raise ValueError('Non square masks not supported')
        return fn(self, nI, **kwargs)
    return wrap


def compute_stride(fn):
    '''
        Decorator: Computes a default value for stride, based on number
                   of nodes.
    '''
    def wrap(self, nL, nO=None, **kwargs):
        stride = kwargs.pop('stride', None)
        if stride is None:
            stride = math.floor(math.sqrt(nL))
        return fn(self, nL, nO=nO, stride=stride, **kwargs)
    return wrap
