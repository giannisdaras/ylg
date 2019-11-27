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


class SparseMask:

    def convert_to_1d(i, j, cols):
        return i * cols + j

    @classmethod
    def get_indices(self, **kwargs):
        raise NotImplementedError()

    @classmethod
    @allow_non_square
    def get_mask(self, nI, nO, **kwargs):
        indices = self.get_indices(nI, nO=nO, **kwargs)
        tensor = np.zeros([nO, nI], dtype=np.float32)
        tensor[indices[:, 0].astype(int), indices[:, 1].astype(int)] = 1
        return tensor

    @classmethod
    @allow_non_square
    @numpy
    def get_grid_mask(self, gridI, gridO, **kwargs):
        '''
            Returns a mask that corresponds to grid inputs.
        '''
        rowsI, colsI = gridI
        rowsO, colsO = gridO
        nI = rowsI * colsI
        nO = rowsO * colsO

        indices = self.get_indices(gridI, nO=gridO, **kwargs)
        tensor = np.zeros([nO, nI], dtype=np.float32)
        tensor[indices[:, 0].astype(int), indices[:, 1].astype(int)] = 1
        return tensor

    def validate_bounds(indices, nO, nI):
        '''
            Decorator: Removes out of bounds for a grid.
        '''
        for index, (i, j) in enumerate(indices):
            if i >= nO or j >= nI:
                indices.pop(index)
        return indices

    def enumerate_cells(rows, cols):
        # maps distances to cells
        distances = defaultdict(list)

        # maps numbers to cells
        enumeration = {}

        for i in range(rows):
            for j in range(cols):
                distance = i + j
                distances[distance].append([i, j])

        sorted_distances = sorted(list(distances.keys()))

        numbers = list(range(rows * cols))
        for distance in sorted_distances:
            cells = distances[distance]
            for cell in cells:
                enumeration[numbers.pop(0)] = cell
        return enumeration

    @classmethod
    @disallow_non_square
    @numpy
    def get_square_grid_indices_from_1d(self, grid):
        rows, cols = grid
        enumeration = self.enumerate_cells(rows, cols)
        mask_indices = self.get_indices(grid[0] * grid[1])
        indices = []
        for sO, sI in mask_indices:
            x_i, x_j = enumeration[sO]
            y_i, y_j = enumeration[sI]

            x = self.convert_to_1d(x_i, x_j, cols)
            y = self.convert_to_1d(y_i, y_j, cols)
            indices.append([x, y])
        return indices

    @classmethod
    @allow_non_square
    @numpy
    def get_grid_indices_from_1d(self, gridI, gridO):
        rowsI, colsI = gridI
        rowsO, colsO = gridO

        if rowsI == rowsO and colsI == colsO:
            return self.get_square_grid_indices_from_1d(gridI)

        blocks_ratio = (rowsO * colsO) // (rowsI * colsI)
        offset = (rowsI * colsI)
        indices = self.get_square_grid_indices_from_1d(gridI)
        offset_array = np.zeros(indices.shape)
        for i in range(1, blocks_ratio):
            new_indices = self.get_square_grid_indices_from_1d(gridI)
            offset_array[:, 0] += offset
            new_indices = offset_array + new_indices
            indices = np.concatenate([indices, new_indices])
        return indices

    @classmethod
    @allow_non_square
    def get_grid_mask_from_1d(self, gridI, gridO, **kwargs):
        rowsO, colsO = gridO
        nO = rowsO * colsO
        rowsI, colsI = gridI
        nI = rowsI * colsI

        indices = self.get_grid_indices_from_1d(gridI, nO=gridO, **kwargs)
        tensor = np.zeros([nO, nI], dtype=np.float32)
        tensor[indices[:, 0].astype(int), indices[:, 1].astype(int)] = 1
        return tensor
