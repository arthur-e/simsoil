import warnings
import numpy as np
from scipy.linalg import solve_banded
from scipy.sparse import dia_matrix

# Instruct pdoc3 to ignore the tests
__pdoc__ = {}
__pdoc__['tests'] = False


class Namespace(object):
    '''
    Dummy class for holding attributes.
    '''
    def __init__(self):
        pass

    def add(self, label, value):
        '''
        Adds a new attribute to the Namespace instance.

        Parameters
        ----------
        label : str
            The name of the attribute; will be accessed, e.g.:
            `Namespace.label`
        value : None
            Any kind of value to be stored
        '''
        setattr(self, label, value)


def suppress_warnings(func):
    'Decorator to suppress NumPy warnings'
    def inner(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return func(*args, **kwargs)
    return inner


def tridiag_solver(tri, r, kl = 1, ku = 1, banded = None):
    '''
    Solution to the tridiagonal equation by solving the system of equations
    in sparse form. Creates a banded matrix consisting of the diagonals,
    starting with the lowest diagonal and moving up, e.g., for matrix:

        A = [[10.,  2.,  0.,  0.],
             [ 3., 10.,  4.,  0.],
             [ 0.,  1.,  7.,  5.],
             [ 0.,  0.,  3.,  4.]]
        banded = [[ 3.,  1.,  3.,  0.],
                  [10., 10.,  7.,  4.],
                  [ 0.,  2.,  4.,  5.]]

    The banded matrix is what should be provided to the optoinal "banded"
    argument, which should be used if the banded matrix can be created faster
    than `scipy.sparse.dia_matrix()`.

    Parameters
    ----------
    tri : numpy.ndarray
        A tridiagonal matrix (N x N)
    r : numpy.ndarray
        Vector of solutions to the system, Ax = r, where A is the tridiagonal
        matrix
    kl : int
        Lower bandwidth (number of lower diagonals) (Default: 1)
    ku : int
        Upper bandwidth (number of upper diagonals) (Default: 1)
    banded : numpy.ndarray
        (Optional) Provide the banded matrix with diagonals along the rows;
        this can be faster than scipy.sparse.dia_matrix()

    Returns
    -------
    numpy.ndarray
    '''
    assert tri.ndim == 2 and (tri.shape[0] == tri.shape[1]),\
        'Only supports 2-dimensional square matrices'
    if banded is None:
        banded = dia_matrix(tri).data
    # If it is necessary, in a future implementation, to extract diagonals;
    #   this is a starting point for problems where kl = ku = 1
    # n = tri.shape[0]
    # a, b, c = [ # (n-1, n, n-1) refer to the lengths of each vector
    #     sparse[(i+1),(max(0,i)):j]
    #     for i, j in zip(range(-1, 2), (n-1, n, n+1))
    # ]
    return solve_banded((kl, ku), np.flipud(banded), r)
