import numpy as np


__all__ = ['pooledVar', 'moving_average']

def moving_average(x,axis=0,type='simple',weight=1):
    """
    Moving Average for an Array.


    Parameters
    ----------
    x: Array Vector
        The arrays must have the same shape, except in the dimension.

    Returns
    -------
    sp : float
        The calculated pooled variance.
    """

    x = np.array(x)
    n = np.shape(x)[axis]
    weights = np.ones(n)/n 
    for i in range(n):
       np.convolve(x[i,:], n, mode='valid') 

def pooledVar(x, y):
    """
    Pooled Variance of two samples.


    Parameters
    ----------
    x, y : 1 dimensional array_like
        The arrays must have the same shape, except in the dimension.

    Returns
    -------
    sp : float
        The calculated pooled variance.
    """
    n1 = np.size(x)
    n2 = np.size(y)
    s1 = np.var(x, ddof=1)
    s2 = np.var(y, ddof=1)
    return ((n1-1)*s1+(n2-1)*s2)/(n1+n2-2)