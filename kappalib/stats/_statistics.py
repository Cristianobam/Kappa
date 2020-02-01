import numpy as np


__all__ = ['pooledVar', 'moving_average']

def moving_average(x,w,mode='linear'):
    """
    Moving Average for an Array.


    Parameters
    ----------
    x: Array Vector
        The arrays must have the same shape, except in the dimension.

    w: Int
        Window size

    mode: {linear, exp}
        Linear: Simple Moving Average 
        Exponential: Exponetial Weighted Moving Average

    Returns
    -------
    mean : array
        The calculated moving average.
    """  
    if mode == 'linear':
        weight = np.ones(w)/w
    elif mode == 'exp':
        weight = np.exp(-np.linspace(0,w,w))
    x = np.array(x)
    mean = np.convolve(x,weight, mode='full')[:-w+1]
    
    return mean

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