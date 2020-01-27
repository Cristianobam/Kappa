import numpy as np
from ._summary import *
from scipy.stats import t, f, chi, norm

__all__ = ['pooledVar', 'ttest', 'descriptives','correlation']


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
    n = np.shape(t)[axis]
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

def ttest(x=None, y=None, alternative='two-sided', mu=0, data=None, paired=False, var_equal=False, alpha=0.05, effect_size=False, mean_difference=False, confidence_interval=False):
    """
    T-test for means of two samples from descriptive statistics.

    Alternative Hypothesis	Rejection Region
    Ha: μ1 ≠ μ2	|T| > t1-α/2,ν
    Ha: μ1 > μ2	T > t1-α,ν
    Ha: μ1 < μ2	T < tα,ν

    Parameters
    ----------
    x, y : 1 dimensional array_like
        The arrays must have the same shape, except in the dimension.
    alternative: string - (two-sided, less, bigger)
    mu: integer number
    data : Pandas Dataframe or Dictionaries
        Must contain only 2 keys
    paired : Boolean Variable
    var_equal : Boolean Variable
    alpha : Integer number
        The value must correspond to ]0,1[
    effect_size: Boolean Variable
        Calculated facing https://doi.org/10.3389/fpsyg.2013.00863
    mean_difference: Boolean Variable
    confidence_interval : Boolean Variable

    Returns
    -------
    TTest object
    """
    xname = 'x'
    yname = 'y'
    ttest = 'Student\'s t'

    if data is not None:
        if len(dict(data).keys()) == 2:
            x, y = dict(data).values()
            xname = list(dict(data).keys())[0]
            yname = list(dict(data).keys())[1]
        elif len(dict(data).keys()) == 1:
            x = list(dict(data).values())[0]
            xname = dict(data).keys()
        else:
            raise ValueError(f'Oops, I Caught An Error! DATA bigger than 2.')

    if y is None:
        n = np.size(x)  # Number of Samples
        v = n - 1  #  Degrees of Freedom

        df = np.mean(x) - mu  # Difference
        s = np.var(x, ddof=1)  # Sample Variance

        denom = np.sqrt(s/n)  # Standart Error

        method = 'One Samples T-Test'
    else:
        if paired:
            n = np.size(x)  # Number of Samples
            v = n - 1  # Degrees of Freedom

            d = (x - y).astype(np.float64)  # Difference
            sd = np.var(d, ddof=1)  #  Difference Variance
            df = np.mean(d)  # Mean Difference 

            denom = np.sqrt(sd / n)  # Standart Error

            method = 'Paired Sample T-Test'
        else:
            if var_equal:
                n1 = np.size(x)  # Number of elements in sample 1
                n2 = np.size(y)  # Number of elements in sample 2
                v = n1 + n2 - 2  # Degrees of Freedom

                s1 = np.var(x, ddof=1) # Variance of sample 1
                s2 = np.var(y, ddof=1) # Variance of sample 2

                sp =  pooledVar(x, y) # Pooled Variance

                df = np.mean(x) - np.mean(y)  # Mean Difference
                denom = np.sqrt(sp*(1/n1+1/n2))  # Standart Error

                method = 'Independent Samples T-Test'
            else:
                n1 = np.size(x)  # Number of elements in sample 1
                n2 = np.size(y)  # Number of elements in sample 2

                s1 = np.var(x, ddof=1) # Variance of sample 1
                s2 = np.var(y, ddof=1) # Variance of sample 2

                q1 = s1/n1
                q2 = s2/n2
                v = (q1+q2)**2/(q1**2/(n1-1)+(q2**2/(n2-1)))

                df = np.mean(x) - np.mean(y)  # Mean Difference
                denom = np.sqrt(q1+q2)  # Standart Error

                method = 'Independent Samples T-Test'
                ttest = 'Welch\'s t'  
    
    with np.errstate(divide='ignore', invalid='ignore'):
        statistic = np.divide(df, denom)  # Statistic
    
    p = _ttest_distribution(alternative, statistic, v)
    CI = _ttest_critic(alpha/2, v)*denom * np.array([-1,1]) + df if alternative == 'two-sided' else _ttest_critic(alpha, v)*denom * np.array([-np.inf,1]) + df if alternative == 'less' else _ttest_critic(alpha, v)*denom * np.array([-1,np.inf]) + df
    cohensD =  statistic/np.sqrt(n) if y is None or 'Paired' in method else statistic*np.sqrt(1/n1+1/n2)

    results = {'x': xname, 'y': yname, 'ttest':ttest, 'stats':statistic,'alpha':alpha,
               'v':v, 'pvalue':p, 'mean_difference':df, 'CI_l':CI[0], 'CI_u':CI[-1],
               'CohensD':cohensD, 'method':method, 'mu':mu, 'alternative':alternative}
    
    return TTest(results, effect_size, mean_difference, confidence_interval)

def correlation(x=None, y=None, alternative='two-sided', data=None, alpha=0.05, confidence_interval=False):
    """
    Correlation test measures the strength of a linear association between two variables.

    Parameters
    ----------
    x, y : 1 dimensional array_like
        The arrays must have the same shape, except in the dimension.
    alternative: string - (two-sided, less, bigger)
    data : Pandas Dataframe or Dictionaries
    alpha : Integer number
        The value must correspond to ]0,1[
    confidence_interval : Boolean Variable

    Returns
    -------
    Correlation object
    """
    
    names = ['x', 'y']

    if data is not None:
        n = len(list(dict(data).values())[0])
        keys = list(dict(data).keys())
        r = list()
        for i in keys:
            for j in keys:
                num = sum((data[i]-data[i].mean())*(data[j]-data[j].mean()))
                denom = (sum((data[i]-data[i].mean())**2)*sum((data[j]-data[j].mean())**2))**.5
                r.append(num/denom)
    
        r = np.reshape(r, (data.shape[1],data.shape[1]))

        names = keys

    else:
        n = len(x)
        r = list()
        for i in [x,y]:
            for j in [x,y]:
                num = sum((i-i.mean())*(j-j.mean()))
                denom = (sum((i-i.mean())**2)*sum((j-j.mean())**2))**.5
                r.append(num/denom)

        r = np.reshape(r, (2,2))
    
    with np.errstate(divide='ignore', invalid='ignore'):
            statistic = np.sqrt(n-2)*(r/np.sqrt(1-r**2))
        
    v = n - 2
    p = _ttest_distribution(alternative, statistic, v)
    
    fishersz, sigmaz = _fisher_transformation(r,n)
    
    CIu = _z_critic(alpha/2)*sigmaz + fishersz if alternative == 'two-sided' else -np.inf + fishersz if alternative == 'less' else - _z_critic(alpha)*sigmaz + fishersz
    CIl = - _z_critic(alpha/2)*sigmaz + fishersz if alternative == 'two-sided' else _z_critic(alpha)*sigmaz + fishersz if alternative == 'less' else np.inf + fishersz
    CIu = np.tanh(CIu)
    CIl = np.tanh(CIl)

    results = {'names':names, 'r':r, 'stats':statistic, 'alpha':alpha,
               'v':v, 'pvalue':p, 'alternative':alternative, 'CI_l':CIu, 'CI_u':CIl}

    return results

def _fisher_transformation(r,n):
    with np.errstate(divide='ignore', invalid='ignore'):
        fishersz = 0.5 * (np.log(1+r)-np.log(1-r))
        sigmaz = 1/np.sqrt(n-3)
    
    return fishersz, sigmaz

def ftest(x, y, alternative = "two.sided", alpha = 0.95):
    pass

def descriptives(x):
    n = np.size(x)
    miss = np.isnan(x).sum()
    mean = np.mean(x)
    median = np.median(x)
    variance = np.var(x, ddof=1)
    minimum = np.min(x)
    maximum = np.max(x)

    _table = [['N', n],['Missing',miss],['Mean',mean],['Median',median],
              ['Variance', variance],['Minimum',minimum],['Maximum',maximum]]

    print(_table)

def _z_critic(alpha):
    return norm.ppf(1.0 - alpha)  # calculate the critical value

def _z_distribution(alternative, statistic, v):
    tails = {'two-sided': lambda statistic, v: (1.0 - norm.cdf(abs(statistic))) * 2.0,
             'less': lambda statistic, v: (1.0 - norm.cdf(abs(statistic))),
             'greater': lambda statistic, v: norm.cdf(abs(statistic))}
    return tails[alternative](statistic, v)

def _ttest_critic(alpha, v):
    return t.ppf(1.0 - alpha, v)  # calculate the critical value

def _ttest_distribution(alternative, statistic, v):
    tails = {'two-sided': lambda statistic, v: (1.0 - t.cdf(abs(statistic), v)) * 2.0,
             'greater': lambda statistic, v: (1.0 - t.cdf(statistic, v)),
             'less': lambda statistic, v: t.cdf(statistic, v)}
    return tails[alternative](statistic, v)