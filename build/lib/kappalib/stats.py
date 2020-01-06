import matplotlib.pyplot as plt
from kappalib._summary import *
import numpy as np
from scipy.stats import t

def ttest(x, y, test='2sample', alpha=.05):
    """
    T-test for means of two samples from descriptive statistics.
    This is a two-sided test for the null hypothesis that two independent
    samples have identical average (expected) values.


    Parameters
    ----------
    x, y : 1 dimensional array_like
        The arrays must have the same shape, except in the dimension.
    test : {'1sample', '2sample', 'paired', '1','2','3},
        Defines how to handle ttest function.
        The following options are available (default is '2sample'):
          * '1sample', '1': *two independent* samples T-test
          * independent, '2sample', '2': *two unrelated* samples T-test
          * 'paired', '3': *two related* samples T-test

    Returns
    -------
    statistic : float or array
        The calculated t-statistic.
    pvalue : float or array
        The two-tailed p-value. 
    df : degrees of freedom
        The calculated degrees of freedom (floor rounded)
    x_mean : x sample mean
        The calculated x mean
    y_mean : y sample mean
        The calculated y mean
    """

    if test in ['1','1sample']:
        
        n = np.size(x)
        df = n - 1

        d = np.mean(x) - y
        v = np.var(x, ddof=1)

        denom = np.sqrt(v / n)

    elif test in ['2','2sample','independent']:
        
        n1 = np.size(x)
        n2 = np.size(y)

        d = (x - y).astype(np.float64)
        v1 = np.var(x, ddof=1)
        v2 = np.var(y, ddof=1)
        df = (v1/n1+v2/n2)**2/((v1/n1)**2/(n1-1)+(v2/n2)**2/(n2-1))

        dm = np.mean(d)
        denom = np.sqrt(v1/n1+v2/n2)

    elif test in ['3','paired']:

        n = np.size(x)
        df = n - 1

        d = (x - y).astype(np.float64)
        v = np.var(d, ddof=1)

        dm = np.mean(d)
        denom = np.sqrt(v / n)

    else:
        raise r'Error. {} not in [1sample, 2sample, paired, 1, 2, 3].'.format(test)

    with np.errstate(divide='ignore', invalid='ignore'):
         statistic = np.divide(dm, denom)
    _, p = _ttest_distribution(df,statistic,alpha)
    
    summary = Summary({'test': 'tstats', 'statistic': statistic, 'pvalue':p, 'df':df, 'xmean':x.mean(), 'ymean':y.mean()})
    summary.summary()
    return summary
    
def descriptives(x):
    n = np.size(x)
    miss = np.isnan(x).sum()
    mean = np.mean(x)
    median = np.median(x)
    variance = np.var(x, ddof=1)
    minimum = np.min(x)
    maximum = np.max(x)

    _table = [['N',n],['Missing',miss],['Mean',mean],['Median',median],
              ['Variance',variance],['Minimum',minimum],['Maximum',maximum]]
    
    print(tabulate(_table))


def _ttest_distribution(df,statistic,alpha):
    cv = t.ppf(1.0 - alpha, df) # calculate the critical value
    p = (1.0 - t.cdf(abs(statistic), df)) * 2.0 # calculate the p-value
    return cv, p
