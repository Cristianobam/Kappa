import matplotlib.pyplot as plt
from kappalib._summary import *
import numpy as np
from scipy.stats import t

def ttest(x, y, test='2sample', nan_policy='propagate', alpha=.05):
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

    xmean = x.mean()
    ymean = y.mean()

    d = xmean - ymean

    nx = len(x)
    ny = len(y)

    if test in ['1','1sample']:
        denom = x.std(ddof=1)/np.sqrt(nx)
        df = nx - 1

    elif test in ['2','2sample','independent']:
        xvar = x.var(ddof=1)
        yvar = y.var(ddof=1)

        denom = np.sqrt(xvar/nx+yvar/ny)

        df = (xvar/nx+yvar/ny)**2/((xvar/nx)**2/(nx-1)+(yvar/ny)**2/(ny-1))

    elif test in ['3','paired']:
        
        pass

    else:
        raise r'Error. {} not in [1sample, 2sample, paired, 1, 2, 3].'.format(test)

    with np.errstate(divide='ignore', invalid='ignore'):
         statistic = np.divide(d, denom)
    _, p = _ttest_distribution(df,statistic,alpha)
    
    summary = Summary({'test': 'tstats', 'statistic': statistic, 'pvalue':p, 'df':df, 'xmean':xmean, 'ymean':ymean,})
    return summary
    
def _ttest_distribution(df,statistic,alpha):
    cv = t.ppf(1.0 - alpha, df) # calculate the critical value
    p = (1.0 - t.cdf(abs(statistic), df)) * 2.0 # calculate the p-value
    return cv, p
