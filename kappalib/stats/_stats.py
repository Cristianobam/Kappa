import numpy as np
from ._summary import Summary
from scipy.stats import t, f, chi

__all__ = ['pooledVar', 'ttest', 'ftest']


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



tails = {'two-sided': lambda statistic, v: (1.0 - t.cdf(abs(statistic), v)) * 2.0,
             'less': lambda statistic, v: (1.0 - t.cdf(abs(statistic), v)),
             'bigger': lambda statistic, v: t.cdf(abs(statistic), v)}

if alternative not in tails:
        raise ValueError(f'{alternative} is not a valid input.')

def ttest(x, y=None, alternative="two-sided", mu=0, paired=False, var_equal=False, alpha=0.95, effect_size=False):
    """
    T-test for means of two samples from descriptive statistics.
    This is a two-sided test for the null hypothesis that two independent
    samples have identical average (expected) values.

    Alternative Hypothesis	Rejection Region
    Ha: μ1 ≠ μ2	|T| > t1-α/2,ν
    Ha: μ1 > μ2	T > t1-α,ν
    Ha: μ1 < μ2	T < tα,ν

    Parameters
    ----------
    x, y : 1 dimensional array_like
        The arrays must have the same shape, except in the dimension.

    Returns
    -------
    statistic : float or array
        The calculated t-statistic.
    pvalue : float or array
        The two-tailed p-value.
    v : degrees of freedom
        The calculated degrees of freedom (floor rounded)
    x_mean : x sample mean
        The calculated x mean
    y_mean : y sample mean
        The calculated y mean
    """

    if y != None:
        n = np.size(x)  # Number of Samples
        v = n - 1  #  Degrees of Freedom

        df = np.mean(x) - mu  # Difference
        s = np.var(x, ddof=1)  # Sample Variance

        denom = np.sqrt(s/n)  # Standart Error

        method = 'One Sample t-test'
    else:
        if paired:
            n = np.size(x)  # Number of Samples
            v = n - 1  # Degrees of Freedom

            d = (x - y).astype(np.float64)  # Difference
            sd = np.var(d, ddof=1)  #  Difference Variance
            df = np.mean(d)  # Difference Mean

            denom = np.sqrt(sd / n)  # Standart Error

            method = 'Paired t-test'
        else:
            if var_equal:
                n1 = np.size(x)  # Number of elements in sample 1
                n2 = np.size(y)  # Number of elements in sample 2
                v = n1 + n2 - 2  # Degrees of Freedom

                s1 = np.var(x, ddof=1) # Variance of sample 1
                s2 = np.var(y, ddof=1) # Variance of sample 2

                sp =  pooledVar(x, y) # Pooled Variance

                df = np.mean(x) - np.mean(y)  # Difference
                denom = np.sqrt(sp*(1/n1+1/n2))  # Standart Error

                method = 'Two Sample t-test'
            else:
                n1 = np.size(x)  # Number of elements in sample 1
                n2 = np.size(y)  # Number of elements in sample 2


                s1 = np.var(x, ddof=1) # Variance of sample 1
                s2 = np.var(y, ddof=1) # Variance of sample 2

                q1 = s1/n1
                q2 = s2/n2
                v = (q1+q2)**2/(q1**2/(n1-1)+(q2**2/(n2-1)))

                df = np.mean(x) - np.mean(y)  # Difference
                denom = np.sqrt(q1+q2)  # Standart Error

                method = 'Welch Two Sample t-test'
    
    with np.errstate(divide='ignore', invalid='ignore'):
        statistic = np.divide(df, denom)  # Statistic

    return TTest()


def ftest(x, y, alternative = c("two.sided", "less", "greater"), alpha = 0.95):
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


def _ttest_distribution(v, statistic,alpha):
    cv = t.ppf(1.0 - alpha, v)  # calculate the critical value
    p = (1.0 - t.cdf(abs(statistic), v)) * 2.0  # calculate the p-value
    return cv, p
