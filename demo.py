import numpy as np
import kappalib as kp
import scipy.stats as ss

n = 35
x = np.random.randn(n)
y = x + 2

print(ss.ttest_ind(x,y))
t = kp.stats.ttest(x,y)