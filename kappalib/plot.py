import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'whitesmoke'
plt.rcParams['legend.facecolor'] = 'none'
plt.rcParams['legend.edgecolor'] = 'none'

def scatter(x,y,jitter=.75,errBar='std',facecolor='none',edgecolor='firebrick',alpha=1,s=50):
  np.random.seed(23)
  X = x + np.random.randn(*np.shape(x))*jitter/10
  plt.scatter(X,y,facecolor=facecolor,edgecolor=edgecolor,alpha=alpha,s=s)