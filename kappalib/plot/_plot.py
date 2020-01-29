import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import t

plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'whitesmoke'
plt.rcParams['legend.facecolor'] = 'none'
plt.rcParams['legend.edgecolor'] = 'none'

__all__ = ['scatter', 'heatmap', 'linear_regression']

def scatter(x,y,jitter=.75,errBar='std',facecolor='none',edgecolor='firebrick',alpha=1,s=50,**kwargs):
  np.random.seed(23)
  X = x + np.random.randn(*np.shape(x))*jitter/10
  plt.scatter(X,y,facecolor='none',edgecolor='firebrick',alpha=1,s=50,**kwargs)

def heatmap(data, annot=False, fmt=2, linewidth=3, linecolor='white', 
            txtcolor=['white','black'], beta=.25, cbar=True, xticklabels='auto',
            yticklabels='auto', ax=None, **kwargs):

    if 'cmap' not in kwargs:
        kwargs['cmap'] = 'RdBu_r'
    
    if 'origin' not in kwargs:
        kwargs['origin'] = 'lower'
        
    with plt.rc_context({'axes.edgecolor': 'white', 'axes.linewidth': 0}):  
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        else:
            ax.axes.set_frame_on(False)
            
        img = ax.imshow(data, **kwargs)
        
        xsize = np.shape(data)[0]
        ysize = np.shape(data)[-1]
        
        ax.set_yticks(range(ysize))
        ax.set_xticks(range(xsize))

        if xticklabels == 'auto' and yticklabels == 'auto':
            yticklabels = range(ysize)
            xticklabels = range(xsize)
            ax.set_yticklabels(yticklabels)
            ax.set_xticklabels(xticklabels)
        elif isinstance(xticklabels,list) and yticklabels == 'auto':
            yticklabels = range(ysize)
            ax.set_yticklabels(yticklabels)
            ax.set_xticklabels(xticklabels)
        elif isinstance(yticklabels,list) and xticklabels == 'auto':
            xticklabels = range(xsize)
            ax.set_yticklabels(yticklabels)
            ax.set_xticklabels(xticklabels)
        else:
            ax.set_yticklabels(yticklabels)
            ax.set_xticklabels(xticklabels)

        ax.set_xticks(np.arange(-.5,xsize+.5,1),minor=True)
        ax.set_yticks(np.arange(-.5,ysize+.5), minor=True)
        ax.grid(which='minor', color=linecolor, linestyle='-', linewidth=linewidth)

        plt.colorbar(img)
        
        ax.tick_params(axis='both', which='both', length=0)
    
    if isinstance(annot, bool):
        if annot == True:
            max_data = np.max(abs(data))
            for k,i in enumerate(range(xsize)):
                for l,j in enumerate(range(ysize)):
                    num = round(data[k,l], fmt)
                    ax.text(i,j,num,horizontalalignment='center',
                            verticalalignment='center',
                            color=txtcolor[0] if abs(num)>=max_data*(1-beta) else txtcolor[-1])

def linear_regression(x,y,data=None,interceptor=True,plot=False,confidence_interval=False,
                     ax=None,grid=True,linecolor='firebrick',facecolor=None,
                      edgeline=True,alpha=.05,**kwargs):
    
    if 'facecolor' not in kwargs:
        kwargs['facecolor'] = 'none'
    
    if 'edgecolor' not in kwargs:
        kwargs['edgecolor'] = 'steelblue'
        
    if facecolor is None:
        facecolor = kwargs['edgecolor']
        
    if 's' not in kwargs:
        kwargs['s'] = 50
        
    if 'label' not in kwargs:
        kwargs['label'] = 'Sample Data'
        
    if isinstance(interceptor,bool):
        if interceptor == True:
            x = np.vstack([x,np.ones_like(x)]).T
    
    results = np.linalg.lstsq(x,y,rcond=None)
    results = {'coefficients':results[0],'residuals':results[1],
                   'rank':results[2],'s':results[-1]}
    
    line = lambda x,m,c: m*x+c
    xx = np.arange(min(x[:,0]),max(x[:,0])+1,1)
    
    if isinstance(confidence_interval, bool):
        if confidence_interval == True:
            yerr = y - line(x[:,0],*results['coefficients'])
            xmean = np.mean(x[:,0])
            n = len(x)
            t_critic = t.ppf(1.0 - alpha/2, n-2)
            s = np.var(yerr, ddof=2)
            conf = t_critic * np.sqrt(s*(1/n+(xx-xmean)**2/(sum((xx-xmean)**2))))
            Y = line(xx,*results['coefficients'])
            results['CI'] = Y + np.array([-abs(conf),abs(conf)])
    
    if isinstance(plot,bool):
        if plot == True:
            if ax is None:
                fig, ax = plt.subplots()
                
            yy = line(xx,*results['coefficients'])
            ax.plot(xx,yy,color=linecolor,label='Fitted')
            ax.scatter(x[:,0],y,**kwargs)
            ax.grid(grid,color='grey',alpha=.3,linestyle='dashed')
            
            if confidence_interval == True:
                if isinstance(edgeline, bool):
                        if edgeline == True:
                            ax.plot(xx,results['CI'][0],linestyle='dashed',color='grey',
                                    linewidth=.9)
                            ax.plot(xx,results['CI'][1],linestyle='dashed',color='grey',
                                    linewidth=.9)
                        else:
                            ax.fill_between(xx,results['CI'][0],results['CI'][1],alpha=.2,
                                facecolor=facecolor) 
    return results