from tabulate import tabulate
from kappalib.plot import *
#class ANOVA():
#class Correl():
#class 

class TTest():  
  __slots__ = ['_x', '_y', '_ttest', '_stats', '_v', '_pvalue','_alpha',
                '_mean_difference', '_CI_l', '_CI_u', '_CohensD', 
                '_method','_header','_body','_mu', '_alternative']

  def __init__(self,result, effect_size, mean_difference, confidence_interval):
      results = result.copy()
      for key, value in results.items():
          self.__setattr__('_' + key, value)
      del results['mu']
      del results['alternative']
      del results['method']
      del results['alpha']
      self._set_table(results, effect_size, mean_difference, confidence_interval)
      
  def stats(self):
      return self._stats

  def d(self):
      return self._CohensD

  def p_value(self):
      return self._pvalue

  def mean_difference(self):
      return self._mean_difference

  def degrees_fredom(self):
      return self._v

  def confidence_interval(self):
      return (self._CI_l, self._CI_u)

  def __getattr__(self, attr):
      raise ValueError(f'Oops, I Caught An Error! {attr.upper()} was not defined.')

  def _set_table(self,results, effect_size, mean_difference, confidence_interval):
      if not effect_size:
          del results['CohensD']
      if not mean_difference:
          del results['mean_difference']
      if not confidence_interval:
          del results['CI_l']
          del results['CI_u']
      if 'paired' not in self._method.lower():
          del results['y']
      
      header_keys = {'x':'','y':'','ttest':'','stats':'statistic','v':'df',
                    'pvalue':'p','mean_difference':'Mean Difference',
                      'CohensD':'Cohen\'s d', 'CI_l':'Lower', 'CI_u':'Upper'}
      

      self._header = [header_keys[i] for i in results.keys()]
      
      self._body = list(results.values())
      index = list(results).index('pvalue')
      self._body[index] = '<0.001' if self._body[index] < 0.001 else self._body[index]
      
  def summary(self):
      print(self._method)
      print(tabulate([self._body], self._header, tablefmt="psql", floatfmt=".3f", stralign='center'))
      if self._mu is not None and self._mu != 0:
          print('Note. ' + chr(956)+ '=' + str(self._mu))
      if self._alternative == 'less':
          print('Note. H0: Group 1 < Group 2')
      elif self._alternative == 'greater': 
          print('Note. H0: Group 1 > Group 2')
    
  def summary_convert(self, format='latex'):
      """
      Convert the summary table into others formats.
      

      Parameters
      ----------
      format : {'latex', 'html'}

      Returns
      -------
      Table string on the required format

      """
      if format == 'latex':
          print(self._method)
          print(tabulate([self._body], self._header, tablefmt="latex", floatfmt=".3f", stralign='center'))
          
          if self._mu is not None and self._mu != 0:
            print('Note.' + r'$\mu = $' + str(self._mu))
          if self._alternative == 'less':
            print(r'Note. $H_0$: Group 1 < Group 2')
          elif self._alternative == 'greater': 
            print(r'Note. $H_0$: Group 1 > Group 2')
      
      elif format == 'html':
          print('<table border="1" class="dataframe">' + tabulate([self._body], self._header, tablefmt="html", floatfmt=".3f", stralign='center'))
      
      else:
          raise Exception(r'Error. {} not in [latex, html].'.format(format))

class Correlation(): 
    __slots__ = ['_names', '_r', '_stats', '_v', '_pvalue', '_header',
                 '_body','_alternative', '_CI_l', '_CI_u', '_alpha']

    def __init__(self,result, confidence_interval):
        results = result.copy()
        for key, value in results.items():
            self.__setattr__('_' + key, value)
        del results['alternative']
        self._set_table(results, confidence_interval)
    
    def confidence_interval(self):
        return (self._CI_l, self._CI_u)    
        
    def r(self):
        return self._r

    def p_value(self):
        return self._pvalue

    def degrees_fredom(self):
        return self._v

    def __getattr__(self, attr):
        raise ValueError(f'Oops, I Caught An Error! {attr.upper()} was not defined.')
    
    def _set_table(self,results, confidence_interval):
        
        _r = np.tril(results['r'])
        _p = np.tril(results['pvalue'])

        self._body = list()
        self._header = ['',''] + results['names']
        
        if confidence_interval:
            stats_descript = 'Pearson\'s R\np-value\n{0}% CI Upper\n{0}% CI Lower\n'.format((1-results['alpha'])*100)
            _cil = np.tril(results['CI_l'])
            _ciu = np.tril(results['CI_u'])
        
            for j,i in enumerate(results['names']):
                data = list()
                k=0
                for r,p,cu,cl in zip(_r[j],_p[j],_ciu[j],_cil[j]):
                    pvalue = 0 if p == 0 else '<0.001' if p < 0.001 else round(p,3)
                    data.append('{:.3f}\n{}\n{:.3f}\n{:.3f}\n'.format(r,pvalue,cu,cl) if j>k else '-\n-\n-\n-\n')
                    k += 1
                self._body.append([i,stats_descript]+data)
                
        else:
            stats_descript = 'Pearson\'s R\np-value\n'
            
            for j,i in enumerate(results['names']):
                data = list()
                k=0
                for r,p in zip(_r[j],_p[j]):
                    pvalue = 0 if p == 0 else '<0.001' if p < 0.001 else round(p,3)
                    data.append('{:.3f}\n{}\n'.format(r,pvalue) if j>k else '-\n-\n')
                    k += 1
                self._body.append([i,stats_descript]+data)
        
    def summary(self):
        print('Correlation Matrix')
        print(tabulate(self._body, self._header, tablefmt="grid", floatfmt=".3f", stralign='center'))
        if self._alternative == 'less':
                print(r'Note. H0: negative correlation')
        elif self._alternative == 'greater': 
                print(r'Note. H0: positive correlation')
    
    def summary_convert(self, format='latex'):
        """
        Convert the summary table into others formats.
      

        Parameters
        ----------
        format : {'latex', 'html'}

        Returns
        -------
        Table string on the required format

        """
        
        if format == 'latex':
            print('Correlation Matrix')
            print(tabulate(self._body, self._header, tablefmt="latex", floatfmt=".3f", stralign='center'))
            
            if self._alternative == 'less':
                print(r'Note. $H_0$: negative correlation')
            elif self._alternative == 'greater': 
                print(r'Note. $H_0$: positive correlation')
                
        elif format == 'html':
            print('<table border="1" class="dataframe">' + tabulate(self._body, self._header, tablefmt="html", floatfmt=".3f", stralign='center'))
            
        else:
            raise Exception(r'Error. {} not in [latex, html].'.format(format))
            
    def matrix(self,ax=None, aspect='equal', cmape='viridis', **kwargs):
        pass