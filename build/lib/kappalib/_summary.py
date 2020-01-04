from tabulate import tabulate

class Summary:
    def __init__(self, results):
        self.test = results['test']
        self.pvalue = results['pvalue']
        self.statistic = results['statistic']
        self.df = results['df']
        self.xmean = results['xmean']
        self.ymean = results['ymean']
        self._genTable()

    def _genTable(self):
        if self.test == 'tstats':
            self._header = ['','statistic','df','p']
            self._table = [['Student\'s t', self.statistic, self.df, self.pvalue]]
            self._table = tabulate(self._table, self._header, tablefmt="simple")
        else:
            pass
    
    def table(self):
        print(self._table)
        
    def convert(self, format='latex'):
        if format == 'latex':
            return tabulate(self._table, self._header, tablefmt='latex')

    def summary(self):
        print(tabulate(self._table, self._header, tablefmt="simple"))
        print(tabulate([[self.xmean,self.ymean]], ['X Mean', 'Y Mean'], tablefmt="simple"))

    def export(self, name, format='latex'):
        if format == 'latex':
            with open(f'{name}.tex', 'w') as f:
                for i in tabulate(self._table, self._header, tablefmt='latex'):
                    f.write(i)
        
        elif format == 'simple':
            with open(f'{name}.txt', 'w') as f:
                for i in tabulate(self._table, self._header, tablefmt='simple'):
                    f.write(i)
        
        elif format == 'html':
            with open(f'{name}.html', 'w') as f:
                for i in tabulate(self._table, self._header, tablefmt='html'):
                    f.write(i)
