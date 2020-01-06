from tabulate import tabulate

class Summary:
    def __init__(self, results):
        self.test = results['test'] 
        self._genTable(results)

    def _genTable(self, results):
        if self.test == 'tstats':
            self.pvalue = results['pvalue']
            self.statistic = results['statistic']
            self.df = results['df']
            self.xmean = results['xmean']
            self.ymean = results['ymean']
            self._header = ['','statistic','df','p']
            self._table = [['Student\'s t', self.statistic, self.df, self.pvalue]]
            self._summary = tabulate(self._table, self._header, tablefmt="simple")  
        else:
            pass
        
    def convert(self, format='latex'):
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
            return tabulate(self._table, self._header, tablefmt='latex')
        
        elif format == 'html':
            return '<table border="1" class="dataframe">' + tabulate(self._table, self._header, tablefmt='html')[7:]
        
        else:
            raise Exception(r'Error. {} not in [latex, html].'.format(format))

    def summary(self):
        """
        Prints the statistic summary table


        Returns
        -------
        Summary table
        """
        print(self._summary+'\n')
        print(tabulate([[self.xmean,self.ymean]], ['X Mean', 'Y Mean'], tablefmt="simple"))

    def export(self, name, format='latex'):
        """
        Export the summary table into others formats.
        

        Parameters
        ----------
        format : {'simple','latex', 'html'}

        Returns
        -------
        Exports the table string on the required format

        """

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
        
        else:
            raise Exception(r'Error. {} not in [simple, latex, html].'.format(format))
