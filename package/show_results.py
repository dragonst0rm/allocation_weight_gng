import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Show_Results():
    def __init__(self,booksize=None):
        self.booksize=booksize

    def calculateSharpe(self, npArray):
        sr = npArray.mean() / npArray.std() * np.sqrt(252)
        print(npArray.std())
        return sr

    def max_drawdown(self, booksize, returnSeries):
        mdd = 0
        a = np.cumsum(returnSeries)
        X = a + booksize
        peak = X[0]
        dds = []
        for x in X:
            if x > peak:
                peak = x
            dd = (peak - x) / booksize
            if dd > mdd:
                mdd = dd
                dds.append(X[X == x])
        print("MDD AT ", dds[-1].index[0] if len(dds) else None)
        print(X)
        # returnSeries.to_csv(r'/home/hoainam/repos/rework_backtrader/Strategies/original_alpha_f1m/vn30_LR/daily.csv')
        return mdd

    def merge(self,weight, dataframe):
        merge = []
        counter = 0
        for (columnName, columnData) in dataframe.iteritems():
            # print(real[counter])
            if len(merge) == 0:
                merge = dataframe[columnName] * weight[counter]
            else:
                merge = merge + dataframe[columnName] * weight[counter]
            counter += 1
        return merge

    def plot_merge(self,merge, booksize):
        print('dd,', self.max_drawdown(booksize, merge))
        print('sharpe,', self.calculateSharpe(merge))
        plt.plot(np.cumsum(merge))
        plt.grid(True)
        plt.legend(('old', 'maxsharpe', 'minDD'))
        plt.show()

