import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import random
from random import randint
from glob import  glob
from cvxopt import matrix
from cvxopt import solvers
from tabulate import tabulate
from show_results import Show_Results
from allocate_weight import Allocate_weight
from test_OS import Test_OS
def get_data(fileList):
    m = pd.DataFrame()
    for file in fileList:
        tempDf = pd.read_csv(file, parse_dates=['datetime'], index_col=3)
        # print(file,tempDf)
        tempPnl = tempDf[['value']]
        tempPnl = tempPnl[tempPnl.index.dayofweek < 5]
        tempPnl['ret'] = (tempPnl.value - tempPnl.value.shift(1))
        tempPnl = tempPnl[['ret']].resample("1D").apply(lambda x: x.sum() if len(x) else np.nan).dropna(how="all")
        # print("strat " + file[9:], calculate_sharp(merge=tempPnl))
        if len(m) == 0:
            m = tempPnl
        else:
            m = pd.merge(m, tempPnl, how='inner', left_index=True, right_index=True)
    colList = []
    for i in fileList:
        colList.append(i.split('/')[-1][(len(fileList)-4):-4])
    m.columns = colList
    print(m)
    # m = m['2020-09-1':'2021-9-1']
    m = m / 1e6
    return m

if __name__ == '__main__':

    path='/home/hoainam/Desktop/allocation_weight_gng/data/'
    sample = "OS"
    list_group = glob(path + '{}/group1/*.csv'.format(sample))
    list_group.sort()
    list_normal = glob(path + '{}/group2/*.csv'.format(sample))
    list_normal.sort()
    fileList = list_group + list_normal
    print(fileList)

    allocate_weight=Allocate_weight()
    print(allocate_weight.max_sharpe_upperbound_and_bound_group(list_group=list_group,list_normal=list_normal,
                                                                dataframe=get_data(fileList),upperbound=0.36, bounded_list=0.45))
