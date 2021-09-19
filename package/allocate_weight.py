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
#########
import subprocess
import logging.config
logging.config.fileConfig("./logger.conf")
LOGGER = logging.getLogger("root")
############
class Allocate_weight():
    def __init__(self):
        pass
    def max_sharpe(self,dataframe,print_corr=True):
        if print_corr==True:
            print(tabulate(dataframe.corr(), tablefmt="pipe", headers="keys"))
        else:
            pass
        cov = (dataframe.cov()).to_numpy()
        meanvec = (dataframe.mean()).to_numpy()
        P = matrix(cov, tc='d')
        q = matrix(np.zeros(len(meanvec)), (len(meanvec), 1), tc='d')
        G = (-1) * matrix(np.identity(len(meanvec)))
        H = np.zeros(len(meanvec))
        h = matrix(H, tc='d')
        A = (matrix(meanvec)).trans()
        b = matrix([1], (1, 1), tc='d')
        solvers.options['maxiters'] = 100
        sol = ((solvers.qp(P, q, G, h, A, b))['x'])
        solution = [x for x in sol]
        sum = 0
        for i in range(len(solution)):
            sum += solution[i]
        optimizedWeigh = [x / sum for x in solution]
        return optimizedWeigh
    def max_sharpe_upperbound(self,dataframe,upperbound,print_corr=True):
        if print_corr == True:
            print(tabulate(dataframe.corr(), tablefmt="pipe", headers="keys"))
        else:
            pass
        cov = (dataframe.cov()).to_numpy()
        meanvec = (dataframe.mean()).to_numpy()
        P = matrix(cov, tc='d')
        q = matrix(np.zeros(len(meanvec)), (len(meanvec), 1), tc='d')
        G = []
        for i in range(len(meanvec)):
            k = [0 for x in range(len(meanvec) - 1)]
            k.insert(i, -1)
            G.append(k)
        for i in range(len(meanvec)):
            k = [-upperbound for x in range(len(meanvec) - 1)]
            k.insert(i, 1 - upperbound)
            G.append(k)
        G = matrix(np.array(G))
        print(G)
        H = np.zeros(2 * len(meanvec))
        h = matrix(H, tc='d')
        A = (matrix(meanvec)).trans()
        b = matrix([1], (1, 1), tc='d')
        sol = ((solvers.qp(P, q, G, h, A, b))['x'])
        solution = [x for x in sol]
        sum = 0
        for i in range(len(solution)):
            sum += solution[i]
        optimizedWeigh = [x / sum for x in solution]
        return optimizedWeigh
    def max_sharpe_upperbound_and_fix_weight(self,list_not_fix,list_fix,dataframe,upperbound, list_weight_fix,print_corr=True):
        if print_corr == True:
            print(tabulate(dataframe.corr(), tablefmt="pipe", headers="keys"))
        else:
            pass
        cov = (dataframe.cov()).to_numpy()
        meanvec = (dataframe.mean()).to_numpy()
        P = matrix(cov, tc='d')
        q = matrix(np.zeros(len(meanvec)), (len(meanvec), 1), tc='d')
        G = []
        for i in range(len(list_not_fix)):
            k = [0 for x in range(len(meanvec) - 1)]
            k.insert(i, -1)
            G.append(k)
        for i in range(len(list_not_fix)):
            k = [-upperbound for x in range(len(meanvec) - 1)]
            k.insert(i, 1 - upperbound)
            G.append(k)
        for i in range(len(list_fix)):
            k = [-list_weight_fix[i] for x in range(len(list_not_fix))]
            l = [-list_weight_fix[i] for x in range(len(list_fix) - 1)]
            l.insert(i, 1 - list_weight_fix[i])
            G.append(k + l)
        for i in range(len(list_fix)):
            k = [list_weight_fix[i] for x in range(len(list_not_fix))]
            l = [list_weight_fix[i] for x in range(len(list_fix) - 1)]
            l.insert(i, list_weight_fix[i] - 1)
            G.append(k + l)
        G = matrix(np.array(G))
        # print(G)
        H = np.zeros(2 * len(meanvec))
        h = matrix(H, tc='d')
        A = (matrix(meanvec)).trans()
        b = matrix([1], (1, 1), tc='d')
        sol = ((solvers.qp(P, q, G, h, A, b))['x'])
        solution = [x for x in sol]
        sum = 0
        for i in range(len(solution)):
            sum += solution[i]
        optimizedWeigh = [x / sum for x in solution]
        return optimizedWeigh
    def max_sharpe_upperbound_and_bound_group(self,list_group,list_normal,dataframe,upperbound, bounded_list):
        cov = (dataframe.cov()).to_numpy()
        meanvec = (dataframe.mean()).to_numpy()
        P = matrix(cov, tc='d')
        # print(P)
        q = matrix(np.zeros(len(meanvec)), (len(meanvec), 1), tc='d')
        G = []
        for i in range(len(meanvec)):
            k = [0 for x in range(len(meanvec) - 1)]
            k.insert(i, -1)
            G.append(k)
        for i in range(len(meanvec)):
            k = [-upperbound for x in range(len(meanvec) - 1)]
            k.insert(i, 1 - upperbound)
            G.append(k)
        k = [-bounded_list for i in range(len(list_normal))]
        for i in range(len(list_group)):
            k.insert(i, 1 - bounded_list)
        G.append(k)
        k = [bounded_list for i in range(len(list_normal))]
        for i in range(len(list_group)):
            k.insert(i, bounded_list - 1)
        G.append(k)
        G = matrix(np.array(G))
        H = np.zeros(2 * len(meanvec) + 2)
        h = matrix(H, tc='d')
        A = (matrix(meanvec)).trans()
        b = matrix([1], (1, 1), tc='d')
        print('G', G)
        print('h', h)
        print('A', A)
        print('b', b)
        sol = (solvers.qp(P, q, G, h, A, b))['x']
        solution = [x for x in sol]
        sum = 0
        for i in range(len(solution)):
            sum += solution[i]
        optimizedWeigh = [x / sum for x in solution]
        return optimizedWeigh
    def gng_simulation(self):
        a = subprocess.run(['Rscript', 'GNG-FIT.R'], stdout=subprocess.PIPE)
        LOGGER.info("Xong...")
