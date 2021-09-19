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
class Test_OS():
    def __init__(self):
        pass
    def test_weight(self,weight,df_test):
        merge=Show_Results.merge(dataframe=df_test, weight=weight)
        Show_Results.plot_merge(merge, booksize=1e9)


