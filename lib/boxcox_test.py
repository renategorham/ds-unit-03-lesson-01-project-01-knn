
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as pl

actual_set = pd.read_csv('../data/wdbc.csv')

#lmbda = stats.boxcox(actual_set['mean_concave_points'])

#print(len(lmbda[0]))
print(min(actual_set['mean_concave_points']))
