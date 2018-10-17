import numpy as np

#create a set of new data for kNN.
#classes = 'pos' and 'neg'.
#apspects = lognormal and poisson.
#size = 100.

def new_data():
    cat_names = ['A', 'B']
    cat_pos = list()
    cat_neg = list()
    data = list()

    for i in range(50):
        h = ((cat_names[0]),
            np.round(np.random.lognormal(4, 1),2),
            np.random.poisson(16))
        cat_pos.append(h)

    for i in range(50):
        h = ((cat_names[1]),
            np.round(np.random.lognormal(2, 1),2),
            np.random.poisson(3))
        cat_neg.append(h)

    keys = ('class', 'aspect_1', 'aspect_2')

    data_pos = [dict(zip(keys, v)) for v in cat_pos]
    data_neg = [dict(zip(keys, v)) for v in cat_neg]
    data = data_pos + data_neg

    return data
