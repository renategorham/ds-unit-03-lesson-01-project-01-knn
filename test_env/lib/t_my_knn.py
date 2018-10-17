import numpy as np
import pandas as pd

#Calculates k nearest neighbors for all possible numbers of k
#train_set is the set of known aspects and classes as returned for train_test.py (list of dict)
#test_set is the set of unknown aspects wiht or without classes as returned for train_test.py (list of dict)
#the test set may contained known classes to be used in accuracy annalysis and optimal k selection.
#y is the key of classes or responses
#aspect_1 is key of the fist aspect in the sets and dictionaries
#aspect_2 is the key of the second aspect in the sets and dictionaries
#pos_class is the value of the class identified as positive
#neg_class is the value if the class idnetified as negative

def knn(train_set, test_set, y, aspect_1, aspect_2, pos_class, neg_class):

    train = list()
    test = list()
    pos = pos_class 
    neg = neg_class
    class_list = [pos, neg]


#restrict the number keys in dictionaries for train and test sets
    train = [{'id': i[0], 'class':i[1][y], 
              'obs':[i[1][aspect_1],i[1][aspect_2]]} 
               for i in enumerate(train_set)]

    test = [{'id': i[0], 'class':i[1][y], 
             'obs':[i[1][aspect_1],i[1][aspect_2]]} 
              for i in enumerate(test_set)]

    #three lists are used to manipluate and hold data during iterations
    #u, in the iteration below is the calcuation of the magnitidue
    d = list()
    g = list()
    r = list()
    for i in range(len(train[0]['obs'])):
        for j in enumerate(train):
            for k in enumerate(test):
                if (i+1<len(train[0]['obs'])):
                    u = ({'id':k[1]['id'],
                          'mag':np.sqrt((j[1]['obs'][i] - k[1]['obs'][i])**2+(j[1]['obs'][i+1] - k[1]['obs'][i+1])**2),
                          'train_class':j[1]['class'],
                          'test_class':k[1]['class']})
                    d.append(u)
                else:
                    break

                d = sorted(d, key=lambda m: (m['id'], m['mag']))   


    r = [(i['id'], i['test_class'], i['train_class'], i['mag']) for i in d]
    
#dictionaries are transformed to data frames for simplification of some analysis
    df = pd.DataFrame(r, columns=['id', 'test_class', 'train_class', 'mag'])
    df['k'] = df.groupby('id').cumcount() + 1
    for i in class_list:
        df[i] = np.nan
        for j in range(len(i)):
            df[i] = np.where(df.loc[:,'train_class'] == i[j], df.groupby(['id','train_class'])['mag'].cumsum() / (df.groupby(['id','train_class']).cumcount() + 1),
               np.nan)
            df[i].ffill(inplace=True)
            df[i].bfill(inplace=True)

    df['result_by_mag'] = df[[pos, neg]].idxmin(axis=1)
    
#the dataframe is written to a csv for further analysis
    df.to_csv('./data/analyzed.csv')

    return df