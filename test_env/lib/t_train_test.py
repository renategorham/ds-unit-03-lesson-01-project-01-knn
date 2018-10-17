import random
import numpy as np
import pandas as pd
import csv


#separate a proportional training set
#data is a list of dictionaries
#y the name of the response or class
#size is the proportion reserved to the training set (default = 0.2)

def train_test(data,y,size=0.2):
    y_list = list()
    for i in data:
        y_value = (i[y])
        y_list.append(y_value)
        y_set = list(set(y_list))

#y_1 and y_2 are the count of each class
    y_1 = y_list.count(y_set[-1])
    y_2 = y_list.count(y_set[0])
    y_total = y_1 + y_2

    p_1 = y_1 / y_total
  
#the list of dictionaries are shuffled (in case there were ordered)
    random.shuffle(data)
    random.shuffle(data)
    random.shuffle(data)

#calcuate size of sets to return - use ceiling and abolute value to prevent fractions
    #of training and test set sizes
    train_size = np.ceil(y_total * size).astype(int)

#size of training set responses
    train_y_1 = np.ceil(train_size * p_1).astype(int)
    train_y_2 = np.abs(train_size - train_y_1).astype(int)

#slice dataframe by set sizes 
    for i in data:
        if (i[y] == y_set[-1]):
           list_train_y_1 = data[:train_y_1]
           list_test_y_1 = data[train_y_1:y_1]
        else:
            list_train_y_2 = data[:train_y_2]
            list_test_y_2 = data[train_y_2:y_2]

#contact the training and test sets, then shuffle for no paticular reason
    train = list_train_y_1 + list_train_y_2
    test = list_test_y_1 + list_test_y_2

    random.shuffle(train)
    random.shuffle(test)

    for i in train:
        

     df_train = pd.DataFrame(train)
     df_test = pd.DataFrame(test)

     df_train.to_csv('./data/train.csv')
     df_test.to_csv('./data/test.csv')

    # with open('./data/train.csv', 'w', newline='') as out_fp:
    #     writer = csv.writer(out_fp)
    #     for d in train:
    #         writer.writerows(d.items())

    # with open('./data/test.csv', 'w', newline='') as out_fp:
    #     writer = csv.writer(out_fp)
    #     for d in test:
    #         writer.writerows(d.items())

    return train, test
    
    
    
