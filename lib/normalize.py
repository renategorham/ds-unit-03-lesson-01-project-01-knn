import numpy as np
import pandas as pd 
import scipy.stats as stats


#reset and rescale the data to mean zero and standard deviation of 1.
    #'train_set' is a dataframe containing classes and aspects.
    #'train_class' the column in the dataframe containing classes.
    #'train_aspect_1' and 'train_aspect_2' are the two aspects from which unknows are classified.
    #'test_set' is a dataframe of aspects and unknown classifications.
    #'train_aspect_1' and 'train_aspect_2' are the two aspects of unknow class.
    #'known_test_class' is an optional column of 

def normalize(train_set, test_set, y, aspect_1, aspect_2):
    #reset the basis, fit the data sets into a normal distribution.
#added a small scalar to the datasets to keep results greater than 0.
    train = list()
    test = list()

    for i in train_set:
        new_dict = {'class':i[y], 'a1':i[aspect_1], 'a2':i[aspect_2]}
        train.append(new_dict)
   

    for i in test_set:
        new_dict = {'class':i[y], 'a1':i[aspect_1], 'a2':i[aspect_2]}
        test.append(new_dict)
    
    #known['a1_1'] = [lambda x: float(i['a1']) + 1 for i in known]
    #known['a2_1'] = [float(i['a2']) + 1 for i in known]

    for i in train:
        i['a1_1'] = float(i['a1']) + 1
        i['a2_1'] = float(i['a2']) + 1

    for i in test:
        i['a1_1'] = float(i['a1']) + 1
        i['a2_1'] = float(i['a2']) + 1

    combined = train + test

    a1_combined = [d['a1_1'] for d in combined]
    a2_combined = [d['a2_1'] for d in combined]

    lambda_a1_combined = stats.boxcox(a1_combined)[1]
    lambda_a2_combined = stats.boxcox(a2_combined)[1]

    for i in train:
        i['a1_T'] = i['a1_1']**lambda_a1_combined
        i['a2_T'] = i['a2_1']**lambda_a2_combined
   
    for i in test:
        i['a1_T'] = i['a1_1']**lambda_a1_combined
        i['a2_T'] = i['a2_1']**lambda_a2_combined

    combined_2 = train + test

    a1_combined_2 = [d['a1_T'] for d in combined_2]
    a2_combined_2 = [d['a2_T'] for d in combined_2]

    mean_a1_combined = np.mean(a1_combined_2)
    std_a1_combined = np.std(a1_combined_2)
    mean_a2_combined = np.mean(a2_combined_2)
    std_a2_combined = np.std(a2_combined_2)
  

    for i in train:
        i['aspect_1'] = (i['a1_T'] - mean_a1_combined) / std_a1_combined
        i['aspect_2'] = (i['a2_T'] - mean_a2_combined) / std_a2_combined
   
    for i in test:
        i['aspect_1'] = (i['a1_T'] - mean_a1_combined) / std_a1_combined
        i['aspect_2'] = (i['a2_T'] - mean_a2_combined) / std_a2_combined

    all_data = train + test
     
    return all_data, train, test