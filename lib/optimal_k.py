import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

#using kNN in two dimensions, calculate true positive and false positive rates for all k 1 to len(predict_set).
#produce a ROC plot and determine optimal k where the change in the true postive rate starts to decrease
    #and the change in the false positive rate starts to increase.
#predict_set is a dataframe of test elements.
#predict_class is the known classification of the predict_set.
#b1 is a column and the first element of the pair in predict_set. b2 is the second element.
#actual_set is a dataframe of known elements and classes.
#actual_class is the known class of each element pair.
#a1 is a column and the first element of the pair in actual_set. a2 is the second element.


def optimal_k(actual_set, actual_class, a1, a2, predict_set, b1, b2,  predict_class):

#empty dataframe for appending closest points.
    closest = pd.DataFrame()

#reset the basis, fit the data sets into a normal distribution.
#added a small scalar to the datasets to keep results greater than 0.    
    actual_set['a1_t'] = stats.boxcox(actual_set[a1]+1)[0]
    actual_set['a2_t'] = stats.boxcox(actual_set[a2]+1)[0]
    predict_set['b1_t'] = stats.boxcox(predict_set[b1]+1)[0]
    predict_set['b2_t'] = stats.boxcox(predict_set[b2]+1)[0]

#standardize data points to even the scales, a1_z and a2_z will be the columns of standardized values.
#this may not be technically necessary, but it is aesthetically pleasing.
#this also assumes that a1 and a2 are normally distributed.
    a1_mean = np.mean(actual_set['a1_t'])
    a1_std = np.std(actual_set['a1_t'])
    actual_set['a1_z'] = (actual_set['a1_t'] - a1_mean) / a1_std

    a2_mean = np.mean(actual_set['a2_t'])
    a2_std = np.std(actual_set['a2_t'])
    actual_set['a2_z'] = (actual_set['a2_t'] - a2_mean) / a2_std

#standardize predict points.

    b1_mean = np.mean(predict_set['b1_t'])
    b1_std = np.std(predict_set['b1_t'])
    predict_set['b1_z'] = (predict_set['b1_t'] - b1_mean) / b1_std

    b2_mean = np.mean(predict_set['b2_t'])
    b2_std = np.std(predict_set['b2_t'])
    predict_set['b2_z'] = (predict_set['b2_t'] - b2_mean) / b2_std
#calculate all distances for each new point to known points

    for i in predict_set.index:
        #k = int(i) +1
        
    #reposition the predict_set element pairs as the origin.  a1_z_i and a2_z_i are new columns
        #where each predict_set element pair set at the origin.
        actual_set['a1_z'+'_'+str(i)] = actual_set['a1_z'] - predict_set['b1_z'][i]
        actual_set['a2_z'+'_'+str(i)] = actual_set['a2_z'] - predict_set['b2_z'][i]

    #square root of dot product for calcuate the magnitude of all vectors
        actual_set['magnitude'+'_'+str(i)] = np.sqrt(actual_set['a1_z'+'_'+str(i)]**2 + actual_set['a2_z'+'_'+str(i)]**2)
     

#find the k smallest vector magnitudes
        smallest = actual_set.nsmallest(5, 'magnitude'+'_'+str(i))[actual_class]
        smallest.reset_index(drop = True, inplace = True)
        closest = closest.append(smallest, ignore_index = True)
        
#interate k from 1 to the length of the data set.

    #collect the closest and votes for each k
        closest = closest.transpose()
        vote = closest.apply(pd.value_counts)
        #vote = vote.transpose().fillna(0)
            #class_1 = vote.columns[0]
            #class_2 = vote.columns[1]
            #vote['assigned_class'+'_'+str(k)] = np.where(vote.iloc[:,0] > vote.iloc[:,1], class_1, class_2)

    #count element pairs in class    
            #count_class =  vote['assigned_class'+'_'+str(k)].groupby(vote['assigned_class'+'_'+str(k)]).count()

            #frames = (pd.DataFrame(vote['assigned_class'+'_'+str(k)]),pd.DataFrame(predict_set[predict_class]))
            #alignment = pd.concat(frames, axis=1)
            #alignment['true_pos'+'_'+str(k)] = np.where(alignment['assigned_class'+'_'+str(k)] == class_1, np.where(alignment[predict_class] == class_1, 1, 0),0)
            #alignment['false_pos'+'_'+str(k)] = np.where(alignment['assigned_class'+'_'+str(k)] == class_2, np.where(alignment[predict_class] == class_1, 1, 0),0)
            #alignment['true_neg'+'_'+str(k)] = np.where(alignment['assigned_class'+'_'+str(k)] == class_2, np.where(alignment[predict_class] == class_2, 1, 0),0)
            #alignment['false_neg'+'_'+str(k)] = np.where(alignment['assigned_class'+'_'+str(k)] == class_1, np.where(alignment[predict_class] == class_2, 1, 0),0)


        print(vote)
