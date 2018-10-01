import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

#simple k-Nearest Neighbor for two dimensions.
#predict_set is a dataframe of test elements.
#predict_class is an optional column of the known classification
    #of the predict_set. Could be used to test the accuracy of the classifier.
#b1 is a column and the first element of the pair in predict_set. b2 is the second element.
#actual_set is a dataframe of known elements and classes.
#actual_class is the known class of each element pair.
#a1 is a column and the first element of the pair in actual_set. a2 is the second element.
#k is the number of nearest neighbors to vote on or identify the class of the unknown.

def kNN(k, actual_set, actual_class, a1, a2, predict_set, b1, b2,  predict_class=None):
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

#standardize new points.

    b1_mean = np.mean(predict_set['b1_t'])
    b1_std = np.std(predict_set['b1_t'])
    predict_set['b1_z'] = (predict_set['b1_t'] - b1_mean) / b1_std

    b2_mean = np.mean(predict_set['b2_t'])
    b2_std = np.std(predict_set['b2_t'])
    predict_set['b2_z'] = (predict_set['b2_t'] - b2_mean) / b2_std

#calculate all distances for each new point to known points

    for i in predict_set.index:
        
    #reposition the predict_set element pairs as the origin.  a1_z_i and a2_z_i are new columns
        #where each predict_set element pair set at the origin.
        actual_set['a1_z'+'_'+str(i)] = actual_set['a1_z'] - predict_set['b1_z'][i]
        actual_set['a2_z'+'_'+str(i)] = actual_set['a2_z'] - predict_set['b2_z'][i]

    #square root of dot product for calcuate the magnitude of all vectors
        actual_set['magnitude'+'_'+str(i)] = np.sqrt(actual_set['a1_z'+'_'+str(i)]**2 + actual_set['a2_z'+'_'+str(i)]**2)

#find the k smallest vector magnitudes
        smallest = actual_set.nsmallest(k, 'magnitude'+'_'+str(i))[actual_class]
        smallest.reset_index(drop = True, inplace = True)
        closest = closest.append(smallest, ignore_index = True)

#collected the predictions        
    closest = closest.transpose()
    vote = closest.apply(pd.value_counts) / k
    vote = vote.transpose().fillna(0)
    class_1 = vote.columns[0]
    class_2 = vote.columns[1]
    vote['assigned_class'] = np.where(vote.iloc[:,0] > vote.iloc[:,1], class_1, class_2)

#count element pairs in class    
    count_class =  vote['assigned_class'].groupby(vote['assigned_class']).count()

#if the predict_set does not have known classes, return a scatter plot comparing the actual and prediction sets and a count of classes
#if the predict_set does have known classes, return as above plus confusion matrix

    if (predict_class == None):
    #plot predicted and actual points 
        plt.scatter(predict_set[predict_set[predict_class]==class_1].b1_z, \
                    predict_set[predict_set[predict_class]==class_1].b2_z, color='r',label='Predicted '+class_1)
        plt.scatter(predict_set[predict_set[predict_class]==class_2].b1_z, \
                    predict_set[predict_set[predict_class]==class_2].b2_z, color='y',label='Predicted '+class_2)
        plt.scatter(actual_set[actual_set[actual_class]==class_1].a1_z, \
                    actual_set[actual_set[actual_class]==class_1].a2_z, color='g',label='Actual '+class_1)
        plt.scatter(actual_set[actual_set[actual_class]==class_2].a1_z, \
                    actual_set[actual_set[actual_class]==class_2].a2_z, color='b',label='Actual '+class_2)
        
        plt.title('Scatter Plot of Actuals and Predicted Class\nby '+str(a1)+' and '+str(a2)+'\nstandardized elements')
        plt.xlabel(a1)
        plt.ylabel(a2)
        plt.legend()
        plt.grid()

        print('Class: {}, Number: {}\nClass: {}, Number: {}'.format(count_class.index[0], count_class[0], count_class.index[1], count_class[1])) 

    else:
    #plot predicted and actual points
        plt.scatter(predict_set[predict_set[predict_class]==class_1].b1_z, \
                    predict_set[predict_set[predict_class]==class_1].b2_z, color='r',label='Predicted '+class_1)
        plt.scatter(predict_set[predict_set[predict_class]==class_2].b1_z, \
                    predict_set[predict_set[predict_class]==class_2].b2_z, color='y',label='Predicted '+class_2)
        plt.scatter(actual_set[actual_set[actual_class]==class_1].a1_z, \
                    actual_set[actual_set[actual_class]==class_1].a2_z, color='g',label='Actual '+class_1)
        plt.scatter(actual_set[actual_set[actual_class]==class_2].a1_z, \
                    actual_set[actual_set[actual_class]==class_2].a2_z, color='b',label='Actual '+class_2)
       
        plt.title('Scatter Plot of Actuals and Predicted Class\nby '+str(a1)+' and '+str(a2)+'\nstandardized elements')
        plt.xlabel(a1)
        plt.ylabel(a2)
        plt.legend()
        plt.grid()

        frames = (pd.DataFrame(vote['assigned_class']),pd.DataFrame(predict_set[predict_class]))
        alignment = pd.concat(frames, axis=1)
        alignment['true_pos'] = np.where(alignment['assigned_class'] == class_1, np.where(alignment[predict_class] == class_1, 1, 0),0)
        alignment['false_pos'] = np.where(alignment['assigned_class'] == class_2, np.where(alignment[predict_class] == class_1, 1, 0),0)
        alignment['true_neg'] = np.where(alignment['assigned_class'] == class_2, np.where(alignment[predict_class] == class_2, 1, 0),0)
        alignment['false_neg'] = np.where(alignment['assigned_class'] == class_1, np.where(alignment[predict_class] == class_2, 1, 0),0)

        print('Class: {}, Number: {}\nClass: {}, Number: {}'.format(count_class.index[0], count_class[0], count_class.index[1], count_class[1]))
        print('True Positives: {}\nFalse Positives: {}\nTrue Negatives; {}\nFalse Negatives: {}'.format(alignment['true_pos'].sum(), \
                                                                                                        alignment['false_pos'].sum(), \
                                                                                                        alignment['true_neg'].sum(), \
                                                                                                        alignment['false_neg'].sum()))
        

                
    plt.show()

    
                                                                                                                                                                  
