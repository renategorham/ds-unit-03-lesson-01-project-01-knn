import pandas as pd
import numpy as np

#separate a proportional training set
#df is a pandas dataframe
#y is the column name of the response in the dataframe (df)
#size is the proportion reserved to the training set (default = 0.2)

def train_test(df,y,size=0.2):
    
    #calculate proportion of y for two responses
    #first, identify the two responses
    y_1 = df[y].unique()[0]
    y_2 = df[y].unique()[1]

    #count each response
    ny_1 = df[df[y]==y_1].count()[0]
    ny_2 = df[df[y]==y_2].count()[0]
    ny_total = ny_1 + ny_2

    #calculate propostion of each response
    p_1 = ny_1 / df[y].count()
    p_2 = 1 - p_1

    #shuffle the data set, incase it is ordered (3x)
    df = df.sample(frac=1).reset_index(drop=True)
    df = df.sample(frac=1).reset_index(drop=True)
    df = df.sample(frac=1).reset_index(drop=True)

    #calcuate size of sets to return - use ceiling and abolute value to prevent fractions
    #full training and test set sizes
    train_size = np.ceil(ny_total * size).astype(int)
    test_size =  np.abs(ny_total - train_size).astype(int)

    #size of training set responses
    train_y_1 = np.ceil(train_size * p_1).astype(int)
    train_y_2 = np.abs(train_size - train_y_1).astype(int)
    
    #size of testing set responses
    #test_y_1 = ny_total - train_y_1 - ny_1
    #test_y_2 = ny_total - train_y_1 - train_y_2 - test_y_1

    #slice dataframe by set sizes
    df_train_y_1 = df[df[y]==y_1][:train_y_1]
    df_test_y_1 = df[df[y]==y_1][train_y_1:]
    df_train_y_2 = df[df[y]==y_2][:train_y_2]
    df_test_y_2 = df[df[y]==y_2][train_y_2:]

    #contact the training and test sets, then shuffle for no paticular reason
    df_train = pd.concat((df_train_y_1,df_train_y_2))
    df_test = pd.concat((df_test_y_1,df_test_y_2))

    df_train = df_train.sample(frac=1).reset_index(drop=True)
    df_test = df_test.sample(frac=1).reset_index(drop=True)

    return df_train, df_test
    
    
    
