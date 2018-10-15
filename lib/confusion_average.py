import pandas as pd 
import numpy as np 

def confusion_matrix(csv, true_values, predicted_values, positive_class, negative_class):

    df = pd.read_csv(csv)
    pos = positive_class
    neg = negative_class
    
    
    df['true_pos'] = np.where(df['test_class'] == pos, np.where(df['result_by_mag'] == pos, 1, 0),0)
    df['true_neg'] = np.where(df['test_class'] == neg, np.where(df['result_by_mag'] == neg, 1, 0),0)
    df['false_pos'] = np.where(df['test_class'] == neg, np.where(df['result_by_mag'] == pos, 1, 0),0)
    df['false_neg'] = np.where(df['test_class'] == pos, np.where(df['result_by_mag'] == neg, 1, 0),0)

    df_sum = df.groupby('k')['true_pos', 'true_neg', 'false_pos', 'false_neg'].sum()

    df_sum['tp_rate'] = df_sum['true_pos'] / (df_sum['true_pos'] +  df_sum['false_neg'])
    df_sum['fp_rate'] = df_sum['false_pos'] / (df_sum['false_pos'] +  df_sum['true_neg'])

    return df_sum
