import pandas as pd 
import numpy as np 

def confusion_matrix(csv, true_values, positive_class, negative_class):

    df = pd.read_csv(csv)
    pos = positive_class
    neg = negative_class

    df['vote_pos'] = np.where(df[true_values] == pos, 1, 0)
    df['vote_neg'] = np.where(df[true_values] == neg, 1, 0)

    df['vote_pos_cum'] = df.groupby('id')['vote_pos'].cumsum()
    df['vote_neg_cum'] = df.groupby('id')['vote_neg'].cumsum()

    df['result_by_vote'] = np.where(df['vote_pos_cum'] >= df['vote_neg_cum'], np.where(df['vote_pos_cum'] == df['vote_neg_cum'], 'tie', pos), neg)
    
    
    df['true_pos'] = np.where(df[true_values] == pos, np.where(df['result_by_vote'] == pos, 1, 0),0)
    df['true_neg'] = np.where(df[true_values] == neg, np.where(df['result_by_vote'] == neg, 1, 0),0)
    df['false_pos'] = np.where(df[true_values] == neg, np.where(df['result_by_vote'] == pos, 1, 0),0)
    df['false_neg'] = np.where(df[true_values] == pos, np.where(df['result_by_vote'] == neg, 1, 0),0)
    df['ties'] = np.where(df['result_by_vote'] == 'tie', 1, 0)

    df_sum = df.groupby('k')['true_pos', 'true_neg', 'false_pos', 'false_neg', 'ties'].sum()

    df_sum['tp_rate'] = df_sum['true_pos'] / (df_sum['true_pos'] +  df_sum['false_neg'])
    df_sum['fp_rate'] = df_sum['false_pos'] / (df_sum['false_pos'] +  df_sum['true_neg'])

    return df_sum
