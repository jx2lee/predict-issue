import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler


def import_data(csv_path, convert_type, index_col):

    # import data
    df = pd.read_csv(csv_path, encoding = 'utf-8')
    # del row Closed Date == 1970/01/01 09:00:00
    df = df[df['Closed Date'] != '1970/01/01 09:00:00']
    df = df.reset_index(drop=True) # reindex
    # Registered data & Closed Date convert type (char -> datetime obj), List
    for col in convert_type:
        df[col] = pd.to_datetime(df[col], format="%Y/%m/%d")
    # convert index to datetime object 
    df.index = df[index_col]
    del df[index_col]
    # resample df
    df = df.resample('d').count()
    df = df[df['Product'] > 0]
    df = df.drop(['Product', 'Module', 'Category', 'Priority', 'Severity', 'Customer', 'Project', 'Owner', 'Version'], axis = 1)
    df = df.rename(columns={'Closed Date' : 'cnt'})
    df['cnt'] = df['cnt'].astype(int) #convert type
    return df


def split_data(df, n_train_time, period_size):

    # train / test set
    val = df.values.reshape(-1)
    tr = val[:n_train_time]
    te = val[n_train_time:-(period_size)]
    num_train = len(tr)
    num_test = len(te)

    tr_X, tr_y = [], []
    for i in range(0, len(tr)-period_size):
        tr_X.append(tr[i:i+period_size])
        tr_y.append(tr[i+period_size])

    te_X, te_y = [], []
    for i in range(0, len(te)-period_size):
        te_X.append(te[i:i+period_size])
        te_y.append(te[i+period_size])

    tr_X = np.asarray(tr_X).reshape(len(tr_X), 1, period_size)
    te_X = np.asarray(te_X).reshape(len(te_X), 1, period_size)
    tr_y = np.asarray(tr_y)
    te_y = np.asarray(te_y)

    return tr_X, tr_y, te_X, te_y, num_train, num_test


def batch_generator(X, y, num_train, batch_size, period_size, output_size):
    
    while True:
        X_shape = (batch_size, period_size)
        X_batch = np.zeros(shape=X_shape, dtype = np.float)
        y_shape = (batch_size, output_size)
        y_batch = np.zeros(shape=y_shape, dtype = np.float)

        for i in range(batch_size):
            idx = np.random.randint(num_train - batch_size)
            X_batch[i] = X[idx:idx+1]
            y_batch[i] = y[idx:idx+1]

        yield (X_batch, y_batch)        

