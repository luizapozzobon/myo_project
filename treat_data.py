import pandas as pd
import numpy as np
from sklearn import preprocessing

def calc_mean_and_variance(df):
    scaler = preprocessing.MinMaxScaler(feature_range=(-127, 127))
    #print('loc', df.iloc[:, 2:10])
    df.iloc[:, 2:10]        = pd.DataFrame(scaler.fit_transform(df.iloc[:, 2:10]), columns=df.columns[2:10])
    df['Sensor Mean']       = df.iloc[:, 2:10].abs().mean(axis=1)
    df['Sensor Variance']   = df.iloc[:, 2:10].var(axis=1)
    df['Gyro Mean']       = df.iloc[:, 10:13].abs().mean(axis=1)
    df['Gyro Variance']   = df.iloc[:, 10:13].var(axis=1)
    df['Orientation Mean']       = df.iloc[:, 13:18].abs().mean(axis=1)
    df['Orientation Variance']   = df.iloc[:, 13:18].var(axis=1)
    return df

def zero_crossings(df):
    zcr = []
    for i in range(2, 10):
        column = df.iloc[:, i]
        zcr.append(np.where(np.diff(np.sign(column))))
    return zcr

dados = pd.read_csv('./datasets/myo-parado-david3-2019-05-09 10:32:57-.csv')
print(dados.head(10))
dados_novo = calc_mean_and_variance(dados)
zero = zero_crossings(dados_novo)
for i in range(8):
    print(zero[i])
