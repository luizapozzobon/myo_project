import pandas as pd
import numpy as np
from sklearn import preprocessing
import sys
import os
import glob
def calc_mean_and_variance(df):
    scaler = preprocessing.MinMaxScaler(feature_range=(-127, 127))
    #print('loc', df.iloc[:, 2:10])
    df.iloc[:, 2:10]        = pd.DataFrame(scaler.fit_transform(df.iloc[:, 2:10]), columns=df.columns[2:10])
    df2 = pd.DataFrame(columns=['Sensor Mean','Sensor Variance', 'Gyro Mean', 'Gyro Variance','Orientation Mean', 'Orientation Variance'])
    df2['Sensor Mean']       = df.iloc[:, 9:17].abs().mean(axis=1)
    df2['Sensor Variance']   = df.iloc[:, 9:17].var(axis=1)
    df2['Gyro Mean']       = df.iloc[:, 1:4].abs().mean(axis=1)
    df2['Gyro Variance']   = df.iloc[:, 1:4].var(axis=1)
    df2['Orientation Mean']       = df.iloc[:, 5:9].abs().mean(axis=1)
    df2['Orientation Variance']   = df.iloc[:, 5:9].var(axis=1)
    return df, df2

def zero_crossings(df):
    zcr = []
    for i in range(2, 10):
        column = df.iloc[:, i]
        zcr.append(np.where(np.diff(np.sign(column))))
    return zcr

def signal_change(df):
    sc = []
    for i in range(0, 8):
        signal_changed = []
        changed_count = 0
        column = df.loc[:, 'Sensor '+str(i)]
        for j in range(1, 500-1):
            val1 = np.abs(column[j]) - np.abs(column[j-1])
            val2 = np.abs(column[j+1]) - np.abs(column[j])
            #print(column[j-1], column[j], column[j+1], np.sign(val1), np.sign(val2), np.sign(val1)+np.sign(val2))
            if np.sign(val1) + np.sign(val2) == 0 and (val1 != 0 or val2 != 0):
                signal_changed.append(j)
                changed_count += 1
        #print("Mudanças de sinal/Contador do Sensor {}: {} / {}".format(i, signal_changed, changed_count))
    return signal_changed

if len(sys.argv) >= 2 :
    pasta = os.path.dirname(os.path.abspath(__file__)) +"/"+ sys.argv[1]
    print("Pasta ", pasta)
    files = glob.glob(pasta+"/*.csv")
    #print(files)
    data = []
    for f in files:
        #print(f)
        dados = pd.read_csv(f)
        #print(dados.head(10))
        dados_novo, mean_var_dados = calc_mean_and_variance(dados)
        print(mean_var_dados.head())
        zero = zero_crossings(dados_novo)
        signal_change(dados_novo)
        #for i in range(8):
        #print(zero[i])
        data.append([zero,signal_change, mean_var_dados])
    print(data)
else:
    print("missing argument")
