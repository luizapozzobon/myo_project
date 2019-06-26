import pandas as pd
import numpy as np
from sklearn import preprocessing
import sys
import os
import glob
def calc_mean_and_variance(df):
    scaler = preprocessing.MinMaxScaler(feature_range=(-127, 127))
    #print('loc', df.iloc[:, 2:10])
    df2 = pd.DataFrame(columns=['Sensor Mean','Sensor Variance', 'Gyro Mean', 'Gyro Variance','Orientation Mean', 'Orientation Variance'])
    df.iloc[:]        = pd.DataFrame(scaler.fit_transform(df.iloc[:]), columns=df.columns)
    df2['Sensor Mean']       = df[['Sensor 0','Sensor 1','Sensor 2','Sensor 3','Sensor 4','Sensor 5','Sensor 6','Sensor 7']].abs().mean(axis=1) 
    df2['Sensor Variance']   = df[['Sensor 0','Sensor 1','Sensor 2','Sensor 3','Sensor 4','Sensor 5','Sensor 6','Sensor 7']].abs().var(axis=1)
    df2['Gyro Mean']       = df[['Gyro 0','Gyro 1','Gyro 2',]].abs().mean(axis=1)
    df2['Gyro Variance']   = df[['Gyro 0','Gyro 1','Gyro 2',]].var(axis=1)
    df2['Orientation Mean']       = df[['Orientation 0','Gyro 1','Gyro 2',]].abs().mean(axis=1)
    df2['Orientation Variance']   = df[['Orientation 0','Gyro 1','Gyro 2',]].abs().var(axis=1)
    return df.join(df2)

def calc_mean_and_variance_column(df):
    data = pd.DataFrame(columns=[
        'Gyro 0 Mean', 'Gyro 1 Mean', 'Gyro 2 Mean',
        'Orientation 0 Mean', 'Orientation 1 Mean', 'Orientation 2 Mean', 'Orientation 3 Mean',
        'Sensor 0 Mean','Sensor 1 Mean', 'Sensor 2 Mean', 'Sensor 3 Mean', 'Sensor 4 Mean',
        'Sensor 5 Mean','Sensor 6 Mean', 'Sensor 7 Mean',

        'Gyro 0 Variance', 'Gyro 1 Variance', 'Gyro 2 Variance',
        'Orientation 0 Variance', 'Orientation 1 Variance', 'Orientation 2 Variance', 'Orientation 3 Variance',
        'Sensor 0 Variance','Sensor 1 Variance', 'Sensor 2 Variance', 'Sensor 3 Variance', 'Sensor 4 Variance',
        'Sensor 5 Variance','Sensor 6 Variance', 'Sensor 7 Variance',

        'ZeroCrossing', 'signal_change',

        'Label'])
    # print(df.columns)
    values = df.get_values()

    means_raw = values.mean(axis=0)
    means = pd.DataFrame(columns=[
        'Gyro 0 Mean', 'Gyro 1 Mean', 'Gyro 2 Mean',
        'Orientation 0 Mean', 'Orientation 1 Mean', 'Orientation 2 Mean', 'Orientation 3 Mean',
        'Sensor 0 Mean','Sensor 1 Mean', 'Sensor 2 Mean', 'Sensor 3 Mean', 'Sensor 4 Mean',
        'Sensor 5 Mean','Sensor 6 Mean', 'Sensor 7 Mean',
        ])
    means.loc[0] = means_raw

    var_raw = values.var(axis=0)
    variances = pd.DataFrame(columns=[
        'Gyro 0 Variance', 'Gyro 1 Variance', 'Gyro 2 Variance',
        'Orientation 0 Variance', 'Orientation 1 Variance', 'Orientation 2 Variance', 'Orientation 3 Variance',
        'Sensor 0 Variance','Sensor 1 Variance', 'Sensor 2 Variance', 'Sensor 3 Variance', 'Sensor 4 Variance',
        'Sensor 5 Variance','Sensor 6 Variance', 'Sensor 7 Variance',
        ])
    variances.loc[0] = var_raw

    # print(means)
    # print(variances)
    # print(variances.join(means))
    return(means.join(variances))

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
    return signal_changed

from pathlib import Path




if len(sys.argv) >= 3 :
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pasta = current_dir +"/"+ sys.argv[1]
    pasta_treated = pasta + "treated"
    print("Pasta ", pasta)
    files = glob.glob(pasta+"/*.csv")


    Label = sys.argv[2]


    #verifica se arquivo existe. se nçaoe xiste cria um df novo, senao abre o csv
    comp = Path(current_dir +'/datasets/oficial/compilado.csv')
    if comp.is_file():
        compilado = pd.read_csv(current_dir +'/datasets/oficial/compilado.csv')
    else:
        compilado = pd.DataFrame(columns=['Gyro 0', 'Gyro 1', 'Gyro 2',
           'Orientation 0', 'Orientation 1', 'Orientation 2', 'Orientation 3',
           'Sensor 0', 'Sensor 1', 'Sensor 2', 'Sensor 3', 'Sensor 4',
           'Sensor 5','Sensor 6', 'Sensor 7', 'Sensor Mean',
           'Sensor Variance', 'Gyro Mean', 'Gyro Variance','Orientation Mean',
           'Orientation Variance'])



    #verifica se arquivo existe. se nçaoe xiste cria um df novo, senao abre o csv
    feat = Path(current_dir +'/datasets/oficial/features.csv')
    if feat.is_file():
        features = pd.read_csv(current_dir +'/datasets/oficial/features.csv')
    else:
        features = pd.DataFrame(columns=[
            'Gyro 0 Variance', 'Gyro 1 Variance', 'Gyro 2 Variance',
            'Orientation 0 Variance', 'Orientation 1 Variance', 'Orientation 2 Variance', 'Orientation 3 Variance',
            'Sensor 0 Variance','Sensor 1 Variance', 'Sensor 2 Variance', 'Sensor 3 Variance', 'Sensor 4 Variance',
            'Sensor 5 Variance','Sensor 6 Variance', 'Sensor 7 Variance',
           
            'Gyro 0 Mean', 'Gyro 1 Mean', 'Gyro 2 Mean',
            'Orientation 0 Mean', 'Orientation 1 Mean', 'Orientation 2 Mean', 'Orientation 3 Mean',
            'Sensor 0 Mean','Sensor 1 Mean', 'Sensor 2 Mean', 'Sensor 3 Mean', 'Sensor 4 Mean',
            'Sensor 5 Mean','Sensor 6 Mean', 'Sensor 7 Mean'
            ])


    # itera sobre todos os csv da pasta indicada
    for f in files:
        print(f)
        filename = f.replace(pasta,'')
        dados = pd.read_csv(f).drop(columns=['Unnamed: 0','Timestamp','Label'])
        dados_novos = calc_mean_and_variance(dados)
        dados_novos['Label'] = Label

        features_temp = calc_mean_and_variance_column(dados)
        features_temp['Label'] = Label
        features = features.append(features_temp,ignore_index=True)
        new_name = filename.replace('.csv','novo.csv')
        # print(new_name)
        if not os.path.exists(pasta_treated):
            os.mkdir(pasta_treated)
        dados_novos.to_csv(pasta_treated +'/'+ new_name,index = False)
        compilado = compilado.append(dados_novos,ignore_index=True)


    compilado.to_csv(current_dir + '/datasets/oficial/compilado.csv',index = False)
    features.to_csv(current_dir +'/datasets/oficial/features.csv',index = False)

else:
    print("missing argument")
