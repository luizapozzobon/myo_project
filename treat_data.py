import pandas as pd
import numpy as np
from sklearn import preprocessing
import sys
import os
import glob


def calc_mean_and_variance(df):
    # Cálculo para as linhas
    # Não usado
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

def filter_data(data, low_band=5, mid_band=58, high_band=72, sfreq=200):
    # TODO testar para dataset inteiro -> APENAS SENSORES
    """
    time: Time data
    emg: EMG data
    high: high-pass cut off frequency
    low: low-pass cut off frequency
    sfreq: sampling frequency
    """
    # create highpass filter for EMG for first 5hz
    low_band = low_band/(sfreq)
    b1, a1 = sp.signal.butter(4, low_band, btype='highpass')
    # process EMG signal: filter EMG
    emg_filtered = sp.signal.filtfilt(b1, a1, emg)

    # normalise cut-off frequencies to sampling frequency
    high_band = high_band/(sfreq)
    mid_band = mid_band/(sfreq)
    # create second bandpass filter for EMG
    b1, a1 = sp.signal.butter(4, [mid_band, high_band], btype='bandstop')
    emg_filtered2 = sp.signal.filtfilt(b1, a1, emg_filtered)

    return emg_filtered2

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

def wilson_amplitude(df, th):
    # TODO fazer para todos os dados
    sensor = []
    inds = 0
    ind = 0
    df2 = df.copy()

    s0 = np.array(df['Sensor 0'])
    s1 = np.array(df['Sensor 1'])
    s2 = np.array(df['Sensor 2'])
    s3 = np.array(df['Sensor 3'])
    s4 = np.array(df['Sensor 4'])
    s5 = np.array(df['Sensor 5'])
    s6 = np.array(df['Sensor 6'])
    s7 = np.array(df['Sensor 7'])

    ind_th0 = (s0 < th) & (-th < s0)
    ind_th1 = (s1 < th) & (-th < s1)
    ind_th2 = (s2 < th) & (-th < s2)
    ind_th3 = (s3 < th) & (-th < s3)
    ind_th4 = (s4 < th) & (-th < s4)
    ind_th5 = (s5 < th) & (-th < s5)
    ind_th6 = (s6 < th) & (-th < s6)
    ind_th7 = (s7 < th) & (-th < s7)
    s0[ind_th0] = 0
    s1[ind_th1] = 0
    s2[ind_th2] = 0
    s3[ind_th3] = 0
    s4[ind_th4] = 0
    s5[ind_th5] = 0
    s6[ind_th6] = 0
    s7[ind_th7] = 0

    df2['Sensor 0'] = s0
    df2['Sensor 1'] = s1
    df2['Sensor 2'] = s2
    df2['Sensor 3'] = s3
    df2['Sensor 4'] = s4
    df2['Sensor 5'] = s5
    df2['Sensor 6'] = s6
    df2['Sensor 7'] = s7

    return df2


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
        # IDEIA -> fazer com e sem amplitude de wilson pra comparar

        print(f)
        filename = f.replace(pasta,'')
        dados = pd.read_csv(f).drop(columns=['Unnamed: 0','Timestamp','Label'])

        #----------- TODO testar isso
        # Filtrar, normalizar, zero crossings, signal change, calcular features e tocar tudo pra um csv
        filtered = filter_data(dados)

        #TODO falta normalizar
        normalized = filtered

        #TODO arrumar zero crossings pro formato que o prof disse na aula?
        zero_cr = zero_crossings(normalized)
        signal_ch = signal_change(normalized)
        features = features.append(zero_cr, ignore_index=True)
        features = features.append(signal_ch, ignore_index=True)

        features_temp = calc_mean_and_variance_column(normalized)
        features_temp['Label'] = Label
        features = features.append(features_temp, ignore_index=True)

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
