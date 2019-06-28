import pandas as pd
import numpy as np
from sklearn import preprocessing
import sys
import os
import glob
import scipy as sp
from scipy import signal

def calc_mean_and_variance(df):
    # Cálculo para as linhas
    # Não usado
    scaler = preprocessing.MinMaxScaler(feature_range=(-127, 127))
    #print('loc', df.iloc[:, 2:10])
    df1 = pd.DataFrame(columns=['Sensor Mean','Sensor Variance', 'Gyro Mean', 'Gyro Variance','Orientation Mean', 'Orientation Variance'])
    df.iloc[:]        = pd.DataFrame(scaler.fit_transform(df.iloc[:]), columns=df.columns)
    df2['Sensor Mean']       = df[['Sensor 0','Sensor 1','Sensor 2','Sensor 3','Sensor 4','Sensor 5','Sensor 6','Sensor 7']].abs().mean(axis=1)
    df2['Sensor Variance']   = df[['Sensor 0','Sensor 1','Sensor 2','Sensor 3','Sensor 4','Sensor 5','Sensor 6','Sensor 7']].abs().var(axis=1)
    df2['Gyro Mean']       = df[['Gyro 0','Gyro 1','Gyro 2',]].abs().mean(axis=1)
    df2['Gyro Variance']   = df[['Gyro 0','Gyro 1','Gyro 2',]].var(axis=1)
    df2['Orientation Mean']       = df[['Orientation 0','Gyro 1','Gyro 2',]].abs().mean(axis=1)
    df2['Orientation Variance']   = df[['Orientation 0','Gyro 1','Gyro 2',]].abs().var(axis=1)
    return df.join(df2)

def filter_data(emg, low_band=5, mid_band=58, high_band=72, sfreq=200):
    # TODO testar para dataset inteiro -> APENAS SENSORES
    """
    emg: EMG data
    high: high-pass cut off frequency
    low: low-pass cut off frequency
    sfreq: sampling frequency
    """
    for column in emg:
        # create highpass filter for EMG for first 5hz
        low_band = low_band/(sfreq/2)
        #b1, a1 = sp.signal.butter(2, low_band, btype='highpass')
        #emg_filtered = sp.signal.filtfilt(b1, a1, emg[column].values)
        sos = sp.signal.butter(2, low_band, btype='highpass', output='sos')
        emg_filtered = sp.signal.sosfilt(sos, emg[column].values)

        # normalise cut-off frequencies to sampling frequency
        high_band = high_band/(sfreq/2)
        mid_band = mid_band/(sfreq/2)
        # create second bandpass filter for EMG
        #b1, a1 = sp.signal.butter(2, [mid_band, high_band], btype='bandstop')
        #emg_filtered2 = sp.signal.filtfilt(b1, a1, emg_filtered)
        sos = sp.signal.butter(2, [mid_band, high_band], btype='bandstop', output='sos')
        emg_filtered2 = sp.signal.sosfilt(sos, emg_filtered)
        emg[column] = emg_filtered2
    return emg

def norm_data(df):
    scaler = preprocessing.MinMaxScaler(feature_range=(-127, 127))
    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return df


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

def zero_crossings(df, columns):
    zcr = []
    for i in range(8):
        zcr.append(np.where(np.diff(np.sign(df[columns[i]].values)))[0])
    new_zcr = []
    for i in range(8):
        zcr_sensor = []
        for j in range(len(zcr[i])-1):
            zcr_sensor.append(zcr[i][j+1] - zcr[i][j])
        new_zcr.append(np.mean(zcr_sensor))
    return new_zcr

def signal_change(df):
    sc = []
    for i in range(0, 8):
        signal_changed = []
        for j in range(1, 500-1):
            column = df.loc[:, 'Sensor '+str(i)]

            center_value = column[j]
            lower_value = column[j-1]
            upper_value = column[j+1]

            sig_center = np.sign(center_value)
            sig_lower = np.sign(lower_value)
            sig_upper = np.sign(upper_value)

            #print(sig_lower, sig_center, sig_upper)
            #print(j, lower_value, center_value, upper_value)

            if sig_center == sig_lower and sig_center == sig_upper:
                if (np.abs(center_value) > np.abs(lower_value)) and (np.abs(center_value) > np.abs(upper_value)):
                    signal_changed.append(j)
            elif sig_upper == sig_lower and sig_lower != sig_center:
                signal_changed.append(j)
            elif sig_center == sig_lower and sig_center != sig_upper:
                if np.abs(center_value) >= np.abs(lower_value):
                    signal_changed.append(j)
            elif sig_center != sig_lower and sig_lower == sig_upper:
                if np.abs(center_value) >= np.abs(upper_value):
                    signal_changed.append(j)
        sc.append(signal_changed)
    new_signal_changed = []
    for i in range(8):
        sensor_sc = []
        for j in range(len(sc[i]) - 1):
            sensor_sc.append(sc[i][j+1] - sc[i][j])
        new_signal_changed.append(np.mean(sensor_sc))
    return new_signal_changed

def wilson_amplitude(emg, th):
    # TODO fazer para todos os dados
    sensor = []
    inds = 0
    ind = 0
    df2 = emg.copy()

    for column in emg:
        s0 = np.array(emg['Sensor 0'])
        s1 = np.array(emg['Sensor 1'])
        s2 = np.array(emg['Sensor 2'])
        s3 = np.array(emg['Sensor 3'])
        s4 = np.array(emg['Sensor 4'])
        s5 = np.array(emg['Sensor 5'])
        s6 = np.array(emg['Sensor 6'])
        s7 = np.array(emg['Sensor 7'])

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
            'Sensor 5 Mean','Sensor 6 Mean', 'Sensor 7 Mean', 'zcr 0', 'zcr 1', 'zcr 2', 'zcr 3', 'zcr 4',
            'zcr 5', 'zcr 6', 'zcr 7', 'sch 0', 'sch 1', 'sch 2', 'sch 3', 'sch 4', 'sch 5', 'sch 6', 'sch 7'
            ])

    # itera sobre todos os csv da pasta indicada
    for f in files:
        # IDEIA -> fazer com e sem amplitude de wilson pra comparar

        print(f)
        filename = f.replace(pasta,'')
        dados = pd.read_csv(f).drop(columns=['Unnamed: 0','Timestamp','Label'])

        print('Descrição dos dados: ')
        print(dados.describe())
        #----------- TODO testar isso
        # Filtrar, normalizar, zero crossings, signal change, calcular features e tocar tudo pra um csv
        emg_columns = ['Sensor 0', 'Sensor 1','Sensor 2', 'Sensor 3', 'Sensor 4', 'Sensor 5', 'Sensor 6', 'Sensor 7']
        dados[emg_columns] = filter_data(dados[emg_columns])
        #dados[emg_columns] = wilson_amplitude(dados[emg_columns], 7)
        print(dados.head())

        normalized = norm_data(dados)
        print(normalized.head(10))

        #TODO arrumar zero crossings pro formato que o prof disse na aula?
        zero_cr = zero_crossings(normalized, emg_columns)
        signal_ch = signal_change(normalized)
        print('zero: ', zero_cr)
        print('signal_changed: ', signal_ch)
        df_zero = pd.DataFrame([zero_cr], columns=['zcr '+str(i) for i in range(8)])
        df_sch = pd.DataFrame([signal_ch], columns=['sch '+str(i) for i in range(8)])
        df_zero = df_zero.join(df_sch)

        features_temp = calc_mean_and_variance_column(normalized)
        features_temp['Label'] = Label
        features_temp = features_temp.join(df_zero)
        features = features.append(features_temp, ignore_index=True)

        new_name = filename.replace('.csv','novo.csv')
        # print(new_name)
        if not os.path.exists(pasta_treated):
            os.mkdir(pasta_treated)
        normalized.to_csv(pasta_treated +'/'+ new_name,index = False)
        compilado = compilado.append(normalized, ignore_index=True)

    compilado.to_csv(current_dir + '/datasets/oficial/compilado.csv',index = False)
    features.to_csv(current_dir +'/datasets/oficial/features.csv', index = False, decimal=".")

else:
    print("missing argument")
