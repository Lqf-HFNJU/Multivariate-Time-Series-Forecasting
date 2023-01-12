import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import FastICA


def trans():
    path = r'C:\Users\18933\Desktop\时间序列分析\hw5\final\Multivariate-Time-Series-Forecasting\code\dataset\national_illness.csv'
    df_raw = pd.read_csv(path)
    date = df_raw['date'].copy()
    del df_raw['date']
    n_dim = df_raw.shape[1]
    ica = FastICA(n_dim, whiten="arbitrary-variance")
    ica.fit(df_raw[:])
    df_trans = ica.transform(df_raw)
    pd_trans = pd.DataFrame(df_trans)
    pd_trans['date'] = date
    pd_trans.columns = ['WEIGHTED ILI', 'UNWEIGHTED ILI', 'AGE 0-4', 'AGE 5-24', 'ILITOTAL', 'NUM. OF PROVIDERS', 'OT',
                        'date']
    pd_trans.to_csv('./out.csv', index=False)


if __name__ == '__main__':
    # path = r'C:\Users\18933\Desktop\时间序列分析\hw5\final\Multivariate-Time-Series-Forecasting\code\dataset\electricity.csv'
    # df_raw = pd.read_csv(path)
    # features = df_raw.columns.to_list().remove('date')
    #
    # # series = TimeSeries.from_dataframe(df_raw, 'date', features)
    # # scaler = StandardScaler()
    # # transformer = Scaler(scaler)
    # # series_transformed = transformer.fit_transform(series)
    # #
    # # series_transformed.plot()
    # # plt.show()
    # print(df_raw.shape)
    #
    # n_dim = df_raw.shape[1] // 2
    # # n_dim = 7
    # ica = FastICA(n_dim, whiten="arbitrary-variance")
    # del df_raw['date']
    # S_ = ica.fit_transform(df_raw.to_numpy().reshape((1, -1, df_raw.shape[1])))  # Reconstruct signals
    # A_ = ica.mixing_  # Get estimated mixing matrix
    #
    # # plt.plot(S_)
    # # plt.show()
    # # reconstruct = ica.inverse_transform(S_)
    # # plt.plot(reconstruct)
    # # plt.show()
    # scaler = StandardScaler()
    # df_standard = scaler.fit_transform(df_raw)
    # plt.plot(df_standard)
    # plt.show()
    # df_standard_trans = ica.fit_transform(df_standard)
    # plt.plot(df_standard_trans)
    # plt.show()
    # df_standard_reconstruct = ica.inverse_transform(df_standard_trans)
    # plt.plot(df_standard_reconstruct)
    # plt.show()
    trans()
