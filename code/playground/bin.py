import pandas as pd
import matplotlib.pyplot as plt
from darts import TimeSeries

if __name__ == '__main__':
    path = r'C:\Users\18933\Desktop\时间序列分析\hw5\final\code\datasets\electricity\electricity.csv'
    df_raw = pd.read_csv(path)
    '''
    df_raw.columns: ['date', ...(other features), target feature]
    '''
    cols = list(df_raw.columns)
    cols.remove('OT')
    cols.remove('date')
    df_raw = df_raw[['date'] + cols + ['OT']]
    # print(cols)
    num_train = int(len(df_raw) * 0.7)
    num_test = int(len(df_raw) * 0.2)
    num_vali = len(df_raw) - num_train - num_test
    border1s = [0, num_train - 336, len(df_raw) - num_test - 336]
    border2s = [num_train, num_train + num_vali, len(df_raw)]
    border1 = border1s[0]
    border2 = border2s[0]
    print(df_raw)
