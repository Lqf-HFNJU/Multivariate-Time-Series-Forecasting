import pandas as pd

from utils.timefeatures import time_features

if __name__ == '__main__':
    path = r'C:\Users\18933\Desktop\时间序列分析\hw5\final\Multivariate-Time-Series-Forecasting\code\dataset\electricity.csv'
    df_raw = pd.read_csv(path)
    date = df_raw['date']
    tmp_stamp = df_raw[['date']][:]
    tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
    df_stamp = pd.DataFrame(columns=['date'])
    df_stamp.date = list(tmp_stamp.date.values)
    df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
    df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
    df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
    df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
    df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
    df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
    print(df_stamp)
