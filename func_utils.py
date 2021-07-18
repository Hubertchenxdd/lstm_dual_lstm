import datetime
import torch
import numpy as np
import pandas as pd


def date_range(target_dates, train_duration, test_duration):
    dates = []
    for date in target_dates:
        last_date = datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S.%f')
        test_date = last_date - datetime.timedelta(days=test_duration)
        train_date = test_date - datetime.timedelta(days=train_duration)
        dates.append([train_date, test_date, last_date])
    return dates


def PrepareData(matrix, steps=12):
    seg = matrix.columns.values
    time = matrix.index.values
    n_seg = len(seg)
    n_time = len(time)

    tps = matrix.to_numpy()

    short_data_set = []  # 預測時間之前3小時
    long_data_set = []  # 前一週3小時
    label_set = []
    hour_set = []
    dayofweek_set = []

    for i in range(672, n_time - steps):
        short_data = tps[i: i + steps]
        long_data = tps[i - 672: i - 672 + steps]
        # only for 1 time step
        label_data = tps[i + steps]

        if np.isnan(np.sum(short_data)) | np.isnan(np.sum(long_data)) | np.isnan(np.sum(label_data)):
            pass
        else:
            short_data_set.append(short_data)
            long_data_set.append(long_data)
            label_set.append(label_data)
            t = time[i + steps]
            hour_set.append(str(t)[11:13])
            dayofweek = datetime.datetime.strptime(str(t)[0:10], '%Y-%M-%d').strftime('%w')
            dayofweek_set.append(float(dayofweek))

    short_data_set = np.array(short_data_set)
    long_data_set = np.array(long_data_set)
    label_set = np.array(label_set)
    label_set = label_set.reshape(label_set.shape[0], 1, label_set.shape[1])
    hour_set = np.array(pd.get_dummies(hour_set))
    dayofweek_set = np.array(pd.get_dummies(dayofweek_set))

    return short_data_set, long_data_set, label_set, hour_set, dayofweek_set


def SplitData(s_data, l_data, hour, dow, label, train_por=0.7):
    sample_size = len(s_data)
    train_index = int(np.floor(sample_size * train_por))

    s_data_train = s_data[:train_index]
    l_data_train = l_data[:train_index]
    label_train = label[:train_index]
    hour_train = hour[:train_index]
    dow_train = dow[:train_index]

    s_data_val = s_data[train_index:]
    l_data_val = l_data[train_index:]
    label_val = label[train_index:]
    hour_val = hour[train_index:]
    dow_val = dow[train_index:]

    return s_data_train, l_data_train, label_train, hour_train, dow_train, \
           s_data_val, l_data_val, label_val, hour_val, dow_val


def to_Tensor(s_data, l_data, label, hour, dow):
    return torch.Tensor(s_data), torch.Tensor(l_data), torch.Tensor(label), torch.Tensor(hour), torch.Tensor(dow)
