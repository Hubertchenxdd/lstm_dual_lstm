import pandas as pd
import random
import torch.utils.data as utils
from train import *
from model import *
from func_utils import *


random.seed(42)

data = pd.read_csv('./data/TPS raw.csv')
matrix = pd.DataFrame()
matrix['TIME'] = data.time.unique()
for seg in data.segmentID.unique():
    column = data[data['segmentID'] == seg][['time', 'TrafficIndex_GP']].drop_duplicates(subset=['time'])
    column.columns = ['TIME', str(seg)]
    matrix = matrix.join(column.set_index('TIME'), on='TIME')
matrix = matrix.drop(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '56', '57','71', '72', '101', '105'], axis = 1)

target_dates = ["2020-04-30 23:45:00.000", "2020-05-31 23:45:00.000",
                "2020-06-30 23:45:00.000", "2020-07-31 23:45:00.000",
                "2020-08-31 23:45:00.000", "2020-09-30 23:45:00.000",
                "2020-10-31 23:45:00.000", "2020-11-30 23:45:00.000",
                "2020-12-31 23:45:00.000", "2021-01-31 23:45:00.000",
                "2021-02-28 23:45:00.000", "2021-03-31 23:45:00.000"]


training_length = [30, 60, 90]
testing_length = 30
input_length = [1, 2, 3]

LSTM_models, LSLSTM_models = [], []
cnt = 0

for duration in training_length:
    for d in date_range(target_dates, duration, testing_length):
        train_dt_idx = pd.date_range(start = d[0], end = d[1], freq = "15min")
        test_dt_idx = pd.date_range(start = d[1], end = d[2], freq = "15min")
        train_data = pd.DataFrame(train_dt_idx)
        test_data = pd.DataFrame(test_dt_idx)
        train_data.columns = ['TIME']
        test_data.columns = ['TIME']
        train_data = train_data.set_index('TIME').join(matrix.set_index('TIME'))
        test_data = test_data.set_index('TIME').join(matrix.set_index('TIME'))

        for l in input_length:
            s_data, l_data, label, hour, dow = PrepareData(train_data, l * 4)
            s_data_train, l_data_train, label_train, hour_train, dow_train, \
            s_data_val, l_data_val, label_val, hour_val, dow_val = SplitData(s_data, l_data, hour, dow, label)
            s_data_test, l_data_test, label_test, hour_test, dow_test = PrepareData(test_data, l * 4)

            s_data_train, l_data_train, label_train, hour_train, dow_train = to_Tensor(s_data_train, l_data_train, label_train, hour_train, dow_train)
            s_data_val, l_data_val, label_val, hour_val, dow_val = to_Tensor(s_data_val, l_data_val, label_val, hour_val, dow_val)
            s_data_test, l_data_test, label_test, hour_test, dow_test = to_Tensor(s_data_test, l_data_test, label_test, hour_test, dow_test)

            train_dataset = utils.TensorDataset(s_data_train, l_data_train, hour_train, dow_train, label_train)
            valid_dataset = utils.TensorDataset(s_data_val, l_data_val, hour_val, dow_val, label_val)
            test_dataset = utils.TensorDataset(s_data_test, l_data_test, hour_test, dow_test, label_test)

            train_dataloader = utils.DataLoader(train_dataset, batch_size = 40, shuffle = True, drop_last = True)
            valid_dataloader = utils.DataLoader(valid_dataset, batch_size = 40, shuffle = True, drop_last = True)
            test_dataloader = utils.DataLoader(test_dataset, batch_size = 40, shuffle = False, drop_last = True)

            inputs, inputs_l, inputs_hour, inputs_dow, labels = next(iter(train_dataloader))
            [batch_size, step_size, fea_size] = inputs.size()
            input_dim = fea_size
            hidden_dim = fea_size
            output_dim = fea_size

            print("LSTM---------------------------------")
            print("train: {} to {}".format(d[0], d[1]))
            print("test: {} to {}".format(d[1], d[2]))
            print("input timestep: {}".format(l * 4))


            lstm = LSTM(input_dim, hidden_dim, output_dim, output_last = True)
            lstm, lstm_loss = TrainLSTM(lstm, train_dataloader, valid_dataloader, test_dataloader, num_epochs = 1000)

            model = {
                "model type": "LSTM",
                "training range": str(d[0]) + " to " + str(d[1]),
                "testing range": str(d[1]) + " to " + str(d[2]),
                "input timesteps": l * 4,
                "model": lstm,
                "loss_history": lstm_loss,
                "training data": train_dataloader,
                "validation data": valid_dataloader,
                "testing data": test_dataloader
            }
            LSTM_models.append(model)
            cnt += 1
            print("model saved, model trained: ", cnt)
            print("\n\n")

            print("LSLSTM-------------------------------")
            print("train: {} to {}".format(d[0], d[1]))
            print("test: {} to {}".format(d[1], d[2]))
            print("input timestep: {}".format(l * 4))
            dlstm = Dual_LSTM(input_dim, hidden_dim, output_dim, output_last = True)
            dlstm, dlstm_loss = TrainDLSTM(dlstm, train_dataloader, valid_dataloader, test_dataloader, num_epochs = 1000)

            model = {
                "model type": "LS-LSTM",
                "training range": str(d[0]) + " to " + str(d[1]),
                "testing range": str(d[1]) + " to " + str(d[2]),
                "input timesteps": l * 4,
                "model": dlstm,
                "loss_history": dlstm_loss,
                "training data": train_dataloader,
                "validation data": valid_dataloader,
                "testing data": test_dataloader
            }
            LSLSTM_models.append(model)
            cnt += 1
            print("model saved, model trained: ", cnt)
            print("\n\n")