import torch
from torch.autograd import Variable
import time
import numpy as np

def TrainLSTM(model, train_dataloader, valid_dataloader, test_dataloader, learning_rate=1e-5, num_epochs=300,
              patience=10, min_delta=0.00001):

    loss_MSE = torch.nn.MSELoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model.cuda()

    losses_train = []
    losses_valid = []
    losses_test = []
    losses_epochs_train = []
    losses_epochs_valid = []
    losses_epochs_test = []

    pre_time = time.time()

    for epoch in range(num_epochs):
        trained_number = 0
        valid_dataloader_iter = iter(valid_dataloader)
        test_dataloader_iter = iter(test_dataloader)
        losses_epoch_train = []
        losses_epoch_valid = []
        losses_epoch_test = []

        for data in train_dataloader:
            inputs, inputs_l, inputs_hour, inputs_dow, labels = data

            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            model.zero_grad()

            outputs = model(inputs)

            loss_train = loss_MSE(outputs, torch.squeeze(labels))

            losses_train.append(loss_train.data)
            losses_epoch_train.append(loss_train.data)

            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

            # validation
            try:
                inputs_val, inputs_l_val, inputs_hour_val, inputs_dow_val, labels_val = next(valid_dataloader_iter)
            except StopIteration:
                valid_dataloader_iter = iter(valid_dataloader)
                inputs_val, inputs_l_val, inputs_hour_val, inputs_dow_val, labels_val = next(valid_dataloader_iter)

            if use_gpu:
                inputs_val, labels_val = Variable(inputs_val.cuda()), Variable(labels_val.cuda())
            else:
                inputs_val, labels_val = Variable(inputs_val), Variable(labels_val)

            outputs_val = model(inputs_val)

            loss_valid = loss_MSE(outputs_val, torch.squeeze(labels_val))
            losses_valid.append(loss_valid.data)
            losses_epoch_valid.append(loss_valid.data)

            # testing
            try:
                inputs_test, inputs_l_test, inputs_hour_test, inputs_dow_test, labels_test = next(test_dataloader_iter)
            except StopIteration:
                test_dataloader_iter = iter(test_dataloader)
                inputs_test, inputs_l_test, inputs_hour_test, inputs_dow_test, labels_test = next(test_dataloader_iter)

            if use_gpu:
                inputs_test, inputs_l_test, inputs_hour_test, inputs_dow_test, labels_test = Variable(
                    inputs_test.cuda()), Variable(inputs_l_test.cuda()), Variable(inputs_hour_test.cuda()), Variable(
                    inputs_dow_test.cuda()), Variable(labels_test.cuda())

            else:
                inputs_test, inputs_l_test, inputs_hour_test, inputs_dow_test, labels_test = Variable(
                    inputs_test), Variable(inputs_l_test), Variable(inputs_hour_test), Variable(
                    inputs_dow_test), Variable(labels_test)

            outputs_test = model(inputs_test)

            loss_test = loss_MSE(outputs_test, torch.squeeze(labels_test))
            losses_test.append(loss_test.data)
            losses_epoch_test.append(loss_test.data)

            # output
            trained_number += 1

        avg_losses_epoch_train = sum(losses_epoch_train) / float(len(losses_epoch_train))
        avg_losses_epoch_valid = sum(losses_epoch_valid) / float(len(losses_epoch_valid))
        avg_losses_epoch_test = sum(losses_epoch_test) / float(len(losses_epoch_test))
        losses_epochs_train.append(sum(losses_epoch_train) / float(len(losses_epoch_train)))
        losses_epochs_valid.append(sum(losses_epoch_valid) / float(len(losses_epoch_valid)))
        losses_epochs_test.append(sum(losses_epoch_test) / float(len(losses_epoch_test)))

        # Early Stopping
        if epoch == 0:
            is_best_model = 1
            best_model = model
            min_loss_epoch_valid = 10000.0
            if avg_losses_epoch_valid < min_loss_epoch_valid:
                min_loss_epoch_valid = avg_losses_epoch_valid
        else:
            if min_loss_epoch_valid - avg_losses_epoch_valid > min_delta:
                is_best_model = 1
                best_model = model
                min_loss_epoch_valid = avg_losses_epoch_valid
                patient_epoch = 0
            else:
                is_best_model = 0
                patient_epoch += 1
                if patient_epoch >= patience:
                    print('Early Stopped at Epoch:', epoch)
                    break

        # Print training parameters
        cur_time = time.time()
        print('Epoch: {}, train_loss: {:5f}, valid_loss: {:5f}, test_loss: {:5f}, time: {}, best model: {}'.format( \
            epoch, \
            np.around(avg_losses_epoch_train, decimals=8), \
            np.around(avg_losses_epoch_valid, decimals=8), \
            np.around(avg_losses_epoch_test, decimals=8), \
            np.around([cur_time - pre_time], decimals=2), \
            is_best_model))
        pre_time = cur_time
    return best_model, [losses_train, losses_valid, losses_test, losses_epochs_train, losses_epochs_valid, losses_epochs_test]


def TrainDLSTM(model, train_dataloader, valid_dataloader, test_dataloader, learning_rate=1e-5, num_epochs=300,
                patience=10, min_delta=0.00001):

    loss_MSE = torch.nn.MSELoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model.cuda()

    losses_train = []
    losses_valid = []
    losses_test = []
    losses_epochs_train = []
    losses_epochs_valid = []
    losses_epochs_test = []

    pre_time = time.time()

    for epoch in range(num_epochs):
        trained_number = 0
        valid_dataloader_iter = iter(valid_dataloader)
        test_dataloader_iter = iter(test_dataloader)
        losses_epoch_train = []
        losses_epoch_valid = []
        losses_epoch_test = []

        for data in train_dataloader:
            inputs, inputs_l, inputs_hour, inputs_dow, labels = data

            if use_gpu:
                inputs, inputs_l, inputs_hour, inputs_dow, labels = Variable(inputs.cuda()), Variable(
                    inputs_l.cuda()), Variable(inputs_hour.cuda()), Variable(inputs_dow.cuda()), Variable(labels.cuda())
            else:
                inputs, inputs_l, inputs_hour, inputs_dow, labels = Variable(inputs), Variable(inputs_l), Variable(
                    inputs_hour), Variable(inputs_dow), Variable(labels)

            model.zero_grad()

            outputs = model(inputs, inputs_l, inputs_hour, inputs_dow)

            loss_train = loss_MSE(outputs, torch.squeeze(labels))

            losses_train.append(loss_train.data)
            losses_epoch_train.append(loss_train.data)

            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

            # validation
            try:
                inputs_val, inputs_l_val, inputs_hour_val, inputs_dow_val, labels_val = next(valid_dataloader_iter)
            except StopIteration:
                valid_dataloader_iter = iter(valid_dataloader)
                inputs_val, inputs_l_val, inputs_hour_val, inputs_dow_val, labels_val = next(valid_dataloader_iter)

            if use_gpu:
                inputs_val, inputs_l_val, inputs_hour_val, inputs_dow_val, labels_val = Variable(
                    inputs_val.cuda()), Variable(inputs_l_val.cuda()), Variable(inputs_hour_val.cuda()), Variable(
                    inputs_dow_val.cuda()), Variable(labels_val.cuda())

            else:
                inputs_val, inputs_l_val, inputs_hour_val, inputs_dow_val, labels_val = Variable(inputs_val), Variable(
                    inputs_l_val), Variable(inputs_hour_val), Variable(inputs_dow_val), Variable(labels_val)

            outputs_val = model(inputs_val, inputs_l_val, inputs_hour_val, inputs_dow_val)

            loss_valid = loss_MSE(outputs_val, torch.squeeze(labels_val))
            losses_valid.append(loss_valid.data)
            losses_epoch_valid.append(loss_valid.data)

            # testing
            try:
                inputs_test, inputs_l_test, inputs_hour_test, inputs_dow_test, labels_test = next(test_dataloader_iter)
            except StopIteration:
                test_dataloader_iter = iter(test_dataloader)
                inputs_test, inputs_l_test, inputs_hour_test, inputs_dow_test, labels_test = next(test_dataloader_iter)

            if use_gpu:
                inputs_test, inputs_l_test, inputs_hour_test, inputs_dow_test, labels_test = Variable(
                    inputs_test.cuda()), Variable(inputs_l_test.cuda()), Variable(inputs_hour_test.cuda()), Variable(
                    inputs_dow_test.cuda()), Variable(labels_test.cuda())

            else:
                inputs_test, inputs_l_test, inputs_hour_test, inputs_dow_test, labels_test = Variable(
                    inputs_test), Variable(inputs_l_test), Variable(inputs_hour_test), Variable(
                    inputs_dow_test), Variable(labels_test)

            outputs_test = model(inputs_test, inputs_l_test, inputs_hour_test, inputs_dow_test)

            loss_test = loss_MSE(outputs_test, torch.squeeze(labels_test))
            losses_test.append(loss_test.data)
            losses_epoch_test.append(loss_test.data)

            # output
            trained_number += 1

        avg_losses_epoch_train = sum(losses_epoch_train) / float(len(losses_epoch_train))
        avg_losses_epoch_valid = sum(losses_epoch_valid) / float(len(losses_epoch_valid))
        avg_losses_epoch_test = sum(losses_epoch_test) / float(len(losses_epoch_test))
        losses_epochs_train.append(sum(losses_epoch_train) / float(len(losses_epoch_train)))
        losses_epochs_valid.append(sum(losses_epoch_valid) / float(len(losses_epoch_valid)))
        losses_epochs_test.append(sum(losses_epoch_test) / float(len(losses_epoch_test)))

        # Early Stopping
        if epoch == 0:
            is_best_model = 1
            best_model = model
            min_loss_epoch_valid = 10000.0
            if avg_losses_epoch_valid < min_loss_epoch_valid:
                min_loss_epoch_valid = avg_losses_epoch_valid
        else:
            if min_loss_epoch_valid - avg_losses_epoch_valid > min_delta:
                is_best_model = 1
                best_model = model
                min_loss_epoch_valid = avg_losses_epoch_valid
                patient_epoch = 0
            else:
                is_best_model = 0
                patient_epoch += 1
                if patient_epoch >= patience:
                    print('Early Stopped at Epoch:', epoch)
                    break

        # Print training parameters
        cur_time = time.time()
        print('Epoch: {}, train_loss: {:5f}, valid_loss: {:5f}, test_loss: {:5f}, time: {}, best model: {}'.format( \
            epoch, \
            np.around(avg_losses_epoch_train, decimals=8), \
            np.around(avg_losses_epoch_valid, decimals=8), \
            np.around(avg_losses_epoch_test, decimals=8), \
            np.around([cur_time - pre_time], decimals=2), \
            is_best_model))
        pre_time = cur_time
    return best_model, [losses_train, losses_valid, losses_test, losses_epochs_train, losses_epochs_valid, losses_epochs_test]
