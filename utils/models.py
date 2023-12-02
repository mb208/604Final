import torch
import math
import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error

class WindowDataset(torch.utils.data.Dataset):
    def __init__(self, data, window_size, output_size=4, predictor = 0):
        arr = np.array(data)
        if len(arr.shape) == 1:
            arr = arr.reshape(-1, 1)
        self.data = torch.Tensor(arr)
        self.window_size = window_size
        self.output_size = output_size
        self.predictor = predictor

    def __len__(self):
        return len(self.data) - self.output_size - 1
    
    def __getitem__(self, index):
        if index >= self.window_size - 1 and index + self.output_size + 1 <= len(self.data):
            x_start = index - self.window_size + 1
            x = self.data[x_start:index + 1, :]
            y = self.data[index + 1:index+self.output_size + 1, self.predictor:(self.predictor + 1)]
        elif index < self.window_size - 1:
            # x = torch.zeros(self.window_size)
            padding = self.data[0].repeat(self.window_size - index - 1, 1)
            x = self.data[0:(index + 1), :]
            x = torch.cat((padding, x), 0)
            y = self.data[index + 1:index+self.output_size + 1, self.predictor:(self.predictor + 1)]
        else:
            raise IndexError("Index out of bounds")
        
        if x.shape[1] == 1:
            x = x.squeeze()
        if y.shape[1] == 1:
            y = y.squeeze()
        return x, y
        # if index + self.window_size + self.output_size > len(self.data):
        #     raise IndexError("Index out of bounds")
        # if index + self.window_size + self.output_size > len(self.data):

        # return self.data[index:index+self.window_size], self.data[index+sel
        # f.window_size:index+self.window_size+self.output_size]
        
        
class LSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.0, sigmoid = False):
        super(LSTM, self).__init__()
        self.model_type = 'LSTM'
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.sigmoid = sigmoid
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, dropout=self.dropout, batch_first=True)
        self.linear = torch.nn.Linear(hidden_size, output_size)
        if sigmoid:
            self.activation = torch.nn.Sigmoid()
        # self.linear = torch.nn.Linear(hidden_size, output_size)

    def forward(self, input):
        h0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size).requires_grad_()
        c0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size).requires_grad_()


        output, (hidden, cell) = self.lstm(input, (h0.detach(), c0.detach()))
        output = self.linear(output[:, -1, :])
        if self.sigmoid:
            output = self.activation(output)
        return output 
class SpecialCrossEntropyLoss(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def forward(self, input, target):
        loss = torch.nn.CrossEntropyLoss()
        day1_loss = loss(input[:, 0], target[:, 0])
        day2_loss = loss(input[:, 1], target[:, 1])
        day3_loss = loss(input[:, 2], target[:, 2])
        day4_loss = loss(input[:, 3], target[:, 3])
        return day1_loss + day2_loss + day3_loss + day4_loss

# Prophet utilities
def model_predictor(y: str, X: list, train: pd.DataFrame, 
                    test: pd.DataFrame, stations: dict, 
                    days: int = 4, verbose: bool = True):
    
    models = dict()
    for i, station in stations.items():
        train_st = train.loc[train.station == station]
        test_st = test.loc[test.station == station]

        train_st = pd.DataFrame(train_st.rename(columns={'date': 'ds', y: 'y'}))
        test_st = pd.DataFrame(test_st.rename(columns={'date': 'ds', y: 'y'}))

        models[i] = Prophet()
        for x in X:
            models[i].add_regressor(name=x)

        models[i].fit(train_st[['ds', 'y'] + X])

        future = models[i].make_future_dataframe(periods=days)
        future = future.merge(test_st[['ds'] + X], on='ds') 

        forecast = models[i].predict(future)

        eval_m = (pd.DataFrame(test_st[['ds', 'y']])
                    .merge(forecast[['ds', 'yhat']], on='ds')
                    )
        if verbose:
            print(f"RMSE = {np.sqrt(mean_squared_error(eval_m.y, eval_m.yhat))}")
            
    return models

# I guess this is how you to do train test splits for temporal data 
def train_test_split(data, date, daily=True):
    if daily:
        train_data = data[data['date'] < date]
        test_data = data[data['date'] >= date]
    else:
        train_data = data[data['time'] < date]
        test_data = data[data['time'] >= date]
    return train_data, test_data


def train_model(model, data_loader, optimizer, loss_fn, device = "cpu"):
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()
        output = model(data.to(device))
        loss_t = loss_fn(output, target)
        loss_t.backward()
        optimizer.step()
        train_loss += loss_t.item()

    return train_loss / len(data_loader)

# MUST FIX THIS for classifier problems

def test_model(model, data_loader, loss_fn, device= None):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            if model.model_type == "Transformer":
                output = model(data, device)
            else:
                output = model(data)
            loss_t = loss_fn(output, target)
            test_loss += np.sqrt(loss_t.item())
    return test_loss / len(data_loader)

def test_model_classifier(model, data_loader, loss_fn):
    model.eval()
    test_loss = 0
    splits = np.zeros(4)
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            output = model(data).numpy()
            target = target.numpy()
            predictions = (output > 0.5).astype(int)
            splits = splits + (predictions != target)
            test_loss += np.sum(predictions != target)/target.shape[1]
    return test_loss / len(data_loader), splits/len(data_loader)

def train_loop(model, data_loader, optimizer, loss, device = "cpu", epochs=1):
    for epoch in range(epochs):
        train_loss = train_model(model, data_loader, optimizer, loss, device)
        if epoch % 10 == 0:
            print("Epoch: {}, Loss: {}".format(epoch, train_loss))

