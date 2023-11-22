import torch
import numpy as np
import pandas as pd
from sklearn import preprocessing

class WindowDataset(torch.utils.data.Dataset):
    def __init__(self, data, window_size, output_size, predictor = 0):
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

        # return self.data[index:index+self.window_size], self.data[index+self.window_size:index+self.window_size+self.output_size]
class LSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.0, regress=True):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, dropout=self.dropout, batch_first=True)
        if regress:
            self.linear = torch.nn.Linear(hidden_size, output_size)
        else:
            self.linear = torch.nn.Linear(hidden_size, 1)
        # self.linear = torch.nn.Linear(hidden_size, output_size)

    def forward(self, input):
        h0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size).requires_grad_()
        c0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size).requires_grad_()


        output, (hidden, cell) = self.lstm(input, (h0.detach(), c0.detach()))
        output = self.linear(output[:, -1, :])
        return output


def filter_data_by_station(data, station):
    return data[data['station'] == station]


def load_data(daily, station = None):
    if daily:
        data = pd.read_csv("data/daily_data.csv")
        data["date"] = pd.to_datetime(data["date"])
        data["rainfall"] = (data["rainfall"] == True).astype(int)
        data["snow"] = (data["snow"] == True).astype(int)
    else:
        data = pd.read_csv("data/hourly_data.csv")
        data["time"] = pd.to_datetime(data["time"])
    if station:
        data = filter_data_by_station(data, station)
    else:
        le = preprocessing.LabelEncoder()
        le.fit(data["station"])
        data["station"] = le.transform(data["station"])
    return data

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

def test_model(model, data_loader, loss_fn):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            output = model(data)
            loss_t = loss_fn(output, target)
            test_loss += np.sqrt(loss_t.item())
    return test_loss / len(data_loader)

def train_loop(model, data_loader, optimizer, loss, device = "cpu", epochs=1):
    for epoch in range(epochs):
        train_loss = train_model(model, data_loader, optimizer, loss, device)
        if epoch % 10 == 0:
            print("Epoch: {}, Loss: {}".format(epoch, train_loss))