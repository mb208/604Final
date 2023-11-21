import torch
import numpy as np


class WindowDataset(torch.utils.data.Dataset):
    def __init__(self, data, window_size, output_size):
        arr = np.array(data)
        self.data = torch.Tensor(arr.reshape(-1, 1))
        self.window_size = window_size
        self.output_size = output_size

    def __len__(self):
        return len(self.data) - self.output_size - 1
    
    def __getitem__(self, index):
        if index >= self.window_size - 1 and index + self.output_size + 1 <= len(self.data):
            x_start = index - self.window_size + 1
            x = self.data[x_start:index + 1, :]
            y = self.data[index + 1:index+self.output_size + 1, :]
        elif index < self.window_size - 1:
            # x = torch.zeros(self.window_size)
            padding = self.data[0].repeat(self.window_size - index - 1, 1)
            x = self.data[0:(index + 1), :]
            x = torch.cat((padding, x), 0)
            y = self.data[index + 1:index+self.output_size + 1, :]
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
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.0):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, dropout=self.dropout)
        self.linear = torch.nn.Linear(hidden_size, output_size)

    def forward(self, input):
        output, (hidden, cell) = self.lstm(input)
        output = self.linear(output)
        return output


# I guess this is how you to do train test splits for temporal data 
def train_test_split(data, date):
    data["date"] = pd.to_datetime(data["date"])
    train_data = data[data['date'] < date]
    test_data = data[data['date'] >= date]
    return train_data, test_data


def filter_data_by_station(data, station):
    return data[data['station'] == station]


def train_model(model, data_loader, optimizer, loss_fn):
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()
        output = model(data)
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
            test_loss += loss_t.item()
    return test_loss / len(data_loader)

def train_loop(model, data_loader, optimizer, loss, epochs=1):
    for epoch in range(epochs):
        train_loss = train_model(model, data_loader, optimizer, loss)
        if epoch % 10 == 0:
            print("Epoch: {}, Loss: {}".format(epoch, train_loss))