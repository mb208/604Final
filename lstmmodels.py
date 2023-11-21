import torch

class WindowDataset(torch.utils.data.Dataset):
    def __init__(self, data, window_size, output_size):
        self.data = torch.Tensor(data).float()
        self.window_size = window_size
        self.output_size = output_size
    
    def __len__(self):
        return len(self.data) - self.output_size + 1
    
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

