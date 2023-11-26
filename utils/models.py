import torch
import math
import numpy as np
import pandas as pd
from sklearn import preprocessing

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

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    
class Transformer(torch.nn.Module):
    def __init__(self, d_model=512, output_size=2048, 
                 nhead=8, num_encoder_layers=6, num_decoder_layers=6,
                 num_layers=1, dropout=0.0, sigmoid = False):
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dropout = dropout
        self.sigmoid = sigmoid
        
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.embedding = nn.Embedding(output, dim_model)
        
        self.transformer = torch.nn.Transformer(input_size, 
                                                hidden_size, 
                                                num_layers, 
                                                dropout=self.dropout, batch_first=True)
        
        
        
        self.linear = torch.nn.Linear(d_model, output_size)
        
        if sigmoid:
            self.activation = torch.nn.Sigmoid()
            
    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    # def forward(self, input):
    #     h0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size).requires_grad_()
    #     c0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size).requires_grad_()


    #     output, (hidden, cell) = self.transformer(input, (h0.detach(), c0.detach()))
    #     output = self.linear(output[:, -1, :])
    #     if self.sigmoid:
    #         output = self.activation(output)
    #     return output   
    
    def forward(
        self,
        src,
        tgt,
    ):
        # Src size must be (batch_size, src sequence length)
        # Tgt size must be (batch_size, tgt sequence length)

        # Embedding + positional encoding - Out size = (batch_size, sequence length, dim_model)
        src = self.embedding(src) * math.sqrt(self.dim_model)
        tgt = self.embedding(tgt) * math.sqrt(self.dim_model)
        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)

        # we permute to obtain size (sequence length, batch_size, dim_model),
        src = src.permute(1, 0, 2)
        tgt = tgt.permute(1, 0, 2)

        # Transformer blocks - Out size = (sequence length, batch_size, num_tokens)
        transformer_out = self.transformer(src, tgt)
        out = self.out(transformer_out)

        if self.sigmoid:
            out = self.activation(out)
            
        return out
    
    def get_tgt_mask(self, size) -> torch.tensor:
        # Generates a squeare matrix where the each row allows one word more to be seen
        mask = torch.tril(torch.ones(size, size) == 1) # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0
    
        return mask
    
    def create_pad_mask(self, matrix: torch.tensor, pad_token: int) -> torch.tensor:
        # If matrix = [1,2,3,0,0,0] where pad_token=0, the result mask is
        # [False, False, False, True, True, True]
        return (matrix == pad_token)
    
    
    
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

def test_model(model, data_loader, loss_fn):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
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

