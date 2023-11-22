import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import utils
import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument('--station', default=None, type=str,
                    help='Station to use')
parser.add_argument('--testdate', default= "2023/9/19", type=str,
                    help='Date to split train/test data')
parser.add_argument('--window_size', default= 7, type=int,
                    help='Size of window')
parser.add_argument('--is_daily', default= 1, type=int,
                    help='Daily or hourly data')
parser.add_argument("--predictor", default="temp_min", type=str,
                    help="Predictor to use")
parser.add_argument("--hidden_units", default=32, type=int,
                    help="Number of hidden units in LSTM")

device = ("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    # Load data
    args, unknown = parser.parse_known_args()
    station = args.station
    test_date = args.testdate
    window_size = args.window_size
    daily = args.is_daily
    predictor = args.predictor
    hidden_units = args.hidden_units
    learning_rate = 5e-3

    print("Loading data...")
    data = utils.load_data(daily, station)
    train_data, test_data = utils.train_test_split(data, test_date, daily)

    # Min temperature Data loaders
    train_data= utils.WindowDataset(train_data["temp_min"], window_size, 4)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)

    test_data = utils.WindowDataset(test_data["temp_min"], window_size, 4)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)

    # Create model
    model = utils.LSTM(input_size=window_size, hidden_size=hidden_units, output_size=4, num_layers=1, dropout=0.0)
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train model
    print("Training model...")
    utils.train_loop(model, train_loader,  optimizer, loss_function, epochs=200)

    # Test model
    print("Testing model...")
    print("TEST LOSS: {}".format(utils.test_model(model, test_loader, loss_function)))










