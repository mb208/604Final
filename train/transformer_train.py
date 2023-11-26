import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from utils import utils
from utils import models
import argparse
import os
from sklearn import preprocessing

parser = argparse.ArgumentParser()
parser.add_argument('--testdate', default= "2023/9/19", type=str,
                    help='Date to split train/test data')
parser.add_argument('--window_size', default= 7, type=int,
                    help='Size of window')
parser.add_argument('--is_daily', default= 1, type=int,
                    help='Daily or hourly data')
parser.add_argument("--d_model", default=512, type=int,
                    help="embedding dimension")
parser.add_argument("--nhead", default=8, type=int,
                    help="Number of attention heads in transformer")
parser.add_argument("--n_enc", default=6, type=int,
                    help="Number of encoder layers in transformer")
parser.add_argument("--n_dec", default=6, type=int,
                    help="Number of decoder layers in transformer")
parser.add_argument("--dropout", default=0.1, type=int,
                    help="Number of decoder layers in transformer")
parser.add_argument("--dim_ff", default=2048, type=int,
                    help="dim_feedforward`` in transformer")

device = ("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    # Variable setting
    args, unknown = parser.parse_known_args()
    test_date = args.testdate
    window_size = args.window_size
    daily = args.is_daily
    nhead = args.nhead
    d_model = args.d_model
    n_enc = args.n_enc
    n_dec = args.n_dec
    dropout = args.dropout
    dim_ff = args.dim_ff
    learning_rate = 5e-3
    if predictor == None:
        list_of_vars = ["temp_min", "temp_max", "temp_mean", "station"]
        list_of_vars_2 = ["temp_min", "temp_max", "temp_mean", "rainfall", "snow", "station"]
        list_of_vars_3 = ["temp_min", "temp_max", "temp_mean", "hist_temp_min", "hist_temp_max", "hist_temp_mean", "station"]
        list_of_vars_4 = ["temp_min", "temp_max", "temp_mean", "rainfall", "snow", "hist_temp_min", "hist_temp_max",
                           "hist_temp_mean", "hist_rainfall", "hist_snow", "station"]
    else:
        list_of_vars =  [predictor, "station"]
        list_of_vars_2 = [predictor, "station"]
        list_of_vars_3 =  [predictor, "station"]
        list_of_vars_4 = [predictor, "station"]
    names_models = {0 : ["temp_min", "temp_min_hist"], 1 : ["temp_max", "temp_max_hist"], 2 : ["temp_mean", "temp_mean_hist"], 
                    3 : ["rainfall", "rainfall_hist"], 4 : ["snow", "snow_hist"]}
    loss_functions = {0 : torch.nn.MSELoss(), 1 : torch.nn.MSELoss(), 2 : torch.nn.MSELoss(),
        3 : models.SpecialCrossEntropyLoss(), 4 : models.SpecialCrossEntropyLoss()}
    predictors_list = { 0 : [list_of_vars, list_of_vars_3], 1 : [list_of_vars, list_of_vars_3], 2 : [list_of_vars, list_of_vars_3],
        3 : [list_of_vars_2, list_of_vars_4], 4 : [list_of_vars_2, list_of_vars_4]}
    test_processes = {0 : models.test_model, 1 : models.test_model, 2 : models.test_model,
        3 : models.test_model_classifier, 4 : models.test_model_classifier}
    model_path = "../models/lstm/"

    # Load data
    print("Loading data...")
    data = utils.load_data(daily = daily)
    train_data_pd, test_data_pd = models.train_test_split(data, test_date, daily)

    # Load historical data
    print("Loading historical data...")
    historical_data = utils.load_data_with_historical(daily)
    train_data_hist_pd, test_data_hist_pd = models.train_test_split(historical_data, test_date, daily)

   

    for i in range(5):
        train_data= models.WindowDataset(train_data_pd[predictors_list[i][0]], window_size, 4, i)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)

        test_data = models.WindowDataset(test_data_pd[predictors_list[i][0]], window_size, 4, i)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
        x, y = next(iter(train_loader))

        # Create model
        inputsize = x.shape[2]
        model = models.Transformer(d_model=d_model, 
                                   nhead=nhead, 
                                   num_encoder_layers=n_enc,
                                   num_decoder_layers=n_dec, 
                                   dropout=dropout,
                                   dim_feedforward=dim_ff,
                                   output_size=4)
        loss_function = loss_functions[i]
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Train model
        print("Training model {}...".format(names_models[i][0]))
        models.train_loop(model, train_loader,  optimizer, loss_function, epochs=200)
        torch.save(model.state_dict(), model_path + names_models[i][0] + ".pth")

        # Test model
        print("Testing model {}...".format(names_models[i][0]))
        print("TEST LOSS for model {} : {}".format(names_models[i][0], test_processes[i](model, test_loader, loss_function)))

        # Train model with historical data
        train_data_hist= models.WindowDataset(train_data_hist_pd[predictors_list[i][1]], window_size, 4, i)
        train_loader_hist = torch.utils.data.DataLoader(train_data_hist, batch_size=32, shuffle=True)

        test_data_hist = models.WindowDataset(test_data_hist_pd[predictors_list[i][1]], window_size, 4, i)
        test_loader_hist = torch.utils.data.DataLoader(test_data_hist, batch_size=1, shuffle=False)
        x, y = next(iter(train_loader_hist))

        # Create model
        inputsize = x.shape[2]
        model_hist = models.Transformer(d_model=d_model, 
                                   nhead=nhead, 
                                   num_encoder_layers=n_enc,
                                   num_decoder_layers=n_dec, 
                                   dropout=dropout,
                                   dim_feedforward=dim_ff,
                                   output_size=4)
        loss_function = loss_functions[i]
        optimizer = torch.optim.Adam(model_hist.parameters(), lr=learning_rate)

        # Train model
        print("Training model {}...".format(names_models[i][1]))
        models.train_loop(model_hist, train_loader_hist,  optimizer, loss_function, epochs=50)
        torch.save(model_hist.state_dict(), model_path + names_models[i][1] + ".pth")

        # Test model
        print("Testing model {}...".format(names_models[i][1]))
        print("TEST LOSS for model {} : {}".format(names_models[i][1], test_processes[i](model_hist, test_loader_hist, loss_function)))

    

    print("Done training and testing models")