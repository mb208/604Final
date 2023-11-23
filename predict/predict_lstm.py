import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

import utils
import argparse
import os
from sklearn import preprocessing


parser = argparse.ArgumentParser()
parser.add_argument('--station', default=None, type=str,
                    help='Station to use')
parser.add_argument('--window_size', default= 7, type=int,
                    help='Size of window')
parser.add_argument('--is_daily', default= 1, type=int,
                    help='Daily or hourly data')
parser.add_argument("--predictor", default=None, type=str,
                    help="Predictor to use")
parser.add_argument("--hidden_units", default=32, type=int,
                    help="Number of hidden units in LSTM")


device = ("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    # Variable setting
    args, unknown = parser.parse_known_args()

    station = args.station
    window_size = args.window_size
    daily = args.is_daily
    predictor = args.predictor
    hidden_units = args.hidden_units

    names_models = {
        0 : "temp_min",
        1 : "temp_max",
        2 : "temp_mean"
    }
    model_path = "models/lstm/"

    data = utils.pull_data(daily)

    le = preprocessing.LabelEncoder()
    le.fit(data["station"])

     # Map station to label ids
    station_mapping = dict(zip(le.transform(le.classes_), le.classes_))

    data["station"] = le.transform(data["station"])

     # Selecting predictors to use
    if predictor == None:
        list_of_vars = ["temp_min", "temp_max", "temp_mean"]
    else:
        list_of_vars =  [predictor, "station"]

    for i in range(3):
        # window_data = utils.WindowDataset(data[list_of_vars], window_size, 4, i)
        data_loader = torch.utils.data.DataLoader(window_data, batch_size=1, shuffle=False)
        
        model = utils.LSTM(input_size=len(list_of_vars), 
                           hidden_size=hidden_units,
                           output_size=4,
                           num_layers=1, 
                           dropout=0.0)
        
        model.load_state_dict(torch.load(os.path.join(model_path, names_models[i] + ".pth")))
        model.eval()

        with torch.no_grad():
            k=1
            for batch_idx, (data, target) in enumerate(data_loader):
                output = model(data)

                print(f"{names_models[i]} = {output}")
                k +=1
            print(k)

    



    