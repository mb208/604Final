import sys; sys.path.insert(0, '..') # add parent folder path where lib folder is
import numpy as np
import pandas as pd
import torch

from utils import utils
from utils import models
import argparse
import os
import datetime
from sklearn import preprocessing
import json


parser = argparse.ArgumentParser()
parser.add_argument('--is_daily', default= 1, type=int,
                    help='Daily or hourly data')
parser.add_argument("--hidden_units", default=32, type=int,
                    help="Number of hidden units in LSTM")
parser.add_argument("--test_date", default=None, type=str)


device = ("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    # Variable setting
    args, unknown = parser.parse_known_args()

    daily = args.is_daily
    hidden_units = args.hidden_units
    
    # Handle whether running from root or predict folder
    if os.getcwd().split('/')[-1] == "predict":
        model_path = "../models/lstm/"
    else:
        model_path = "./models/lstm/"

    names_models = {0 : "temp_min", 1 : "temp_max", 2 : "temp_mean", 3 : "rainfall", 4 : "snow"}
    model_params = {0 : [64, 1, False], 1 : [32, 1, False] , 2 : [128, 1, False], 
                    3 : [128, 2, True], 4 : [128, 2, True]}
    
    output_transform = {0 : lambda x: x, 
                        1 : lambda x: x, 
                        2 : lambda x: x, 
                        3 : lambda x: x > .5, 
                        4 : lambda x: x > .5}
    # For creating output file
    days = [datetime.date.today() + datetime.timedelta(days=i) for i in range(1, 5)]
    predictors = ["min", "max", "avg", "rain", "snow"] # must line up with index of names_models

    # Storing values that will be needed during iteration
    vars = [["temp_min", "temp_max", "temp_mean", "station"]]*3 + \
        [["temp_min", "temp_max", "temp_mean", "rainfall", "snow", "station"]]*2
        
    
    data = utils.pull_data(daily)

    le = preprocessing.LabelEncoder()
    le.fit(data["station"])

    # Map station (city icao id)) to label ids
    data["station"] = le.transform(data["station"])
    city_mapping = dict(data[["station","icao"]].drop_duplicates().to_numpy())
    
    # Initial dictioary for city weather predicitons
    pred = dict()

    # iterate over the 5 different models
    for i in range(5):
        model = models.LSTM(input_size=len(vars[i]), 
                            hidden_size=model_params[i][0], 
                            output_size=4, num_layers=model_params[i][1],
                            dropout=0.0, sigmoid=model_params[i][2])
        
        model.load_state_dict(torch.load(os.path.join(model_path, names_models[i] + ".pth")))
        model.eval()
        test_data = torch.Tensor(np.array(data[vars[i]], dtype=np.float64))

        # collection predictions for outcome (i) for each city
        with torch.no_grad(): 
            for station in city_mapping.keys():
                pred[city_mapping[station]] = pred.setdefault(city_mapping[station],{})  
                output = model(test_data[test_data[:,-1]==station].unsqueeze(0)).ravel().tolist()

                for j,day in enumerate(days):
                    pred[city_mapping[station]][str(day)] = pred[city_mapping[station]].setdefault(str(day), {})
                    pred[city_mapping[station]][str(day)][predictors[i]] = output_transform[i](output[j])
    
    print(json.dumps(pred))

        