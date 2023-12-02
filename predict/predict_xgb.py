import sys; sys.path.insert(0, '..') # add parent folder path where lib folder is
import numpy as np
import pandas as pd

from utils import utils
from utils import models
import argparse
import os
import datetime
from sklearn import preprocessing
import json
import pickle


parser = argparse.ArgumentParser()
# parser.add_argument('--is_daily', default= 1, type=int,
#                     help='Daily or hourly data')
parser.add_argument("--test_date", default=None, type=str)

if __name__ == "__main__":
    # Variable setting
    args, unknown = parser.parse_known_args()

    # daily = args.is_daily
    test_date = args.test_date
    test_date="11/24/2023"
    
    # Handle whether running from root or predict folder
    if os.getcwd().split('/')[-1] == "predict":
        model_path = "../models/xgb/"
    else:
        model_path = "./models/xgb/"

    names_models = {0 : "temp_min", 1 : "temp_max", 2 : "temp_mean", 3 : "rainfall", 4 : "snow"}

    # For creating output file
    
    predictors = ["min", "max", "avg", "rain", "snow"] 
    data = utils.pull_data(test_date=test_date)
    data["date"] = pd.to_datetime(data["date"])
    
    le = preprocessing.LabelEncoder()
    le.fit(data["station"])

    # Map station (city icao id)) to label ids
    data["station"] = le.transform(data["station"])
    city_mapping = dict(data[["station","icao"]].drop_duplicates().to_numpy())
    
    if test_date is None:
        days = [datetime.date.today() + datetime.timedelta(days=i) for i in range(1, 5)]
    else:
        fcst = pickle.load(open(model_path + names_models[i] + ".pkl", "rb"))
        
        fcst.predict(1, data.loc[data.station==0])
        
        
    
    