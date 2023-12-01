import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from utils import utils
from utils import models
import argparse
import os
from sklearn import preprocessing
import xgboost as xgb
from prophet import Prophet

import mlforecast
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score, roc_auc_score,  precision_score, recall_score
from window_ops.rolling import rolling_mean, rolling_max, rolling_min
import pickle


parser = argparse.ArgumentParser()
parser.add_argument('--testdate', default= "2023/11/16", type=str,
                    help='Date to split train/test data')

device = ("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    # Variable setting
    args, unknown = parser.parse_known_args()
    test_date = args.testdate
    daily = True
    station=None
    
    learning_rate = 51e-2

    list_of_vars = ["temp_min", "temp_max", "temp_mean", "rainfall", "snow", "station"]
    
    model_lists = [xgb.XGBRegressor(n_estimators=1000, learning_rate=learning_rate, n_jobs=4,random_state=0)]*3 + \
    [xgb.XGBClassifier(n_estimators=1000, learning_rate=learning_rate, n_jobs=4,random_state=0)]*2
    
    
    loss_functions = {0 : lambda x: np.sqrt(mean_squared_error(x)), 
                      1 : lambda x: np.sqrt(mean_squared_error(x)), 
                      2 : lambda x: np.sqrt(mean_squared_error(x)),
                      3 : accuracy_score, 
                      4 : accuracy_score}
    
    model_path = "../models/xgb/"

    # Load data
    print("Loading data...")
    data = utils.load_data(daily, station)
    data["date"] = pd.to_datetime(data["date"])
    train_data_pd, test_data_pd = models.train_test_split(data, test_date, daily)
    
    
    # Prparing data for xgboost fit
    train_data_pd.sort_values(by='date', ascending=True, inplace=True)
    test_data_pd.sort_values(by='date', ascending=True, inplace=True)
    
    train_stations = train_data_pd.station
    test_stations = test_data_pd.station
    
    train_data_pd = pd.get_dummies(train_data_pd, columns=['station'])
    train_data_pd['station'] = train_stations

    # test_data_pd  = pd.get_dummies(test_data_pd, columns=['station'])
    
    train_data_pd.dropna(inplace=True)
    test_data_pd.dropna(inplace=True)
    
    static_feats = [feat for feat in train_data_pd.columns if 'station' in feat]
    
    for feat in train_data_pd.columns:
        if 'snow' in feat or 'rain' in feat:
            train_data_pd[feat] = train_data_pd[feat].astype(int)
            test_data_pd[feat] = test_data_pd[feat].astype(int)
        

    for i in range(5):
        print("Training model {}...".format(i))

        # Create model
        fcst = mlforecast.MLForecast(models=[model_lists[i]],
                   freq='D',
                   lags=[1,7,14],
                   lag_transforms={
                       1: [(rolling_mean, 7)],
                   },
                   date_features=['dayofweek', 'month', "year", "dayofyear"],
                   num_threads=6)
        
        print("Training model {}...".format(list_of_vars[i]))
        fcst.fit(train_data_pd, time_col = 'date', target_col=list_of_vars[i], static_cols=static_feats)
        
        model_pkl_file = model_path + list_of_vars[i] + ".pkl"
        print("Saving model {}...".format(model_pkl_file))
        with open(model_pkl_file, 'wb') as file:  
            pickle.dump(fcst, model_pkl_file)
            
        
        results = test_data_pd.merge(fcst.predict(h = 4, X_df = test_data_pd),
                                     on = ['date', 'station'])
        
        print("TEST LOSS for model {} : {}".format(list_of_vars[i], 
                                                   loss_functions[i](results[list_of_vars[i]], 
                                                                     results["XGBClassifier"])))

    print("Done training and testing models")