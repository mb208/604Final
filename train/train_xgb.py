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
from window_ops.rolling import rolling_mean
import pickle


parser = argparse.ArgumentParser()
# default= "2023/11/24"
parser.add_argument('--testdate', default= None, type=str,
                    help='Date to split train/test data')

if __name__ == "__main__":
    # Variable setting
    args, unknown = parser.parse_known_args()
    test_date = args.testdate
    daily = True
    
    learning_rate = 51e-2
    
    model_lists = [xgb.XGBRegressor(n_estimators=1000, learning_rate=learning_rate, n_jobs=4,random_state=0)]*3 + \
    [xgb.XGBClassifier(n_estimators=1000, learning_rate=learning_rate, n_jobs=4,random_state=0)]*2
    
    pred_labels = ["XGBRegressor"]*3 + ["XGBClassifier"]*2
    
    loss_functions = {0 : lambda y,yhat: np.sqrt(mean_squared_error(y,yhat)), 
                      1 : lambda y,yhat: np.sqrt(mean_squared_error(y,yhat)), 
                      2 : lambda y,yhat: np.sqrt(mean_squared_error(y,yhat)),
                      3 : accuracy_score, 
                      4 : accuracy_score}
    
    model_path = "./models/xgb/"
    
    if os.path.exists(model_path) == False:
        os.mkdir(model_path)

    # Load data
    print("Loading data...")
    data = utils.load_data(daily, cov=True)
    data["date"] = pd.to_datetime(data["date"])
    
    # If test_date non None type then split data into train and test and evaluate trained model.
    if test_date:
        train_data_pd, test_data_pd = models.train_test_split(data, test_date, daily)
        test_data_pd.sort_values(by='date', ascending=True, inplace=True)
        print(test_data_pd.shape)
        test_data_pd.dropna(inplace=True)
        print(test_data_pd.shape)
        for feat in train_data_pd.columns:
            if 'snow' in feat or 'rain' in feat:
                test_data_pd[feat] = test_data_pd[feat].astype(int)
    else:
        train_data_pd = data.copy()
    
    list_of_vars = []
    # list_of_vars = data.drop(columns=["date", "station"]).columns.tolist()
    list_of_vars = ["temp_min", "temp_max", "temp_mean", "rainfall", "snow", "station"]

    
    # Prparing data for xgboost fit
    train_stations = train_data_pd.station.astype(str)
    train_data_pd.sort_values(by='date', ascending=True, inplace=True)
    
    # test_stations = test_data_pd.station
    # train_data_pd = pd.get_dummies(train_data_pd, columns=['station'])
    # train_data_pd['station'] = train_stations
    
    train_data_pd.dropna(inplace=True)
    

    # static_feats = [feat for feat in train_data_pd.columns if 'station' in feat]
    
    for feat in train_data_pd.columns:
        if 'snow' in feat or 'rain' in feat:
            train_data_pd[feat] = train_data_pd[feat].astype(int)
        

    for i in range(5):
        print("Training model {}...".format(i))

        # Create model
        fcst = mlforecast.MLForecast(models=[model_lists[i]],
                   freq='D',
                   lags=[1,2,3,4,5,6,7],
                   lag_transforms={
                       1: [(rolling_mean, 7)],
                   },
                   date_features=['dayofweek', 'month', "year", "dayofyear"],
                   num_threads=6)
        
        print("Training model {}...".format(list_of_vars[i]))
        fcst.fit(train_data_pd[['date', 'station'] + [list_of_vars[i]]],
                 id_col='station',
                 time_col='date',
                 target_col=list_of_vars[i], 
                #  max_horizon=4,
                 static_features=[]
                #  static_features=static_feats
                 )
        
        model_pkl_file = model_path + list_of_vars[i] + ".pkl"
        print("Saving model {}...".format(model_pkl_file))
        with open(model_pkl_file, 'wb') as file:  
            pickle.dump(fcst, file)
            
        if test_date:
            results = test_data_pd.merge(fcst.predict(h = 4),
                                                    #   , X_df = test_data_pd),
                                        on = ['date', 'station'])
            
            print("TEST LOSS for model {} : {}".format(list_of_vars[i], 
                                                    loss_functions[i](results[list_of_vars[i]], 
                                                                        results[pred_labels[i]])))

    print("Done training and testing models")