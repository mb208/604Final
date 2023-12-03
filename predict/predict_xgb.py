import sys; sys.path.insert(0, '..') # add parent folder path where lib folder is
import numpy as np
import pandas as pd

from utils import utils
import argparse
import os
import datetime
from datetime import datetime as dt
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, accuracy_score
import json
import pickle

import warnings

import warnings
warnings.filterwarnings("ignore")



parser = argparse.ArgumentParser()
# parser.add_argument('--is_daily', default= 1, type=int,
#                     help='Daily or hourly data')
parser.add_argument("--test_date", default=None, type=str)

if __name__ == "__main__":
    # Variable setting
    args, unknown = parser.parse_known_args()

    # daily = args.is_daily
    test_date = args.test_date
    
    # Handle whether running from root or predict folder
    if os.getcwd().split('/')[-1] == "predict":
        model_path = "../models/xgb/"
    else:
        model_path = "./models/xgb/"

    names_models = {0 : "temp_min", 1 : "temp_max", 2 : "temp_mean", 3 : "rainfall", 4 : "snow"}
    pred_labels = ["XGBRegressor"]*3 + ["XGBClassifier"]*2
    
    loss_functions = {0 : lambda y,yhat: np.sqrt(mean_squared_error(y,yhat)), 
                      1 : lambda y,yhat: np.sqrt(mean_squared_error(y,yhat)), 
                      2 : lambda y,yhat: np.sqrt(mean_squared_error(y,yhat)),
                      3 : accuracy_score, 
                      4 : accuracy_score}
    

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
        results = pd.DataFrame({"date": days})
        results["date"] = pd.to_datetime(results["date"])

        for i in range(5):
            fcst = pickle.load(open(model_path + names_models[i] + ".pkl", "rb"))
            if i == 0:
                results = results.merge(fcst.predict(16), on=["date"]).sort_values(by="station")
            else:
                results = results.merge(fcst.predict(16), on=["date", "station"]).sort_values(by="station")
            results.rename(columns={pred_labels[i]: predictors[i]}, inplace=True)

        pred = {
            city_mapping[stat]: {
                str(day) : dict(zip(predictors, results.loc[(results["date"]  == dt.strftime(day, "%m-%d-%Y"))
                                                 &(results["station"]==stat),
                                                  predictors].values.tolist()[0]))
                for day in days
        } for stat in city_mapping.keys()
        }

        print(json.dumps(pred))
    else:
        # repeat_counts = {
        #         '2023-11-24': 1,
        #         '2023-11-25': 2,
        #         '2023-11-26': 3,
        #         '2023-11-27': 4,
        #         '2023-11-28': 4,
        #         '2023-11-29': 3,
        #         '2023-11-30': 2,
        #         '2023-12-01': 1
        #     }
        repeat_counts = dict(zip([(datetime.datetime.strptime(test_date, '%m/%d/%Y') + 
                                datetime.timedelta(days=i)).strftime('%Y-%m-%d') for i in range(0, 8)],
                                [1, 2, 3, 4, 4, 3, 2, 1]))
        results = pd.DataFrame()
        for date, count in repeat_counts.items():
            subset = data[data['date'] == date]
            results = results._append([subset] * count, ignore_index=True)
        # results.head()
        for i in range(1, 5):
            
            fcst = pickle.load(open(model_path + names_models[i] + ".pkl", "rb"))

            results = results.merge(fcst.predict(8), on=["station", "date"])
            print("TEST LOSS for model {} : {}".format(names_models[i], 
                                                   loss_functions[i](results[names_models[i]], 
                                                                     results[pred_labels[i]])))
            
            results.drop(columns=[pred_labels[i]],axis=1, inplace=True)
        
        
        
    
    