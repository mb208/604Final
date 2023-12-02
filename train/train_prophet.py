import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import utils
from utils import models
import argparse
import os
from sklearn import preprocessing
from prophet import Prophet

from sklearn.metrics import mean_squared_error
import pickle


parser = argparse.ArgumentParser()
parser.add_argument('--testdate', default= "2023/11/16", type=str,
                    help='Date to split train/test data')

if __name__ == "__main__":
    # Variable setting
    args, unknown = parser.parse_known_args()
    test_date = args.testdate
    daily = True
    station=None
    

    response_vars = ["temp_min", "temp_max", "temp_mean"]
    list_of_vars = np.array(["temp_min", "temp_max", "temp_mean", 
                            "rainfall", "snow", "station"])
    model_path = "./models/prophet/"
    
    if os.path.exists(model_path) == False:
        os.mkdir(model_path)

    # Load data
    print("Loading data...")
    data = utils.load_data(daily, station)
    data["date"] = pd.to_datetime(data["date"])
    train_data_pd, test_data_pd = models.train_test_split(data, test_date, daily)
    

    station_ids = data.station.unique()
    stations = dict(zip(range(len(station_ids)), station_ids))
    
    
    model_dict = dict()
    for i in range(3):
        
        print("Training model {}...".format(response_vars[i]))
        # Create model
        idx = np.where(list_of_vars == response_vars[i])
        model_dict[i] = models.model_predictor(y=response_vars[i], 
                                               X=np.delete(list_of_vars, idx).tolist(),
                                               train=train_data_pd,
                                               test=test_data_pd, 
                                               stations=stations, 
                                               days=4)
        
        model_pkl_file = model_path + list_of_vars[i] + ".pkl"
        print("Saving model {}...".format(model_pkl_file))
        with open(model_pkl_file, 'wb') as file:  
            pickle.dump(model_dict, file)
            

        
    print("Done training and testing models")