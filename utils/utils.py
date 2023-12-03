import torch
import numpy as np
import pandas as pd
from sklearn import preprocessing

from datetime import datetime, date, timedelta
import matplotlib.pyplot as plt
from meteostat import Point, Daily, Hourly, Stations
from dateutil.relativedelta import relativedelta 


def filter_data_by_station(data, station):
    return data[data['station'] == station]

def load_data(daily, station = None, trainwindow = None, cov = False):
    if daily:
        if cov == False:
            data = pd.read_csv("./data/daily_data.csv")
        else:
            data = pd.read_csv("./data/daily_data_addcov.csv")
            data["station"] = data["station"].astype(str)
        data["date"] = pd.to_datetime(data["date"])
        if trainwindow != None:
            start_date = datetime.today() - relativedelta(years=trainwindow)
            data = data[data["date"] >= start_date]
        data["rainfall"] = (data["rainfall"] == True).astype(int)
        data["snow"] = (data["snow"] == True).astype(int)
    else:
        data = pd.read_csv("../data/hourly_data.csv")
        data["time"] = pd.to_datetime(data["time"])
    if station:
        data = filter_data_by_station(data, station)
    else:
        le = preprocessing.LabelEncoder()
        le.fit(data["station"])
        data["station"] = le.transform(data["station"])
    return data

__CITIES = (
    "PANC KBOI KORD KDEN KDTW PHNL KIAH KMIA KMSP KOKC KBNA "
    "KARB KJFK KPHX KPWM KPDX KSLC KSAN KSFO KSEA KDCA"
).split(" ")

def pull_data(window=7,test_date=None):
    if test_date:
        t = datetime.strptime(test_date, '%m/%d/%Y')
    else:
        t = date.today()
    current_date = datetime(t.year, t.month, t.day)
    start = current_date - timedelta(days=window)
    stations = Stations()
    stations = stations.region(country="US").fetch()
    meteo_ids = list(stations[stations["icao"].isin(__CITIES)].index.unique())
    
    cities = (stations[stations["icao"]
                         .isin(__CITIES)]["icao"]
                         .drop_duplicates()
                         .reset_index()
                         .rename({"id":"station"}, axis=1))

    data = Hourly(loc = meteo_ids, start=start, end=current_date).fetch().reset_index()
    data["date"] = data.time.dt.date
    data["snow"] = data["coco"].isin([14, 15, 16, 21, 22])
    data = data.groupby(['station','date']).agg(temp_max=('temp', 'max'),
                                                temp_mean=('temp', 'mean'),
                                                temp_min=('temp', 'min'),
                                                rainfall=('prcp', lambda x: (x > 0).any()),
                                                snow=('snow', lambda x: (x > 0).any()),
                                                # other features,
                                                dwpt_mean=('dwpt', 'mean'),
                                                dwpt_max=('dwpt', 'max'),
                                                dwpt_min=('dwpt', 'min'),
                                                rhum_mean=('rhum', 'mean'),
                                                rhum_max=('rhum', 'max'),
                                                rhum_min=('rhum', 'min'),
                                                pres_mean=('pres', 'mean'),
                                                pres_max=('pres', 'max'),
                                                pres_min=('pres', 'min'))
    # Return torch dataset

    # le = preprocessing.LabelEncoder()
    # le.fit(data["station"])
    # data["station"] = le.transform(data["station"])
    return data.reset_index().merge(cities, on="station")
    
    
def load_data_with_historical(daily, station = None, trainwindow = None):
    if daily:
        if trainwindow == None:
            data = pd.read_csv("./data/daily_data_one_year.csv")
        else:
            data = pd.read_csv("./data/daily_data_{}.csv".format(trainwindow))
        data["date"] = pd.to_datetime(data["date"])
        data["rainfall"] = (data["rainfall"] == True).astype(int)
        data["snow"] = (data["snow"] == True).astype(int)
        data["week"] = pd.DatetimeIndex(data['date']).strftime('%U').astype(int)
        historical = pd.read_csv("../data/historical_data.csv")
        historical["week"] = historical["week"].astype(int)
        data = pd.merge(data, historical, on=['week','station'], how='left')
    else:
        data = pd.read_csv("../data/hourly_data.csv")
        data["time"] = pd.to_datetime(data["time"])
    if station:
        data = filter_data_by_station(data, station)
    else:
        le = preprocessing.LabelEncoder()
        print(data["station"].head())
        le.fit(data["station"])
        data["station"] = le.transform(data["station"])
    return data