import requests
import pandas as pd 
import numpy as np
import datetime
from datetime import datetime
import matplotlib.pyplot as plt
from meteostat import Point, Daily, Hourly, Stations
from dateutil.relativedelta import relativedelta 
import argparse



parser = argparse.ArgumentParser()
parser.add_argument('--timewindow', default=10, type=int,
                    help='Time window in years to get data')
parser.add_argument('--enddate', default=None, type=str,
                    help='End date to get data')
parser.add_argument('--traindatawindow', default=1, type=int,
                    help='Time window in years for train data')

lat_lons = {
    "KARB" : (42.2231,-83.7453), # Ann Arbor Airport
    "PANC" : (61.1743,-149.9963), # Ted Stevens Anchorage International Airport
    "KBOI" : (43.5644,-116.2228), # Boise Air Terminal
    "KORD" : (41.9742,-87.9073), # Chicago O'Hare International Airport
    "KDEN" : (39.8561,-104.6737), # Denver International Airport
    "KDTW" : (42.2125,-83.3533), # Detroit Metropolitan Airport
    "PHNL" : (21.3187,-157.9225), # Honolulu International Airport
    "KIAH" : (29.9844,-95.3414), # George Bush Intercontinental Airport
    "KMIA" : (25.7933,-80.2906), # Miami International Airport
    "KMIC" : (45.0628,-93.3533), # Minneapolis Crystal Airport
    "KOKC" : (35.3931,-97.6008), # Will Rogers World Airport
    "KBNA" : (36.1244,-86.6782), # Nashville International Airport
    "KJFK" : (40.6397,-73.7789), # John F. Kennedy International Airport
    "KPHX" : (33.4342,-112.0117), # Phoenix Sky Harbor International Airport
    "KPWM" : (43.6461,-70.3092), # Portland International Jetport
    "KPDX" : (45.5886,-122.5975), # Portland International Airport
    "KSLC" : (40.7884,-111.9778), # Salt Lake City International Airport
    "KSAN" : (32.7336,-117.1897), # San Diego International Airport
    "KSFO" : (37.6189,-122.3750), # San Francisco International Airport
    "KSEA" : (47.4489,-122.3094), # Seattle Tacoma International Airport
    "KDCA" : (38.8522,-77.0378), # Ronald Reagan Washington National Airport
}

if __name__ == "__main__":
    # Setting variables
    args, unknown = parser.parse_known_args()
    time_window = args.timewindow
    end_date = args.enddate
    train_data_window = args.traindatawindow
    if end_date == None:
        end_date = datetime.today()
    else:
        end_date = datetime(end_date)
    start_date = end_date - relativedelta(years=time_window)
    train_start_date = end_date - relativedelta(years=train_data_window)


    # Get Station
    print("Getting data from Meteostat...")
    stations = Stations()
    station_ids = []
    for icao, coords in lat_lons.items():
        lat, lon = coords
        station = stations.nearby(*coords).fetch(1)
        station_ids.append(station.index[0])
    hourly_df = Hourly(loc = station_ids, start=start_date, end=end_date).fetch()
    print("Done getting data from Meteostat")

    # hourly_df.to_csv("data/hourly_data_2.csv")

    data = hourly_df.copy()
    data = data.reset_index()
    data["date"] = data.time.dt.date
    data["is_snow"] = data["coco"].isin([14, 15, 16, 21, 22])

    daily_df = data.groupby(['station','date']).agg(temp_max=('temp', 'max'),
                                                    temp_mean=('temp', 'mean'),
                                                    temp_min=('temp', 'min'),
                                                    rainfall=('prcp', lambda x: (x > 0).any()),
                                                    snow=('is_snow', lambda x: (x > 0).any()))
    
    daily_df.to_csv("data/daily_data.csv")
    daily_df = daily_df.reset_index() 
    daily_df["date"] = pd.to_datetime(daily_df["date"])
    daily_df_with_history = daily_df[daily_df["date"] >= train_start_date]
    daily_df_with_history.set_index(["station", "date"], inplace=True)
    daily_df_with_history.to_csv("data/daily_data_one_year.csv")

    daily_df["week"] = pd.DatetimeIndex(daily_df['date']).strftime('%U')
    # aa_weather = daily_df[daily_df["station"] == "KARB"]
    daily_df["month"] = pd.DatetimeIndex(daily_df['date']).month
    daily_df["day"] = pd.DatetimeIndex(daily_df['date']).day
    daily_df["week"] = pd.DatetimeIndex(daily_df['date']).strftime('%U')
    daily_df["year"] = pd.DatetimeIndex(daily_df['date']).year

    historical_df = daily_df[daily_df["date"] < train_start_date]
    historical_df = historical_df.groupby(['week','station']).agg(hist_temp_max=('temp_max', 'mean'),
                                                                         hist_temp_mean=('temp_mean', 'mean'),
                                                                         hist_temp_min=('temp_min', 'mean'),
                                                                         hist_rainfall=('rainfall', 'mean'),
                                                                         hist_snow=('snow', 'mean'))
    # historical_df = historical_df.reset_index()
    historical_df.to_csv("data/historical_data.csv")
    # print(daily_df.join(historical_df, on=['week','station'], how='left'))
    


