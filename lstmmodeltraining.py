import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import utils
import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument('--station', default=None, type=str,
                    help='Station to use')
parser.add_argument('--testdate', default= "2023/9/19", type=str,
                    help='Date to split train/test data')
parser.add_argument('--window_size', default= 7, type=int,
                    help='Size of window')
parser.add_argument('--is_daily', default= 1, type=int,
                    help='Daily or hourly data')
parser.add_argument("--predictor", default="temp_min", type=str,
                    help="Predictor to use")

if __name__ == "__main__":
    # Load data
    args, unknown = parser.parse_known_args()
    station = args.station
    test_date = args.testdate
    window_size = args.window_size
    daily = args.is_daily

    # Test something something


    
    print("Loading data...")
    data = utils.load_data(daily, station)
    train_data, test_data = utils.train_test_split(data, test_date)
    print(test_data.head())





