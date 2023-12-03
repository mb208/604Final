import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from itertools import product
from datetime import datetime, timedelta

data = pd.read_csv('/Users/Bakso/Documents/git/604Final/data/daily_data.csv')
data.dropna(inplace = True)
data['date'] = pd.to_datetime(data['date'])
latest_date = pd.to_datetime(data['date'].max())
nine_days_ago = latest_date - timedelta(days=9)
data = data[pd.to_datetime(data['date']) <= nine_days_ago]
stations = data['station'].unique()


column_names = ['station', 'date', 'temp_mean', 'temp_min', 'temp_max']
df = pd.DataFrame(columns=column_names)

pred_date_range = pd.date_range(start=nine_days_ago + pd.DateOffset(days=1), periods=9, freq='D')

def optimize_ARIMA(order_list, exog):
    results = []

    for order in order_list:
        model = SARIMAX(exog, order=order).fit(disp=-1)
        aic = model.aic
        results.append([order, aic])

    result_df = pd.DataFrame(results)
    result_df.columns = ['(p, d, q)', 'AIC']

    # Sort in ascending order, lower AIC is better
    result_df = result_df.sort_values(by='AIC', ascending=True).reset_index(drop=True)

    return result_df

for station in stations:
    train_data = data.loc[data['station'] == station].reset_index()

    df1 = pd.DataFrame(columns=column_names)
    df1['date'] = pred_date_range
    df1['station'] = [station] * 9

    for var in ['temp_mean', 'temp_min', 'temp_max']:
        ps = range(0, 5, 1)
        d = 0
        qs = range(1, 20, 2)

        parameters = product(ps, qs)
        parameters_list = list(parameters)

        order_list = []

        for each in parameters_list:
            each = list(each)
            each.insert(1, d)
            each = tuple(each)
            order_list.append(each)

        result = optimize_ARIMA(order_list, exog=train_data[var])

        params = np.array(result['(p, d, q)'])[0]

        best_model = SARIMAX(train_data[var], order= params).fit()

        predictions = np.array(best_model.forecast(steps=9))

        df1[var] = predictions

    df = pd.concat([df, df1], axis=0)

print(df)
df.to_csv('ARIMA_predictions.csv')