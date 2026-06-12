import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def load_data(path):

    df = pd.read_csv(path)

    df['Date'] = pd.to_datetime(df['Date'])

    df.set_index('Date', inplace=True)

    return df


def clean_data(df):

    df.ffill(inplace=True)   # forward fill
    df.bfill(inplace=True)   # backward fill

    return df


def scale_close_price(df):

    close_data = df[['Close']]

    scaler = MinMaxScaler()

    scaled_data = scaler.fit_transform(close_data)

    return scaled_data, scaler


def create_sequences(data, time_step=60):

    X = []
    y = []

    for i in range(len(data)-time_step):

        X.append(data[i:i+time_step])

        y.append(data[i+time_step])

    return np.array(X), np.array(y)