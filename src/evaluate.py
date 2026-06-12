from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import numpy as np


def evaluate_model(actual, predicted):

    mse = mean_squared_error(
        actual,
        predicted
    )

    rmse = np.sqrt(mse)

    mae = mean_absolute_error(
        actual,
        predicted
    )

    return mse, rmse, mae