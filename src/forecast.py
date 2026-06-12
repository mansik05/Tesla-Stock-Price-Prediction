import numpy as np


def forecast_days(
        model,
        data,
        scaler,
        days=5):

    future_input = list(data.flatten())

    forecast = []

    for i in range(days):

        x = np.array(
            future_input[-60:]
        )

        x = x.reshape(1,60,1)

        pred = model.predict(
            x,
            verbose=0
        )

        forecast.append(
            pred[0][0]
        )

        future_input.append(
            pred[0][0]
        )

    forecast = scaler.inverse_transform(
        np.array(forecast).reshape(-1,1)
    )

    return forecast