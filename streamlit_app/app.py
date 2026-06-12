import streamlit as st
import pandas as pd
import numpy as np

from tensorflow.keras.models import load_model 
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# ------------------- #
# Load model #
# ------------------- #

MODEL_PATH = os.path.join("models", "best_lstm.keras")

model = load_model(MODEL_PATH)

# ------------------- #
# UI #
# ------------------- #

st.title(
    "Tesla Stock Price Prediction"
)

days = st.selectbox(
    "Select Forecast Days",
    [1,5,10]
)

# ----------------------- #
# Load Data #
# ----------------------- #
DATA_PATH = os.path.join("data", "TSLA.csv")

df = pd.read_csv(DATA_PATH)

close = df[['Close']]

scaler = MinMaxScaler()

scaled = scaler.fit_transform(
    close
)

# --------------------------- #
# Predict #
# --------------------------- #
if st.button("Predict"):

    # STEP 1: create input sequence
    future = list(scaled[-60:].flatten())

    result = []

    # STEP 2: predict future values
    for i in range(days):

        x = np.array(future[-60:]).reshape(1, 60, 1)

        pred = model.predict(x, verbose=0)

        result.append(pred[0][0])

        future.append(pred[0][0])

    # STEP 3: convert back to original scale
    result = scaler.inverse_transform(
        np.array(result).reshape(-1, 1)
    )

    # STEP 4: display nicely
    output = pd.DataFrame({
        "Day": [f"Day {i+1}" for i in range(days)],
        "Predicted Price": result.flatten()
    })

    st.subheader(" Prediction Result")
    st.table(output)