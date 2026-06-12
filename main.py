from src.data_preprocessing import *
from src.model_rnn import *
from src.model_lstm import *
from src.evaluate import *

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint
)

import numpy as np
import os
os.makedirs("models", exist_ok=True)


df = load_data("data/TSLA.csv")

df = clean_data(df)

scaled_data, scaler = scale_close_price(df)

X, y = create_sequences(
    scaled_data,
    60
)

split = int(len(X)*0.8)

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]

X_train = X_train.reshape(
    X_train.shape[0],
    X_train.shape[1],
    1
)

X_test = X_test.reshape(
    X_test.shape[0],
    X_test.shape[1],
    1
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)


rnn_checkpoint = ModelCheckpoint(
    "models/best_rnn.keras",
    save_best_only=True
)

lstm_checkpoint = ModelCheckpoint(
    "models/best_lstm.keras",
    save_best_only=True
)

# ---------------------------- #
# Train RNN Model #
# ---------------------------- #
rnn = build_rnn()

rnn.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop, rnn_checkpoint]
)

# Train LSTM Model #
lstm = build_lstm()

lstm.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop, lstm_checkpoint]
)

# ----------------------------- #
# Evaluation #
# ----------------------------- #

rnn_pred = rnn.predict(X_test)

lstm_pred = lstm.predict(X_test)

actual = scaler.inverse_transform(y_test)

rnn_pred = scaler.inverse_transform(
    rnn_pred
)

lstm_pred = scaler.inverse_transform(
    lstm_pred
)

print(
    evaluate_model(
        actual,
        rnn_pred
    )
)

print(
    evaluate_model(
        actual,
        lstm_pred
    )
)
