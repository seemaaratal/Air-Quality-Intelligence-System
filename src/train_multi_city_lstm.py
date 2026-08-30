import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    LSTM,
    Dense,
    Dropout,
    Concatenate
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)


# --------------------------------------------------
# RANDOM SEED
# --------------------------------------------------

tf.keras.utils.set_random_seed(42)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

DATA_DIR = "data/model_ready_multi_city"


X_train = np.load(
    f"{DATA_DIR}/X_train.npy"
)

aux_train = np.load(
    f"{DATA_DIR}/aux_train.npy"
)

y_train = np.load(
    f"{DATA_DIR}/y_train.npy"
)


X_test = np.load(
    f"{DATA_DIR}/X_test.npy"
)

aux_test = np.load(
    f"{DATA_DIR}/aux_test.npy"
)

y_test = np.load(
    f"{DATA_DIR}/y_test.npy"
)


current_test = np.load(
    f"{DATA_DIR}/current_test.npy"
)


city_test = np.load(
    f"{DATA_DIR}/city_test.npy"
)


target_scaler = joblib.load(
    "models/multi_target_scaler.pkl"
)


print("\nMULTI-CITY LSTM TRAINING\n")

print(
    "X_train:",
    X_train.shape
)

print(
    "Aux train:",
    aux_train.shape
)

print(
    "y_train:",
    y_train.shape
)


# --------------------------------------------------
# CREATE VALIDATION SET
#
# Training samples were stored city-by-city.
# Keep final 10% of EACH city for validation.
# --------------------------------------------------

cities = [
    "Belagavi",
    "Hubballi",
    "Bengaluru",
    "Delhi",
    "Mumbai",
    "Pune",
    "Hyderabad",
    "Chennai",
    "Kolkata",
    "Ahmedabad"
]


number_of_cities = len(
    cities
)


samples_per_city = (
    len(X_train)
    // number_of_cities
)


train_indices = []

validation_indices = []


for city_number in range(
    number_of_cities
):

    start = (
        city_number
        * samples_per_city
    )

    end = (
        start
        + samples_per_city
    )


    city_sample_count = (
        end - start
    )


    validation_count = int(
        city_sample_count
        * 0.10
    )


    validation_start = (
        end
        - validation_count
    )


    train_indices.extend(
        range(
            start,
            validation_start
        )
    )


    validation_indices.extend(
        range(
            validation_start,
            end
        )
    )


train_indices = np.array(
    train_indices
)


validation_indices = np.array(
    validation_indices
)


X_fit = X_train[
    train_indices
]

aux_fit = aux_train[
    train_indices
]

y_fit = y_train[
    train_indices
]


X_val = X_train[
    validation_indices
]

aux_val = aux_train[
    validation_indices
]

y_val = y_train[
    validation_indices
]


print(
    "\nActual training samples:",
    len(X_fit)
)

print(
    "Validation samples:",
    len(X_val)
)


# --------------------------------------------------
# MODEL
# --------------------------------------------------

history_input = Input(
    shape=(
        X_train.shape[1],
        X_train.shape[2]
    ),
    name="history_input"
)


x = LSTM(
    64,
    return_sequences=True
)(
    history_input
)


x = Dropout(
    0.20
)(x)


x = LSTM(
    32
)(x)


x = Dropout(
    0.20
)(x)


# --------------------------------------------------
# AUXILIARY INPUT
# --------------------------------------------------

aux_input = Input(
    shape=(
        aux_train.shape[1],
    ),
    name="aux_input"
)


a = Dense(
    32,
    activation="relu"
)(
    aux_input
)


a = Dense(
    16,
    activation="relu"
)(a)


# --------------------------------------------------
# MERGE BOTH INPUTS
# --------------------------------------------------

combined = Concatenate()(
    [
        x,
        a
    ]
)


combined = Dense(
    32,
    activation="relu"
)(
    combined
)


combined = Dropout(
    0.20
)(
    combined
)


combined = Dense(
    16,
    activation="relu"
)(
    combined
)


output = Dense(
    2,
    name="aqi_delta"
)(
    combined
)


model = Model(
    inputs=[
        history_input,
        aux_input
    ],
    outputs=output
)


# --------------------------------------------------
# COMPILE
# --------------------------------------------------

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss=tf.keras.losses.Huber(),

    metrics=[
        tf.keras.metrics.MeanAbsoluteError(
            name="mae"
        )
    ]
)


model.summary()


# --------------------------------------------------
# CALLBACKS
# --------------------------------------------------

early_stopping = EarlyStopping(

    monitor="val_loss",

    patience=5,

    restore_best_weights=True,

    verbose=1
)


reduce_lr = ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.5,

    patience=2,

    min_lr=0.00001,

    verbose=1
)


# --------------------------------------------------
# TRAIN
# --------------------------------------------------

history = model.fit(

    [
        X_fit,
        aux_fit
    ],

    y_fit,

    validation_data=(

        [
            X_val,
            aux_val
        ],

        y_val
    ),

    epochs=30,

    batch_size=256,

    callbacks=[
        early_stopping,
        reduce_lr
    ],

    shuffle=False,

    verbose=1
)


# --------------------------------------------------
# SAVE MODEL
# --------------------------------------------------

os.makedirs(
    "models",
    exist_ok=True
)


MODEL_PATH = (
    "models/multi_city_lstm.keras"
)


model.save(
    MODEL_PATH
)


print(
    "\nModel saved:",
    MODEL_PATH
)


# --------------------------------------------------
# TEST PREDICTIONS
# --------------------------------------------------

print(
    "\nEvaluating test data..."
)


predicted_scaled = model.predict(

    [
        X_test,
        aux_test
    ],

    batch_size=512,

    verbose=1
)


# --------------------------------------------------
# INVERSE SCALE DELTAS
# --------------------------------------------------

predicted_delta = (
    target_scaler.inverse_transform(
        predicted_scaled
    )
)


actual_delta = (
    target_scaler.inverse_transform(
        y_test
    )
)


# --------------------------------------------------
# CONVERT DELTA BACK TO AQI
# --------------------------------------------------

predicted_24 = (
    current_test
    + predicted_delta[:, 0]
)


predicted_48 = (
    current_test
    + predicted_delta[:, 1]
)


actual_24 = (
    current_test
    + actual_delta[:, 0]
)


actual_48 = (
    current_test
    + actual_delta[:, 1]
)


# Keep predictions inside AQI range
predicted_24 = np.clip(
    predicted_24,
    0,
    500
)


predicted_48 = np.clip(
    predicted_48,
    0,
    500
)


# --------------------------------------------------
# PERSISTENCE BASELINE
#
# Baseline assumption:
# Future AQI = Current AQI
# --------------------------------------------------

baseline_24 = (
    current_test.copy()
)

baseline_48 = (
    current_test.copy()
)


# --------------------------------------------------
# OVERALL METRICS
# --------------------------------------------------

model_mae_24 = mean_absolute_error(
    actual_24,
    predicted_24
)


model_mae_48 = mean_absolute_error(
    actual_48,
    predicted_48
)


baseline_mae_24 = mean_absolute_error(
    actual_24,
    baseline_24
)


baseline_mae_48 = mean_absolute_error(
    actual_48,
    baseline_48
)


model_rmse_24 = np.sqrt(
    mean_squared_error(
        actual_24,
        predicted_24
    )
)


model_rmse_48 = np.sqrt(
    mean_squared_error(
        actual_48,
        predicted_48
    )
)


baseline_rmse_24 = np.sqrt(
    mean_squared_error(
        actual_24,
        baseline_24
    )
)


baseline_rmse_48 = np.sqrt(
    mean_squared_error(
        actual_48,
        baseline_48
    )
)


print(
    "\n================================="
)

print(
    "OVERALL MODEL PERFORMANCE"
)

print(
    "================================="
)


print("\n24-HOUR FORECAST")

print(
    "LSTM MAE:",
    round(
        model_mae_24,
        2
    )
)

print(
    "Baseline MAE:",
    round(
        baseline_mae_24,
        2
    )
)

print(
    "LSTM RMSE:",
    round(
        model_rmse_24,
        2
    )
)

print(
    "Baseline RMSE:",
    round(
        baseline_rmse_24,
        2
    )
)


print("\n48-HOUR FORECAST")

print(
    "LSTM MAE:",
    round(
        model_mae_48,
        2
    )
)

print(
    "Baseline MAE:",
    round(
        baseline_mae_48,
        2
    )
)

print(
    "LSTM RMSE:",
    round(
        model_rmse_48,
        2
    )
)

print(
    "Baseline RMSE:",
    round(
        baseline_rmse_48,
        2
    )
)


# --------------------------------------------------
# CITY-WISE PERFORMANCE
# --------------------------------------------------

results = []


print(
    "\n================================="
)

print(
    "CITY-WISE MAE"
)

print(
    "================================="
)


for city in np.unique(
    city_test
):

    mask = (
        city_test == city
    )


    city_model_24 = mean_absolute_error(

        actual_24[mask],

        predicted_24[mask]
    )


    city_model_48 = mean_absolute_error(

        actual_48[mask],

        predicted_48[mask]
    )


    city_baseline_24 = mean_absolute_error(

        actual_24[mask],

        baseline_24[mask]
    )


    city_baseline_48 = mean_absolute_error(

        actual_48[mask],

        baseline_48[mask]
    )


    print(
        f"\n{city}"
    )

    print(
        "24h LSTM:",
        round(
            city_model_24,
            2
        ),
        "| Baseline:",
        round(
            city_baseline_24,
            2
        )
    )

    print(
        "48h LSTM:",
        round(
            city_model_48,
            2
        ),
        "| Baseline:",
        round(
            city_baseline_48,
            2
        )
    )


    results.append({

        "City":
            city,

        "LSTM_MAE_24h":
            city_model_24,

        "Baseline_MAE_24h":
            city_baseline_24,

        "LSTM_MAE_48h":
            city_model_48,

        "Baseline_MAE_48h":
            city_baseline_48
    })


# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

os.makedirs(
    "outputs",
    exist_ok=True
)


results_df = pd.DataFrame(
    results
)


results_df.to_csv(

    "outputs/"
    "multi_city_model_results.csv",

    index=False
)


print(
    "\nResults saved:"
)

print(
    "outputs/"
    "multi_city_model_results.csv"
)