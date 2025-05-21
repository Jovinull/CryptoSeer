from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Bidirectional, Flatten
import os
from config import tuning_units, tuning_dropout

def build_model(input_shape, model_type='LSTM'):
    model = Sequential()
    RNNLayer = LSTM if model_type == 'LSTM' else GRU

    if model_type in ['LSTM', 'GRU']:
        model.add(Bidirectional(RNNLayer(tuning_units, return_sequences=True), input_shape=input_shape))
        model.add(Dropout(tuning_dropout))
        model.add(Bidirectional(RNNLayer(tuning_units, return_sequences=True)))
        model.add(Dropout(tuning_dropout))
        model.add(Bidirectional(RNNLayer(tuning_units)))
        model.add(Dropout(tuning_dropout))
    else:
        model.add(Flatten(input_shape=input_shape))
        model.add(Dense(100, activation='relu'))
        model.add(Dropout(tuning_dropout))

    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def load_existing_model(filepath):
    return load_model(filepath) if os.path.exists(filepath) else None