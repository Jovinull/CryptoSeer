from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Bidirectional, Flatten, Input
from tensorflow.keras.optimizers import Adam, RMSprop
import os
from config import (
    tuning_units, tuning_dropout, tuning_num_layers,
    tuning_bidirectional, tuning_optimizer, tuning_learning_rate
)

def build_model(input_shape, model_type='LSTM'):
    model = Sequential()
    model.add(Input(shape=input_shape))

    RNNLayer = LSTM if model_type == 'LSTM' else GRU

    if model_type in ['LSTM', 'GRU']:
        for i in range(tuning_num_layers):
            return_seq = i < tuning_num_layers - 1
            layer = RNNLayer(tuning_units, return_sequences=return_seq)
            if tuning_bidirectional:
                model.add(Bidirectional(layer))
            else:
                model.add(layer)
            model.add(Dropout(tuning_dropout))
    else:
        model.add(Flatten())
        for _ in range(tuning_num_layers):
            model.add(Dense(tuning_units, activation='relu'))
            model.add(Dropout(tuning_dropout))

    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))

    optimizer = Adam(learning_rate=tuning_learning_rate) if tuning_optimizer == 'adam' else RMSprop(learning_rate=tuning_learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model


def load_existing_model(filepath):
    return load_model(filepath) if os.path.exists(filepath) else None