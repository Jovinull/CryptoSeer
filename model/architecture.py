# architecture.py (agora com suporte a Nadam)
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Bidirectional, Flatten, Input
from tensorflow.keras.optimizers import Adam, RMSprop, Nadam
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

    if tuning_optimizer == 'adam':
        optimizer = Adam(learning_rate=tuning_learning_rate)
    elif tuning_optimizer == 'rmsprop':
        optimizer = RMSprop(learning_rate=tuning_learning_rate)
    elif tuning_optimizer == 'nadam':
        optimizer = Nadam(learning_rate=tuning_learning_rate)
    else:
        raise ValueError(f"Otimizador não suportado: {tuning_optimizer}")

    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model


def load_existing_model(filepath):
    return load_model(filepath) if os.path.exists(filepath) else None
