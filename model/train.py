from model.architecture import build_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from model.architecture import load_existing_model

def train_model(x_train, y_train, model_file):
    model = build_model((x_train.shape[1], x_train.shape[2]))
    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=3)
    ]
    model.fit(x_train, y_train, epochs=100, batch_size=32, callbacks=callbacks)
    model.save(model_file)
    return model
