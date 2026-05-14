from keras.layers import LSTM, Dense
from keras.models import Sequential
from data_collection import gestures


def lstm_model():
    # This model consumes a sequence of 10 frames. Each frame contributes 63
    # values from MediaPipe hand landmarks, then the final softmax predicts one
    # of the patient-care gesture labels.
    model = Sequential()

    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(10, 63)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(gestures.shape[0], activation='softmax'))
    return model


# Create model instance
model = lstm_model()
