from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Conv2D

def build_convlstm_model(time_steps, height, width, channels):
    """
    input_shape: (time_steps, height, width, channels)
    """
    model = Sequential()

    model.add(ConvLSTM2D(
        filters=64,
        kernel_size=(3, 3),
        padding='same',
        return_sequences=False,
        input_shape=(time_steps, height, width, channels)
    ))

    model.add(Conv2D(
        filters=1,
        kernel_size=(3, 3),
        activation='sigmoid',
        padding='same'
    ))

    model.compile(optimizer='adam', loss='mse')
    return model
