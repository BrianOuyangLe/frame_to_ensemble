import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Conv2D

def build_convlstm_model(height, width, channels):
    """
    input_shape: (height, width, channels)
    """
    model = Sequential()

    def ssim_loss(y_true, y_pred):
        return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
    
    '''
    model.add(ConvLSTM2D(
        filters=64,
        kernel_size=(3, 3),
        padding='same',
        return_sequences=False,
        input_shape=(time_steps, height, width, channels)
    ))
    '''
    model.add(Conv2D(
        filters=64,
        kernel_size=(16, 16),
        activation='relu',
        padding='same',
        input_shape=(height, width, channels)
    ))
    model.add(Conv2D(
        filters=32,
        kernel_size=(4, 4),
        activation='relu',
        padding='same',
    ))
    model.add(Conv2D(
        filters=32,
        kernel_size=(4, 4),
        activation='relu',
        padding='same',
    ))
    model.add(Conv2D(
        filters=1,
        kernel_size=(3, 3),
        activation='sigmoid',
        padding='same',
    ))

    model.compile(optimizer='adam', loss=ssim_loss, metrics=['mse'])
    return model
