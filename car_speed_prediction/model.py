import tensorflow as tf
from tensorflow.keras.layers import *
import constants
import numpy as np

frames = np.load("frames.npy")
speeds = np.load("speeds.npy")

ds = tf.keras.preprocessing.timeseries_dataset_from_array(
        data=frames,
        targets=speeds,
        sequence_length=constants.frame_window_size,
        sampling_rate=2,
        sequence_stride=2,
        shuffle=True,
        batch_size=32
)

validation_set_size = round(ds.__len__().numpy() * 0.1)
validation_ds = ds.take(validation_set_size)
train_ds = ds.skip(validation_set_size)

# simple cnn lstm architecture

lrcn_model = tf.keras.Sequential()

# CNN
lrcn_model.add(TimeDistributed(Conv2D(32, (3, 3)), input_shape=(constants.frame_window_size,) + constants.image_size))
lrcn_model.add(TimeDistributed(MaxPooling2D((2, 2))))
lrcn_model.add(TimeDistributed(LeakyReLU()))

lrcn_model.add(TimeDistributed(Conv2D(64, (3, 3))))
lrcn_model.add(TimeDistributed(MaxPooling2D((2, 2))))
lrcn_model.add(TimeDistributed(LeakyReLU()))

lrcn_model.add(TimeDistributed(Conv2D(128, (3, 3))))
lrcn_model.add(TimeDistributed(MaxPooling2D((2, 2))))
lrcn_model.add(TimeDistributed(LeakyReLU()))

lrcn_model.add(TimeDistributed(Flatten()))

# LSTM

lrcn_model.add(SimpleRNN(128, activation='tanh'))
lrcn_model.add(Dense(1, activation='relu'))

lrcn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=10e-4), loss="mse")
print(lrcn_model.summary())

tf.keras.utils.plot_model(
    lrcn_model,
    to_file="architecture_model.png",
    show_shapes=True)

earlystopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

lrcn_model.fit(x=train_ds, epochs=30, validation_data=validation_ds, callbacks=[earlystopping_callback])

lrcn_model.save(constants.model_path)

