import tensorflow as tf
from tensorflow.keras.layers import *
import constants
import numpy as np

frames = np.load("frames.npy")
speeds = np.load("speeds.npy")

print(np.shape(frames))
ds = tf.keras.preprocessing.timeseries_dataset_from_array(
    data=frames,
    targets=speeds,
    sequence_length=constants.frame_window_size,
    sampling_rate=1,
    sequence_stride=1,
    shuffle=True,
    batch_size=32
)
# ds.shuffle(len(frames))
validation_set_size = round(ds.__len__().numpy() * 0.1)
validation_ds = ds.take(validation_set_size)
train_ds = ds.skip(validation_set_size)

# simple cnn lstm architecture
# lrcn_model = tf.keras.Sequential()
# lrcn_model.add(Conv2D(32, (5, 5), input_shape = (constants.frame_window_size,) + np.shape(frames[0])))
# lrcn_model.add(MaxPooling2D((2, 2)))
# lrcn_model.add(LeakyReLU())
# lrcn_model.add(BatchNormalization())
# lrcn_model.add(Conv2D(64, (3, 3)))
# lrcn_model.add(MaxPooling2D((2, 2)))
# lrcn_model.add(LeakyReLU())
# lrcn_model.add(BatchNormalization())
# lrcn_model.add(Conv2D(32, (3, 3)))
# lrcn_model.add(MaxPooling2D((2, 2)))
# lrcn_model.add(LeakyReLU())
# lrcn_model.add(BatchNormalization())
# lrcn_model.add(Conv2D(8, (3, 3)))
# lrcn_model.add(MaxPooling2D((2, 2)))
# lrcn_model.add(LeakyReLU())
# lrcn_model.add(BatchNormalization())
# lrcn_model.add(Flatten())
# lrcn_model.add(Dense(32, activation='sigmoid'))
# lrcn_model.add(Dense(16, activation='relu'))
# lrcn_model.add(Dense(1))


def build_model_flat():
    #frame_inp = Input(shape=(224, 224, 3))
    op_flow_inp = Input(shape=(constants.frame_window_size,) + np.shape(frames[0]))
    filters = [3, 5]
    op_flows = []
    # op_flow = BatchNormalization()(op_flow_inp)
    op_flow = (op_flow_inp)
    for i, filter_size in enumerate(filters):
        int_layer = Dropout(.2)(op_flow)
        int_layer = Conv2D(8, (filter_size,filter_size), activation = "relu", data_format = "channels_last")(int_layer)
        int_layer = MaxPooling2D(pool_size = (1,2))(int_layer)
        int_layer = Conv2D(16, (filter_size,filter_size), activation = "relu", data_format = "channels_last")(int_layer)
        int_layer = MaxPooling2D(pool_size = (1,2))(int_layer)
        int_layer = Conv2D(32, (filter_size,filter_size), activation = "relu", data_format = "channels_last")(int_layer)
        int_layer = Conv2D(64, (filter_size,filter_size), activation = "relu", data_format = "channels_last")(int_layer)
        #int_layer = Dropout(.3)(int_layer)
        int_layer = Conv2D(128, (filter_size,filter_size), activation = "relu", data_format = "channels_last")(int_layer)
        int_layer = MaxPooling2D()(int_layer)
        int_layer = Conv2D(256, (filter_size,filter_size), activation = "relu", data_format = "channels_last")(int_layer)
        int_layer = MaxPooling2D()(int_layer)
        int_layer = Conv2D(512, (filter_size,filter_size), activation = "relu",
                           data_format = "channels_last", padding = "same")(int_layer)
        int_layer = MaxPooling2D()(int_layer)
        int_layer_max = GlobalMaxPool2D()(int_layer)
        int_layer_avg = GlobalAvgPool2D()(int_layer)
        conc = concatenate([int_layer_max, int_layer_avg])
        op_flows.append(conc)
    conc = concatenate(op_flows)
    #conc = BatchNormalization()(conc)
    #conc = SpatialDropout1D(.2)(conc)
    #conc = CuDNNGRU(256)(conc)
#     conc = Dropout(.2)(conc)
    conc = Dense(500, activation = "relu")(conc)
#     conc = Dropout(.2)(conc)
    conc = Dense(250, activation = "relu")(conc)
#     conc = Dropout(.1)(conc)
    result = Dense(1, activation='linear')(conc)
    
    model = Model(inputs=[
        #frame_inp,
        op_flow_inp], outputs=[result])
    print(model.summary())
    model.compile(loss="mse", optimizer='adam')

    return model

lrcn_model = build_model_flat()

# CNN
# lrcn_model.add(TimeDistributed(Conv2D(32, (3,3)), input_shape=(constants.frame_window_size,) + np.shape(frames[0])))
# lrcn_model.add(TimeDistributed(MaxPooling2D((2, 2))))
# lrcn_model.add(TimeDistributed(LeakyReLU()))
#
# lrcn_model.add(TimeDistributed(Conv2D(64, (3, 3))))
# lrcn_model.add(TimeDistributed(MaxPooling2D((2, 2))))
# lrcn_model.add(TimeDistributed(LeakyReLU()))
#
# lrcn_model.add(TimeDistributed(Conv2D(128, (3, 3))))
# lrcn_model.add(TimeDistributed(MaxPooling2D((2, 2))))
# lrcn_model.add(TimeDistributed(LeakyReLU()))
#
# lrcn_model.add(TimeDistributed(Flatten()))

# LSTM
# lrcn_model.add(LSTM(128, activation='tanh', return_sequences=True))
# lrcn_model.add(LSTM(64, activation='tanh', return_sequences=True))
# lrcn_model.add(Dropout(0.5))
# lrcn_model.add(Dense(128, activation='relu'))
# lrcn_model.add(Dense(64, activation='relu'))
# lrcn_model.add(Dropout(0.5))
# lrcn_model.add(Dense(32, activation='relu'))
# lrcn_model.add(Dense(1, activation='relu'))

lrcn_model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-4),
    loss="mse")
print(lrcn_model.summary())

tf.keras.utils.plot_model(
    lrcn_model,
    to_file="architecture_model.png",
    show_shapes=True)

earlystopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=7, restore_best_weights=True)

lrcn_model.fit(
    x=train_ds,
    epochs=30,
    validation_data=validation_ds,
    callbacks=[earlystopping_callback])

lrcn_model.save(constants.model_path)
