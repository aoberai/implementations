'''
Model architecure inspired from: https://arxiv.org/pdf/1412.0767.pdf
'''

import tensorflow as tf
from tensorflow.keras.layers import *
import numpy as np
import constants as ct

'''
TODO: Currently not using OpFlow
'''

frames_X = np.load("frames.npy")
# flow_X = np.load("flow.npy")
speeds_Y = np.load("speeds.npy")


ds = tf.keras.preprocessing.timeseries_dataset_from_array(
    data=frames_X,
    targets=speeds_Y,
    sequence_length=ct.TIMESTEPS,
    sampling_rate=1,
    sequence_stride=1,
    shuffle=True,
    batch_size=32
)
# ds.shuffle(len(frames))
validation_set_size = round(ds.__len__().numpy() * 0.1)
validation_ds = ds.take(validation_set_size)
train_ds = ds.skip(validation_set_size)

# TODO: work for timeseries?
def augmentation(inputs):
    # flip over vert 
    x = RandomFlip(mode="horizontal")(inputs)
    # jitter
    x = Resizing(ct.IMG_SIZE[0]*1.2, ct.IMG_SIZE[1]*1.2)(x)
    x = RandomCrop(ct.IMG_SIZE[0], ct.IMG_SIZE[1])(x)
    x = RandomTranslation(height_factor=0.2, width_factor=0.2, fill_mode="wrap")(x)
    # contrast
    x = RandomContrast(0.1)(x)
    # zoom
    x = RandomZoom(0.1)(x)
    # rotation
    x = RandomRotation(0.05)(x)
    # brightness
    x = RandomBrightness(0.1)(x)
    return x

def model():
    inputs = tf.keras.Input((ct.TIMESTEPS,) + (ct.IMG_SIZE))
    x = Conv3D(64, (3, 3, 3),  activation='relu', padding='same')(inputs)
    x = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), padding='valid')(x)

    x = Conv3D(128, (3, 3, 3),  activation='relu', padding='same')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid')(x)

    x = Conv3D(256, (3, 3, 3),  activation='relu', padding='same')(x)
    x = Conv3D(256, (3, 3, 3),  activation='relu', padding='same')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid')(x)

    x = Conv3D(512, (5, 5, 5),  activation='relu', padding='same')(x)
    x = Conv3D(512, (5, 5, 5),  activation='relu', padding='same')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid')(x)

    x = Conv3D(512, (5, 5, 5),  activation='relu', padding='same')(x)
    x = Conv3D(512, (5, 5, 5),  activation='relu', padding='same')(x)

    # TODO: Need casual padding?
    x = Flatten()(x)
    
    x = Dense(1024)(x)
    x = Dropout(0.5)(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)

    x = Dense(256)(x)
    x = Dropout(0.5)(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)

    x = Dense(1)(x)
    x = Dropout(0.5)(x)
    outputs = ReLU()(x)

    return tf.keras.Model(inputs, outputs, name="Decoder")

c3d_model = model()
sgd = tf.keras.optimizers.SGD(lr=1e-5, decay=0.0005, momentum=0.9)

c3d_model.compile(
    optimizer=sgd,
    loss="mse")

print(c3d_model.summary())

tf.keras.utils.plot_model(
    c3d_model,
    to_file="c3d_model.png",
    show_shapes=True,
    expand_nested=True
)

# tf.keras.utils.plot_model(
#     lrcn_model,
#     to_file="architecture_model.png",
#     show_shapes=True)
#
# earlystopping_callback = tf.keras.callbacks.EarlyStopping(
#     monitor='val_loss', patience=7, restore_best_weights=True)
#
# lrcn_model.fit(
#     x=train_ds,
#     epochs=30,
#     validation_data=validation_ds,

#
# lrcn_model.compile(
#     optimizer=tf.keras.optimizers.Adam(
#         learning_rate=1e-4),
#     loss="mse")
# print(lrcn_model.summary())
#
# tf.keras.utils.plot_model(
#     lrcn_model,
#     to_file="architecture_model.png",
#     show_shapes=True)
#
# earlystopping_callback = tf.keras.callbacks.EarlyStopping(
#     monitor='val_loss', patience=7, restore_best_weights=True)
#
# lrcn_model.fit(
#     x=train_ds,
#     epochs=30,
#     validation_data=validation_ds,
#     callbacks=[earlystopping_callback])
#
# lrcn_model.save(ct.model_path)
