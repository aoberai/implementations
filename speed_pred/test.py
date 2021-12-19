import tensorflow as tf


model = tf.keras.models.load_model("tmp/checkpoint/")
model.save("c3d.h5")
