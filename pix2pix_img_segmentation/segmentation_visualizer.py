import cv2
import os
import numpy as np
import tensorflow as tf

generator_model = tf.keras.models.load_model("GeneratorEpoch9.h5")
image_shape = (256, 256, 3)

test_x_path = "/home/aoberai/programming/ml-datasets/comma10k/imgs2/"
test_x_img_paths = [os.path.join(test_x_path, img_name) for img_name in os.listdir(test_x_path)]

test_y_path = "/home/aoberai/programming/ml-datasets/comma10k/masks2/"
test_y_img_paths = [os.path.join(test_y_path, img_name) for img_name in os.listdir(test_y_path)]

# shuffle
seed = np.random.randint(100)
np.random.seed(seed)
np.random.shuffle(test_x_img_paths)
np.random.seed(seed)
np.random.shuffle(test_y_img_paths)

for i in range(len(test_x_img_paths)):
    x_img = cv2.imread(test_x_img_paths[i])
    x_scaling_factor = image_shape[0]/np.shape(x_img)[0]
    x_img = tf.image.random_crop(value=cv2.resize(x_img, None, fx=x_scaling_factor, fy=x_scaling_factor), size=image_shape).numpy()
    gen_img = (generator_model.predict(np.expand_dims(x_img, 0))[0]).astype(np.uint8)
    print(gen_img)
    cv2.imshow("Orig", x_img)
    cv2.imshow("Gen", gen_img)
    cv2.waitKey(1000)


