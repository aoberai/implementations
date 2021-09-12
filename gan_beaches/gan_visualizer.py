
import tensorflow as tf
import cv2
import numpy as np

wants_animation = True
generator = tf.keras.models.load_model("models/generator.h5", compile=False)
noise_dim = 100

if wants_animation:
    animation_speed = 0.1
    starting_noise = (tf.random.normal([noise_dim]).numpy())
    for i in range(len(starting_noise)):
        print("On noise index:", i)
        noise_modifier = [c for c in starting_noise]
        for j in range(int(-2.5/animation_speed), int(2.5/animation_speed), 1): # creates variability on specific var ranging from 2.5 std below to 2.5 above mean of N(0, 1) 
            noise_modifier[i] = j * animation_speed
            cv2.imshow("GAN Animation", generator.predict(np.expand_dims(noise_modifier, 0))[0,:,:,:] * 255)
            cv2.waitKey(1)

else:
    counter = 0
    while True:
        noise = np.expand_dims(tf.random.normal([noise_dim]).numpy(), 0)
        generated_image = generator.predict(noise)[0,:,:,:] * 255
        cv2.imshow("Generated Image", generated_image)
        cv2.waitKey(1)
        if counter != 0:
            input() # wait till enter pressed to load next image
        counter += 1



