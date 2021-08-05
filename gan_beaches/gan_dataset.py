import cv2
import numpy as np
import os
import time
import constants

def preprocessing(image):

    def resize_image(img, size):

        h, w = img.shape[:2]
        c = img.shape[2] if len(img.shape)>2 else 1

        if h == w: 
            return cv2.resize(img, size, cv2.INTER_AREA)

        dif = h if h > w else w

        interpolation = cv2.INTER_AREA if dif > (size[0]+size[1])//2 else cv2.INTER_CUBIC

        x_pos = (dif - w)//2
        y_pos = (dif - h)//2

        if len(img.shape) == 2:
            mask = np.zeros((dif, dif), dtype=img.dtype)
            mask[y_pos:y_pos+h, x_pos:x_pos+w] = img[:h, :w]
        else:
            mask = np.zeros((dif, dif, c), dtype=img.dtype)
            mask[y_pos:y_pos+h, x_pos:x_pos+w, :] = img[:h, :w, :]

        return cv2.resize(mask, size, interpolation)


    image = resize_image(image, (constants.image_shape[0], constants.image_shape[1],))
    image = image/255.0
    return image


dataset_path = "/home/aoberai/programming/ml-datasets/beaches"
dataset_images_path = os.listdir(dataset_path)
dataset_images = []

for image_name in dataset_images_path:
    image_path = os.path.join(dataset_path, image_name)
    try:
        image = cv2.imread(image_path)
        image = preprocessing(image)
        dataset_images.append(image)
    except Exception as e:
        print(image_path, "broken")
        print(e)


    # displays images 
    '''
    cv2.imshow("Beach", image)
    cv2.waitKey(1)
    time.sleep(2)
    print(image_path)
    '''

np.random.shuffle(dataset_images)

np.save("beaches.npy", dataset_images)


print("\n\n\n\n\n\n\n\n Sample Image Pixel: \n\n\n", dataset_images[1][100][100])
