import tensorflow as tf
import numpy as np
import random
from skimage import io
import cv2
import matplotlib.pyplot as plt


def random_crop_resize(image, min_ratio=0.4, max_ratio=0.99):
    h, w = image.shape[0: 2]
    ratio = random.random()
    scale = min_ratio + ratio * (max_ratio - min_ratio)
    new_h = int(h * scale)
    new_w = int(w * scale)
    y = int(np.random.randint(0, h - new_h) / 5 * 2)
    length = w - new_w
    ancho1 = int(length / 4)
    ancho2 = int(length / 4 * 3)
    prob = random.random()
    if prob <= 0.5:
        x = np.random.randint(0, ancho1)
    else:
        x = np.random.randint(ancho2, length)
    image = image[y:y+new_h, x:x+new_w, :]
    image = cv2.resize(image, (h,w))

    return image


def color_distortion(image, s=1.0):
    # image is a tensor with value range in [0, 1].
    # s is the strength of color distortion.
    prob = random.random()
    if prob <= 0.8:
        x = tf.image.random_brightness(image, max_delta=0.5*s)
        x = tf.image.random_contrast(x, lower=1-0.8*s, upper=1+0.8*s)
        x = tf.image.random_saturation(x, lower=1-0.8*s, upper=1+0.8*s)
        x = tf.image.random_hue(x, max_delta=0.2*s)
        x = tf.clip_by_value(x, 0, 1)
    else:
        x = tf.image.rgb_to_grayscale(image)
        x = tf.tile(x, [1, 1, 3])

    return x.numpy()

def blur_or_noise(image):
    prob = random.random()
    if prob <= 0.5:
        sigma = random.random() * 1.9 + 0.1
        image = cv2.GaussianBlur(image, (5,5), sigma)
    else:
        image += np.random.normal(0., 0.1, (image.shape[0], image.shape[1], image.shape[2]))
        image = np.clip(image, 0, 1)
    return image


def apply_transformation(image, trans_num):
    if trans_num == 1:
        x = color_distortion(image)
    elif trans_num == 2:
        x = random_crop_resize(image)
    elif trans_num == 3:
        x = blur_or_noise(image)
    else:
        x = image

    return x

if __name__ == '__main__':
    jpeg_file = '/home/lingyu/CLAE_Advdrop/Personalize/subject0102/1.jpg'
    img = io.imread(jpeg_file)/255.
    new = cv2.GaussianBlur(img, (5,5), 0.5)
    #new = color_distortion(img)
    plt.imshow(new)
    plt.show()
    a = 5