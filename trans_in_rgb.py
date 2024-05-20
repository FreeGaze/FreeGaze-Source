import tensorflow as tf
import numpy as np
import random
from skimage import io
import cv2
import matplotlib.pyplot as plt


def random_crop_resize(image, left_landmark, right_landmark, min_ratio=0.4, max_ratio=0.99):
    h, w = image.shape[0: 2]
    left_landmark = np.clip(left_landmark, 1, h-1)
    right_landmark = np.clip(right_landmark, 1, w-1)
    ratio = random.random()
    scale = min_ratio + ratio * (max_ratio - min_ratio)
    new_h = int(h * scale)
    new_w = int(w * scale)

    length_w = w - new_w
    length_h = h - new_h
    prob = random.random()
    if prob <= 0.5:
        # using right eye landmark
        upper_x = np.minimum(length_w, right_landmark[0])
        upper_y = np.minimum(length_h, right_landmark[1])
        lower_x = np.maximum(0, right_landmark[2] - new_w)
        lower_y = np.maximum(0, right_landmark[3] - new_h)
        x = np.random.randint(lower_x, upper_x)
        y = np.random.randint(lower_y, upper_y)
    else:
        # using left eye landmark
        upper_x = np.minimum(length_w, left_landmark[0])
        upper_y = np.minimum(length_h, left_landmark[1])
        lower_x = np.maximum(0, left_landmark[2] - new_w)
        lower_y = np.maximum(0, left_landmark[3] - new_h)
        x = np.random.randint(lower_x, upper_x)
        y = np.random.randint(lower_y, upper_y)

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


def apply_transformation(image, trans_num, left_landmark, right_landmark):
    if trans_num == 1:
        x = color_distortion(image)
    elif trans_num == 2:
        x = random_crop_resize(image, left_landmark, right_landmark)
    elif trans_num == 3:
        x = blur_or_noise(image)
    else:
        x = image

    return x
