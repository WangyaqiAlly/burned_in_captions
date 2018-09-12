import cv2
import numpy as np
from random import randint

def random_HLS(images, low_H=-20, high_H=20, low_L=-20, high_L=20, low_S=-20, high_S=20):
    if len(images.shape) == 3:
        delta_H = randint(low_H, high_H)
        delta_L = randint(low_L, high_L)
        delta_S = randint(low_S, high_S)
        return random_HLS_single(images, delta_H, delta_L, delta_S)
    elif len(images.shape) == 4:
        result = np.empty_like(images)
        for i, image in enumerate(images):
            delta_H = randint(low_H, high_H)
            delta_L = randint(low_L, high_L)
            delta_S = randint(low_S, high_S)
            result[i] = random_HLS_single(image, delta_H, delta_L, delta_S)
        return np.array(result)

def random_HLS_single(image, delta_H, delta_L, delta_S):
    img = cv2.cvtColor(image.astype('uint8'), cv2.COLOR_RGB2HLS).astype('uint8')
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i,j,0] = max(0, min(180, img[i,j,0]+delta_H)) # Hue
            img[i,j,1] = max(0, min(255, img[i,j,1]+delta_L)) # Light = Brightness
            img[i,j,2] = max(0, min(255, img[i,j,2]+delta_S)) # Saturation
    img = cv2.cvtColor(img, cv2.COLOR_HLS2RGB)
    return img

def random_flip(images):
    if len(images.shape) == 3:
        return random_flip_single(images)
    elif len(images.shape) == 4:
        print 'never choose this'
        result = np.empty_like(images) 
        for i,image in enumerate(images):
            result[i] = random_flip_single(image)
        return np.array(result)

def random_flip_single(image):
    flip = randint(0,1)
    if flip == 1:
        n = image.shape[0]
        m = image.shape[1]
        for i in range(n):
            for j in range(m/2):
                temp = image[i,j].copy()
                image[i,j] = image[i,(m-j-1)].copy()
                image[i,(m-j-1)] = temp
    return image

def random_crop(images, size_x=24, size_y=24):
    if len(images.shape) == 3:
        return random_crop_single(images, size_x, size_y)
    elif len(images.shape) == 4:
        print 'never choose this'
        result = np.zeros((images.shape[0], size_x, size_y, 3), dtype=images.dtype)
        for i,image in enumerate(images):
            result[i] = random_crop_single(image, size_x, size_y)
        return np.array(result)

def random_crop_single(image, size_x, size_y):
    origin_n = image.shape[0]
    origin_m = image.shape[1]
    sx = randint(0, origin_n - size_x)
    sy = randint(0, origin_m - size_y)
    return image[sx:sx+size_x,sy:sy+size_y]

