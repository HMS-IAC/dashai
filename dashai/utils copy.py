# def read_image (read a single channel 16-bit image)
# def detect_centroid (calculate centroid)
# def crop_image (image and bounding box infor as input and return cropped image)

import cv2 as cv
from tifffile import imread     # pip install tifffile
import numpy as np
import os


def read_image(file_path: str, file_name: str):
    full_name = f'{file_path}/{file_name}'
#     print(f'Reading file: {file_name.split(".")[0]}')

    image = imread(files=full_name)

    return image


def combine_channels(channel1, channel2, channel3):
    return np.stack((channel1, channel2, channel3), axis=-1)



def gray_to_rgb(image):
    """
    Convert a grayscale image to an RGB image.

    Parameters:
    - image (numpy.ndarray): Input grayscale image.

    Returns:
    - numpy.ndarray: RGB image.
    """
    if len(image.shape) == 2:
        rgb_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
        for i in range(3):
            rgb_image[:, :, i] = image
    else:
        rgb_image = image

    return rgb_image



def image_scaling(image):
    image = image.astype('float32')
    return ((image-image.min())/(image.max()-image.min()))











from scipy.ndimage import rotate, affine_transform
import random

def random_rotation(image_array, max_angle=25):
    """Randomly rotate the image within the given angle range."""
    angle = np.random.uniform(-max_angle, max_angle)
    return rotate(image_array, angle, reshape=False, mode='nearest')

def random_shift(image_array, max_shift=0.2):
    """Randomly shift the image horizontally and vertically."""
    h, w = image_array.shape[:2]
    dx = np.random.uniform(-max_shift, max_shift) * w
    dy = np.random.uniform(-max_shift, max_shift) * h
    shift_matrix = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]])
    return affine_transform(image_array, shift_matrix[:2, :2], offset=[dx, dy], mode='nearest')

def random_shear(image_array, max_shear=20):
    """Randomly shear the image."""
    shear = np.deg2rad(np.random.uniform(-max_shear, max_shear))
    shear_matrix = np.array([[1, -np.sin(shear), 0], [0, np.cos(shear), 0], [0, 0, 1]])
    return affine_transform(image_array, shear_matrix[:2, :2], offset=[-np.sin(shear) * image_array.shape[0], 0], mode='nearest')

def random_zoom(image_array, min_zoom=0.8, max_zoom=1.2):
    """Randomly zoom in/out on the image."""
    zx, zy = np.random.uniform(min_zoom, max_zoom), np.random.uniform(min_zoom, max_zoom)
    zoom_matrix = np.array([[zx, 0, 0], [0, zy, 0], [0, 0, 1]])
    return affine_transform(image_array, zoom_matrix[:2, :2], mode='nearest')

def random_flip(image_array):
    """Randomly flip the image horizontally or vertically."""
    if random.choice([True, False]):
        # Horizontal flip
        image_array = np.fliplr(image_array)
    if random.choice([True, False]):
        # Vertical flip
        image_array = np.flipud(image_array)
    return image_array

def custom_data_augmentation(image_path):
    """Apply a series of random transformations for data augmentation."""
    image = Image.open(image_path)
    image_array = np.array(image)

    if random.choice([True, False]):
        image_array = random_rotation(image_array)
    if random.choice([True, False]):
        image_array = random_shift(image_array)
    if random.choice([True, False]):
        image_array = random_shear(image_array)
    if random.choice([True, False]):
        image_array = random_zoom(image_array)
    if random.choice([True, False]):
        image_array = random_flip(image_array)

    return Image.fromarray(np.uint8(image_array))



def custom_image_loader(image_path, image_labels):
    images = []
    labels = []
    file_names = os.listdir(image_path)
    for file_name in file_names:
        image = imread(f'{image_path}/{file_name}')
        image = image_scaling(image)
        images.append(image)
        labels.append(image_labels)
    return images, labels