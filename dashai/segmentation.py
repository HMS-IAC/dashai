import numpy as np
import cv2 as cv
from dashai.utils import gray_to_rgb


def normalize_image(image):
    image = (image-image.min())/(image.max()-image.min())
    return (255*image).astype('uint8')

def get_nuclei_segmentation(image):
    # gaussian filter
    image = normalize_image(image)
    blurred = cv.GaussianBlur(image, (3,3), 0)

    # Otsu
    _, thresh = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

    # watershed
    watershed = get_watershed(thresh)

    return 255*(watershed>1).astype('uint8')








    # StarDist




def create_circular_kernel(radius):
    """
    Create a circular kernel for morphological operations.

    Parameters:
    - radius (int): Radius of the circular kernel.

    Returns:
    - numpy.ndarray: Circular kernel.
    """
    return cv.getStructuringElement(cv.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))

def image_opening(image, k):
    """
    Perform the opening operation on the input image.

    Parameters:
    - image (numpy.ndarray): Input image.
    - k (int): Radius of the circular kernel.

    Returns:
    - numpy.ndarray: Result of the opening operation.
    """
    kernel = create_circular_kernel(k)
    opened_image = cv.morphologyEx(image, cv.MORPH_ERODE, kernel, iterations=4)
    return opened_image

def get_watershed(image):
    """
    Perform image segmentation using the watershed algorithm.

    Parameters:
    - image (numpy.ndarray): Input image.

    Returns:
    - numpy.ndarray: Segmentation markers.
    """
    sure_bg = cv.dilate(image, create_circular_kernel(3))
    sure_fg = image_opening(image, 2)
    unknown = cv.subtract(sure_bg, sure_fg)
    _, markers = cv.connectedComponents(sure_fg)
    markers += 1
    markers[unknown == 255] = 0

    markers = cv.watershed(gray_to_rgb(image), markers)

    return markers
