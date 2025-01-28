import os
import csv
from tqdm import tqdm
import string
import pandas as pd
from typing import Optional
import matplotlib.pyplot as plt

def remove_thumb_and_rename_files(old_path: str, new_path: str = None, print_lines: Optional[bool] = False) -> None:
    """
    Cleans up the data in the specified directory by:
    1. Deleting files that contain "thumb" in their names (case-insensitive).
    2. Renaming the remaining files based on a specific naming convention.

    Args:
        path (str): The directory path containing the files to clean up.
        print_lines (bool, optional): Whether to print the names of deleted files. Defaults to False.

    Returns:
        None
    """
    for file_name in tqdm(os.listdir(old_path), desc="Processing files"):
        # Check if "thumb" is in the filename (case-insensitive)
        if "thumb" in file_name.lower():
            # Construct the full file path
            file_path: str = os.path.join(old_path, file_name)
            # Delete the file
            os.remove(file_path)
            if print_lines:
                print(f'Deleted {file_name}')
        else:
            # Construct the full file path for renaming
            old_file_path: str = os.path.join(old_path, file_name)
            # Split the filename to create the new name
            parts: list[str] = file_name.split('_')
            # Ensure parts have enough segments to avoid index errors
            if len(parts) > 3:
                new_name: str = '_'.join(parts[0:3]) + '_' + parts[3][:2] + '.tif'
                if new_path == None:
                    new_file_path: str = os.path.join(old_path, new_name)
                else:
                    new_file_path: str = os.path.join(new_path, new_name)
                # Rename the file
                os.rename(old_file_path, new_file_path)
            else:
                print(f"Skipping file {file_name} due to unexpected format.")

            
def check_directory_structure(folder_path: str) -> None:
    """
    Traverses a folder to create a hierarchical structure.
    Stops traversal if the folder is named 'TimePoint_1' or if it contains TIFF files and no subdirectories.

    Args:
        folder_path (str): Path to the folder to traverse.

    Returns:
        None
    """
    def traverse_directory(path: str, level: int = 0):
        # Print the current directory with indentation
        print("    " * level + f"- {os.path.basename(path)}")

        # Get all entries in the directory
        entries = os.listdir(path)
        
        # Check if it's a terminal folder
        is_timepoint = os.path.basename(path).lower() == "timepoint_1"
        contains_tiff_files = any(entry.lower().endswith(".tif") for entry in entries)

        if is_timepoint or contains_tiff_files:
            return  # Stop traversal

        # Traverse subdirectories
        for entry in entries:
            entry_path = os.path.join(path, entry)
            if os.path.isdir(entry_path):
                traverse_directory(entry_path, level + 1)

    # Ensure the folder exists
    if not os.path.isdir(folder_path):
        print(f"Error: The path '{folder_path}' is not a valid directory.")
        return

    print("Folder Hierarchy:")
    traverse_directory(folder_path)



####################################################################################################################
#       OLD FUNCTIONS
####################################################################################################################

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


def _get_index(row, site_multiple):
    index=[]
    for l in list(string.ascii_uppercase[:row]):
        for _ in range(site_multiple):
            index.append(l)
    return index

def _get_columns(col, site_multiple):
    columns=[f'w{i:02}' for i in range(1,col+1) for j in range(1,site_multiple+1)]
    return columns

def create_plate(well_count, site_row, site_col):
    if well_count == 96:
        row = 8
        col = 12

    total_row = row*site_row
    total_col = col*site_col

    plate = np.zeros((total_row, total_col))

    return plate

def _get_row_col(file_name, site_row, site_col):
    p = file_name.split("_")[1][0]
    w = int(file_name.split("_")[1][1:])
    s = file_name.split("_")[2][1:]

    return (p, w, s, ((string.ascii_uppercase.index(p)*site_row) + (int(s)-1)//site_row), (((int(w)-1)*site_col) + (int(s)-1)%site_row))

def add_axis(plate, well_count, site_row, site_col, savefig=False, fig_name=None):
    if well_count == 96:
        row = 8
        col = 12
    
    column = _get_columns(col, site_col)
    index = _get_index(row, site_row)

    tab_df = pd.DataFrame(plate, index=index, columns=column)
    # tab_df.to_csv(f'{path}/nuclei_masks/cell_count.csv')
    
    
    plt.figure(figsize=(len(column)//2,len(index)//2))
    plt.imshow(plate, cmap='hot')
    plt.xticks(range(0,len(column)), column)
    plt.yticks(range(0,len(index)), index)
    if savefig and fig_name is not None:
        plt.savefig(f'{fig_name}.png', dpi=300)

def write_image_info_to_csv(image_info, folder_path, csv_filename):
    """
    Writes the image_info dictionary to a CSV file.

    Parameters:
    - image_info (dict): Dictionary containing image information.
    - folder_path (str): Path to the folder where the CSV file will be created.
    """

    # Define the default CSV filename
    # csv_filename = 'image_info.csv'
    csv_path = os.path.join(folder_path, csv_filename)

    # Check if the file exists
    # if os.path.isfile(csv_path):
    #     # # If it exists, append date and time to the filename
    #     # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #     # csv_filename = f'image_info_{timestamp}.csv'
    #     # csv_path = os.path.join(folder_path, csv_filename)

    with open(csv_path, mode='a', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=image_info.keys())
        
        # Write the header if the file is newly created
        if os.path.getsize(csv_path) == 0:
            writer.writeheader()

        # Write the image_info dictionary rows
        writer.writerow(image_info)