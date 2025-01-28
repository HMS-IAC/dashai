#####################################################################################
#               MODIFY SYSTEM PATH
#####################################################################################
import sys
import os

# Add the root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
#####################################################################################

# import
import os
from dashai.processing import ExtractImageFeatures
from dashai.utils import write_image_info_to_csv
from tqdm import tqdm
import tifffile

import string
import pandas as pd

# user defined path for input data
PATH = "_CLEANED_DATA/magnification/10x/2024-11-20"

# check image features
folders = os.listdir(f'{PATH}')
for folder in folders:
    print(f'Processing: {folder}')
    files = os.listdir(f'{PATH}/{folder}')
    # files = [file for file in files if file.split('.')[0].endswith('w1')]
    for file in tqdm(files):
        image_info = {
            'file_path': f'{PATH}/{folder}',
            'file_name': file,
            'row_alphabet': file.split("_")[1][0],
            'col_number': int(file.split("_")[1][1:]),
            'site': int(file.split("_")[2][1:]),
            'wavelength': int(file.split("_")[3].split('.')[0][1:])
            # 'image_dimension'
        }
        image = tifffile.imread(f'{PATH}/{folder}/{file}')
        image_features = ExtractImageFeatures()
        image_features.set_image(image)
        features = image_features.extract_all_features()

        write_image_info_to_csv(image_info | features, folder_path=PATH, csv_filename=f'{folder}.csv')



# nucleus count