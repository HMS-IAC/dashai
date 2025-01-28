#####################################################################################
#               MODIFY SYSTEM PATH
#####################################################################################
import sys
import os

# Add the root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
#####################################################################################

# imports
from dashai.utils import check_directory_structure, remove_thumb_and_rename_files

# Check directory structure
# path = "../_RAW_DATA"

# check_directory_structure(folder_path=path)

#####################################################################################

# Cleanup directory
input_directory = "../_RAW_DATA/Comparison of Different Magnifications Nov 18 2024/20xYoungOldMixed/2/2/2024-11-20/19944/TimePoint_1"

output_directory = "../_CLEANED_DATA/magnification/20x/2024-11-20/plate_2"

remove_thumb_and_rename_files(old_path=input_directory, new_path=output_directory)

#####################################################################################

