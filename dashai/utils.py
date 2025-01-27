import os
from tqdm import tqdm
from typing import Optional

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