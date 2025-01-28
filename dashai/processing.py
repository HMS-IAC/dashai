from dashai.segmentation import create_circular_kernel
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops

import numpy as np
import cv2 as cv

import os
from tqdm import tqdm

def remove_border_objects(mask, px=64, same_mask=False):
    """
    Remove tubules near the image border.

    Parameters:
    - mask (numpy.ndarray): Binary mask of tubules.
    - px (int): Padding around the border to keep.
    - same_mask (bool): Whether to modify the input mask or return a new one.

    Returns:
    - numpy.ndarray: Mask with border tubules removed.
    """

    cleared = clear_border(mask[px:mask.shape[0]-px, px:mask.shape[1]-px])

    kernel = create_circular_kernel(radius=3)
    eroded = cv.erode(cleared, kernel=kernel)

    padded = np.pad(eroded, pad_width=px)

    if same_mask:
        dilated = cv.dilate(padded, kernel=kernel, iterations=3)
        return np.multiply(mask, dilated)
    else:
        return padded
    




def remove_objects_outside_iqr(binary_mask):
    # Label the objects in the binary mask
    binary_mask = binary_mask>0
    labeled_mask = label(binary_mask)
    
    # Measure the area of each object
    areas = [prop.area for prop in regionprops(labeled_mask)]
    
    # Calculate the IQR
    #Q1, Q3 = np.percentile(areas, [25, 75])
    #IQR = Q3 - Q1
    
    # Determine the lower and upper bounds for object sizes
    lower_bound = 100
    upper_bound = 750
    #print(upper_bound, IQR)
    
    # Remove objects smaller than the lower bound and larger than the upper bound
    # Note: skimage's remove_small_objects and remove_large_objects functions may not directly
    # accept a range, so you might need to apply them iteratively or customize this part.
    # The following is a simplified approach:
    for region in regionprops(labeled_mask):
        if region.area < lower_bound or region.area > upper_bound:
            # Set the pixels of the removed object to 0 (background)
            binary_mask[labeled_mask == region.label] = 0
    
#    binary_mask = remove_small_objects(binary_mask, lower_bound)
#    binary_mask = remove_large_objects(binary_mask, upper_bound)
    

    return binary_mask





def get_centroid(contour):
    """
    Calculates the centroid of a contour.

    Args:
        contour: Contour of an object.

    Returns:
        tuple: Centroid coordinates (cx, cy).
    """
    moment = cv.moments(contour)

    # Calculate centroid coordinates
    if moment['m00']!=0:
        cx = int(moment['m10'] / moment['m00'])
        cy = int(moment['m01'] / moment['m00'])
    else:
        cx, cy = 0, 0

    return cx, cy



from PIL import Image

def crop_image_around_centroid(image, x,y,w,h):
    # Open the image

    # Calculate the crop box
    # Crop the image
    cropped_image = image[x-(w//2):x+(w//2),y-(h//2):y+(h//2)]

    # Save the cropped image
    return cropped_image



def data_cleanup(path, print_lines=False):
    for file_name in tqdm(os.listdir(path)):
        if "thumb" in file_name.lower():
            file_path = os.path.join(path, file_name)
            os.remove(file_path)
            if print_lines:
                print(f'Deleted {file_name}')
        else:
            old_file_path = os.path.join(path, file_name)
            parts = file_name.split('_')
            new_name = '_'.join(parts[0:3])+'_'+parts[3][:2]+'.tif'
            new_file_path = os.path.join(path, new_name)
            os.rename(old_file_path, new_file_path)




class Sharpness:
    image: np.ndarray | None = None

    # def __init__(self, image: np.ndarray) -> None:
    #     if not isinstance(image, np.ndarray):
    #         raise TypeError("Input must be Numpy array")
        
    #     self.image = image

    def set_image(self, image: np.ndarray) -> None:
        if not isinstance(image, np.ndarray):
            raise TypeError("Input must be Numpy array")
        
        self.image = image


    def variance_of_laplacian(self):
        """
        Computes the variance of the Laplacian of the image.
        """
        laplacian = cv.Laplacian(self.image, cv.CV_64F)
        variance = laplacian.var()
        return variance

    def tenengrad(self):
        """
        Computes the Tenengrad focus measure.
        """
        # Convert to 64-bit float for precision
        image_float = self.image.astype(np.float64)
        # Sobel filters
        gx = cv.Sobel(image_float, cv.CV_64F, 1, 0, ksize=3)
        gy = cv.Sobel(image_float, cv.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(gx**2 + gy**2)
        tenengrad_value = np.mean(gradient_magnitude)
        return tenengrad_value

    def brenners_gradient(self):
        """
        Computes Brenner's gradient focus measure.
        """
        shifted_image = np.roll(self.image, -2, axis=0)
        diff = (self.image - shifted_image) ** 2
        brenner_value = np.sum(diff)
        return brenner_value
    
    def fft_sharpness(self):
        
        # Convert to float32 for precision
        img_float = np.float32(self.image)

        # Step 2: Apply Fourier Transform
        f = np.fft.fft2(img_float)

        # Step 3: Shift the zero-frequency component to the center
        fshift = np.fft.fftshift(f)

        # Compute the magnitude spectrum
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)  # Added 1 to avoid log(0)

        return np.mean(magnitude_spectrum)
    
    def extract_all_features(self):
        return {
            'laplacian' : self.variance_of_laplacian(),
            'tenengrad' : self.tenengrad(),
            'brenners_gradient' : self.brenners_gradient(),
            'fourier_magnitude' : self.fft_sharpness()
        }
    

