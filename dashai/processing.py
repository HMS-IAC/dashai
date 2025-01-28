from dashai.segmentation import create_circular_kernel
from skimage.segmentation import clear_border
from skimage import io
from skimage.measure import label, regionprops
from skimage.restoration import estimate_sigma
from skimage.util import img_as_float, img_as_ubyte
from scipy.stats import entropy, skew, kurtosis
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern, canny

import numpy as np
import cv2 as cv

import os
from tqdm import tqdm

from typing import Optional

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




####################################################################################################
#                   IMAGE FEATURES
####################################################################################################


class Sharpness:
    # image: np.ndarray | None = None

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
    

class Noise:
    # image: np.ndarray | None = None

    # def __init__(self, image: np.ndarray) -> None:
    #     if not isinstance(image, np.ndarray):
    #         raise TypeError("Input must be Numpy array")
        
    #     self.image = image

    def set_image(self, image: np.ndarray) -> None:
        if not isinstance(image, np.ndarray):
            raise TypeError("Input must be Numpy array")
        
        self.image = image

    def noise_level_estimation(self):
        """
        Estimates the noise level in the image.
        """
        sigma_est = estimate_sigma(self.image, average_sigmas=True)

        if np.isnan(sigma_est):
            return 0

        return sigma_est
    
    def signal_to_noise_ratio(self):
        """
        Computes the SNR of the image.
        If signal_region_coords is provided, it should be a tuple: (x, y, width, height)
        defining the region containing the signal.
        """
        # Assuming the background is the darker region
        mean_signal = np.mean(self.image)
        std_noise = np.std(self.image)
        snr = mean_signal / std_noise if std_noise != 0 else 0
        
        return snr
    
    def extract_all_features(self):
        return {
            'noise_level' : self.noise_level_estimation(),
            'snr' : self.signal_to_noise_ratio()
        }
    

class IntensityFeatures:
    def __init__(self, image=None, bit_depth=None):
        """
        Initializes the IntensityFeatures.

        Parameters:
        - image (ndarray, optional): Image array.
        - bit_depth (int, optional): Bit depth of the image.
        """
        self.bit_depth = bit_depth
        if image is not None:
            self.set_image(image)

    def set_image(self, image):
        """
        Sets the image and computes necessary parameters.

        Parameters:
        - image (ndarray): Image array.
        """
        self.image = image
        if self.bit_depth is None:
            self.bit_depth = self._get_bit_depth()
        self.num_bins = 2 ** self.bit_depth
        self.normalized_image = self._normalize_image()

    def _get_bit_depth(self):
        return None
        # bit_depth = int(np.ceil(np.log2(np.max(self.image))))
        # if bit_depth<=8:
        #     return 8
        # elif bit_depth<=12:
        #     return 12
        # elif bit_depth<=16:
        #     return 16
        # else:
        #     print('The image is more than 16-bit.')
        #     return None

    def _normalize_image(self):
        """
        Normalizes the image for histogram and statistical calculations.

        Returns:
        - ndarray: Normalized image.
        """
        image = self.image.astype(np.float64)
        image /= (self.num_bins - 1)
        return img_as_float(image)

    def mean_intensity(self):
        """Calculates the mean intensity of the image."""
        return np.mean(self.image)

    def median_intensity(self):
        """Calculates the median intensity of the image."""
        return np.median(self.image)

    def std_intensity(self):
        """Calculates the standard deviation of the image intensity."""
        return np.std(self.image)

    def variance(self):
        """Calculates the variance of the image intensity."""
        return np.var(self.image)

    def min_intensity(self):
        """Finds the minimum intensity in the image."""
        return np.min(self.image)

    def max_intensity(self):
        """Finds the maximum intensity in the image."""
        return np.max(self.image)

    def dynamic_range(self):
        """Calculates the range of intensities in the image."""
        return self.max_intensity() - self.min_intensity()
    
    def dynamic_range_utilization(self):
        """Calculates the range of intensities in the image."""
        return self.dynamic_range()/(2**self.bit_depth - 1)

    def histogram(self):
        """Calculates the histogram of the image."""
        hist, _ = np.histogram(self.normalized_image.flatten(), bins=self.num_bins, range=(0, 1))
        return hist

    def entropy(self):
        """Calculates the entropy of the image histogram."""
        hist = self.histogram()
        hist = hist / np.sum(hist)  # Normalize histogram to probabilities
        return entropy(hist)

    def skewness(self):
        """Calculates the skewness of the image intensity distribution."""
        sk = skew(self.normalized_image.flatten())
        if np.isnan(sk):
            return 0

        return sk

    def kurtosis(self):
        """Calculates the kurtosis of the image intensity distribution."""
        kurt =  kurtosis(self.normalized_image.flatten())
        if np.isnan(kurt):
            return 0
    
        return kurt

    def extract_all_features(self):
        """
        Extracts all intensity-based features from the image.

        Returns:
        - dict: A dictionary containing all extracted features.
        """
        return {
            'mean_intensity': self.mean_intensity(),
            'median_intensity': self.median_intensity(),
            'std_intensity': self.std_intensity(),
            'variance': self.variance(),
            'min_intensity': self.min_intensity(),
            'max_intensity': self.max_intensity(),
            'dynamic_range': self.dynamic_range(),
            'dynamic_range_utilization': self.dynamic_range_utilization(),
            'bit_depth': self.bit_depth,
            'histogram': self.histogram(),
            'entropy': self.entropy(),
            'skewness': self.skewness(),
            'kurtosis': self.kurtosis(),
        }
    

class TextureFeatures:
    def __init__(self):
        self.image = None

    def set_image(self, image):
        """
        Sets the image for feature extraction.

        Parameters:
        - image (ndarray): The input image.
        """
        self.image = image

    def _img_to_uint8(self, image):
        """
        Converts an image to uint8 format, adjusting for bit depth.

        Parameters:
        - image (ndarray): The input image.

        Returns:
        - ndarray: The image converted to uint8 format.
        """
        # Normalize the image to the range 0-255 and convert to uint8
        image = image - image.min()
        if image.max() > 0:
            image = (image / image.max()) * 255
        return image.astype(np.uint8)

    def glcm_features(self, distances=None, angles=None):
        """
        Extracts texture-based features from the image using GLCM.

        Parameters:
        - distances (list of int, optional): Distances for GLCM computation.
        - angles (list of float, optional): Angles for GLCM computation.

        Returns:
        - dict: A dictionary containing the extracted texture features.
        """
        if self.image is None:
            raise ValueError("Image not set. Use set_image method to set the image.")

        # Define default distances and angles if not provided
        if distances is None:
            distances = [1, 2, 4, 8]
        if angles is None:
            angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

        # Convert image to uint8
        image_uint8 = self._img_to_uint8(self.image)

        # Compute GLCM
        glcm = graycomatrix(
            image_uint8,
            distances=distances,
            angles=angles,
            symmetric=True,
            normed=True
        )

        # Extract texture features
        features = {
            'contrast': graycoprops(glcm, 'contrast').mean(),
            'dissimilarity': graycoprops(glcm, 'dissimilarity').mean(),
            'homogeneity': graycoprops(glcm, 'homogeneity').mean(),
            'energy': graycoprops(glcm, 'energy').mean(),
            'correlation': graycoprops(glcm, 'correlation').mean(),
            'ASM': graycoprops(glcm, 'ASM').mean(),
        }

        return features

    def lbp_features(self, radius=1, n_points=8):
        """
        Extracts texture-based features from the image using Local Binary Patterns (LBP).

        Parameters:
        - radius (int, optional): The radius of the circle. Default is 1.
        - n_points (int, optional): Number of points to consider for LBP. Default is 8.

        Returns:
        - dict: A dictionary containing the extracted LBP features.
        """
        if self.image is None:
            raise ValueError("Image not set. Use set_image method to set the image.")

        # Compute LBP
        lbp = local_binary_pattern(
            self.image,
            n_points,
            radius,
            method='uniform'
        )
        
        # Fixed number of bins for uniform LBP
        n_bins = n_points + 2  # +2 accounts for uniform and non-uniform patterns
        
        lbp_hist, _ = np.histogram(
            lbp,
            bins=n_bins,
            range=(0, n_bins),
            density=True
        )
        
        # Ensure the histogram length is always n_bins, even if some bins have 0 values
        features = {f'lbp_bin_{i}': lbp_hist[i] if i < len(lbp_hist) else 0 for i in range(n_bins)}

        return features

    
    def extract_all_features(self):
        """
        Extracts all texture features from the image by combining GLCM and LBP features.

        Returns:
        - dict: A dictionary containing all the extracted texture features.
        """
        if self.image is None:
            raise ValueError("Image not set. Use set_image method to set the image.")

        # Extract GLCM features
        glcm_feats = self.glcm_features()

        # Extract LBP features
        lbp_feats = self.lbp_features()

        # Combine the features
        all_features = {**glcm_feats, **lbp_feats}

        return all_features
    

class ExtractImageFeatures:
    def __init__(self):
        self.image = None

    def set_image(self, image):
        """
        Sets the image for feature extraction.

        Parameters:
        - image (ndarray): The input image.
        """
        self.image = image

    def extract_all_features(self):
        sharpness = Sharpness()
        sharpness.set_image(image=self.image)
        sharpness_features = sharpness.extract_all_features()

        noise = Noise()
        noise.set_image(image=self.image)
        noise_features = noise.extract_all_features()

        intensity = IntensityFeatures(bit_depth=16)
        intensity.set_image(image=self.image)
        intensity_features = intensity.extract_all_features()

        texture = TextureFeatures()
        texture.set_image(image=self.image)
        texture_features = texture.extract_all_features()

        all_features = {**sharpness_features, **noise_features, **intensity_features, **texture_features}

        return all_features
