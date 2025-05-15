import numpy as np
from scipy.ndimage import sobel
from tqdm.autonotebook import tqdm
from src.data_filt_setups.data_filt_setups import register_data_filtering_setup
from src.data_filt_setups.utils import SliceDataset


@register_data_filtering_setup()
class SobelFlatRegionDetection:
    def __init__(self, file_path: str = None, dataset=None, **kwargs):
        """
        Initialize the SobelFlatRegionDetection class.

        * file_path: Path to the dataset JSON file.
        * dataset: Pre-loaded dataset object (if available).
        * kwargs: Additional keyword arguments, including:
            - is_debug: Boolean indicating whether to run in debug mode.
            - max_samples: Maximum number of samples to evaluate (useful for debug or size mode).
            - threshold: The threshold for the gradient variance to classify flat regions.
            - mode: Either 'size' or 'threshold' to determine filtering logic.
        """
        if not file_path and not dataset:
            raise ValueError("Either file_path or dataset must be provided")
        self.file_path = file_path
        self.dataset = dataset if dataset else SliceDataset(file_path)
        self.max_samples = int(kwargs.get('max_samples', 10000))
        self.threshold = float(kwargs.get('threshold', 0.07))  # Default threshold for flat region detection
        self.is_debug = kwargs.get('is_debug', False)
        self.mode = kwargs.get('mode', 'size')  # Default mode is 'size'

    def calculate_gradient_variance(self, sample):
        """
        Calculate the gradient variance of a sample using the Sobel filter.

        * sample: The sample containing image data.
        * return: The variance of the gradient magnitude.
        """
        image = np.abs(sample[1]) / sample[2]['max']  # Assuming the image is at index 1 and metadata at index 2
        # Compute the gradient along x and y using Sobel filter
        sobel_x = sobel(image, axis=0)  # Gradient along x-direction
        sobel_y = sobel(image, axis=1)  # Gradient along y-direction

        # Compute gradient magnitude
        gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

        # Calculate the variance of the gradient magnitude
        gradient_variance = np.var(gradient_magnitude)
        return gradient_variance
    
    def filter_by_threshold(self):
        """
        Filter the dataset by removing samples where the gradient variance is below the threshold.
        """
        filtered_samples = []
        for sample in tqdm(self.dataset, desc="Filtering flat regions by threshold"):
            gradient_variance = self.calculate_gradient_variance(sample)

            if gradient_variance < self.threshold:
                filtered_samples.append((sample[-2], sample[-1]))  # Store file name and slice number
        return filtered_samples

    def filter_by_size(self):
        """
        Filter the dataset by selecting the top samples with the lowest gradient variance, up to max_samples.
        """
        samples_with_variance = []
        for sample in tqdm(self.dataset, desc="Calculating gradient variances"):
            gradient_variance = self.calculate_gradient_variance(sample)
            samples_with_variance.append((sample[-2], sample[-1], gradient_variance))  # Store file name, slice number, and gradient variance
        
        # Sort samples by gradient variance (lowest to highest)
        samples_with_variance.sort(key=lambda x: x[2])

        # Calculate how many samples need to be removed to reach the target size
        num_samples_to_remove = len(samples_with_variance) - self.max_samples
        if num_samples_to_remove <= 0:
            print("No samples need to be removed; dataset size is already less than or equal to max_samples.")
            return []
        
        # Select the samples to remove (those with the lowest gradient variance)
        filtered_samples = [(fname, slice_num) for fname, slice_num, _ in samples_with_variance[:num_samples_to_remove]]

        return filtered_samples

    def filter(self):
        """
        Perform flat region detection over the entire dataset.

        * return: A list of filtered samples as (filename, slice_num) tuples in non-debug mode.
                  In debug mode, return a list of samples and their gradient variances.
        """
        if self.is_debug:
            variance_record = []
            samples_record = []
            count = 0
            for sample in tqdm(self.dataset, desc="Calculating gradient variances in debug mode"):
                gradient_variance = self.calculate_gradient_variance(sample)
                variance_record.append(gradient_variance)
                samples_record.append((sample[-2], sample[-1]))  # Store file name and slice number
                count += 1
                if count == 10000:  # Limit to 10000 samples in debug mode
                    break
            return samples_record, variance_record
        else:
            if self.mode == "threshold":
                return self.filter_by_threshold()
            elif self.mode == "size":
                return self.filter_by_size()
            else:
                raise ValueError(f"Unknown mode: {self.mode}")