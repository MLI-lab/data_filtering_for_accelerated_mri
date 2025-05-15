import numpy as np
from scipy.ndimage import laplace
from tqdm.autonotebook import tqdm
from src.data_filt_setups.data_filt_setups import register_data_filtering_setup
from src.data_filt_setups.utils import SliceDataset


@register_data_filtering_setup()
class LaplacianBlurDetection:
    def __init__(self, file_path: str = None, dataset=None, **kwargs):
        """
        Initialize the LaplacianBlurDetection class.

        * file_path: Path to the dataset JSON file.
        * dataset: Pre-loaded dataset object (if available).
        * kwargs: Additional keyword arguments, including:
            - is_debug: Boolean indicating whether to run in debug mode.
            - max_samples: Maximum number of samples to evaluate (useful for debug or size mode).
            - threshold: The threshold of the Laplacian variance for blur detection.
            - mode: Either 'size' or 'threshold' to determine filtering logic.
        """
        if not file_path and not dataset:
            raise ValueError("Either file_path or dataset must be provided")
        self.file_path = file_path
        self.dataset = dataset if dataset else SliceDataset(file_path)
        self.max_samples = int(kwargs.get('max_samples', 10000))
        self.threshold = float(kwargs.get('threshold', 8e-5))  # Default threshold for blur detection
        self.is_debug = kwargs.get('is_debug', False)
        self.mode = kwargs.get('mode', 'size')  # Default mode is 'size'

    def calculate_laplacian_variance(self, sample):
        """
        Calculate the variance of the Laplacian of the image from a given sample.

        * sample: The sample containing image data to be evaluated.
        * return: The variance of the Laplacian of the image.
        """
        image = sample[1]  # Assuming the image data is at index 1
        image = (image - image.mean()) / (image.std() + 1e-20)  # Normalize the image
        laplacian = laplace(image)
        variance = laplacian.var()
        return variance

    def filter_by_threshold(self):
        """
        Filter the dataset by removing samples where the Laplacian variance is below the threshold.
        """
        filtered_samples = []
        for sample in tqdm(self.dataset, desc="Filtering blurred samples by threshold"):
            laplacian_variance = self.calculate_laplacian_variance(sample)
            if laplacian_variance < self.threshold:
                filtered_samples.append((sample[-2], sample[-1]))  # Store file name and slice number
        return filtered_samples

    def filter_by_size(self):
        """
        Filter the dataset by selecting the top samples with the lowest Laplacian variance, up to max_samples.
        """
        samples_with_variance = []
        for sample in tqdm(self.dataset, desc="Calculating Laplacian variances"):
            laplacian_variance = self.calculate_laplacian_variance(sample)
            samples_with_variance.append((sample[-2], sample[-1], laplacian_variance))  # Store file name, slice number, and variance
        # Sort samples by Laplacian variance (lowest to highest)
        samples_with_variance.sort(key=lambda x: x[2])
        # Calculate how many samples need to be removed to reach the target size
        num_samples_to_remove = len(samples_with_variance) - self.max_samples
        if num_samples_to_remove <= 0:
            print("No samples need to be removed; dataset size is already less than or equal to max_samples.")
            return []
        # Select the samples to remove (those with the lowest Laplacian variance)
        filtered_samples = [(fname, slice_num) for fname, slice_num, _ in samples_with_variance[:num_samples_to_remove]]
        return filtered_samples

    def filter(self):
        """
        Perform blur detection over the entire dataset.

        * return: A list of filtered samples as (filename, slice_num) tuples in non-debug mode.
                  In debug mode, return a list of samples and their Laplacian variances.
        """
        if self.is_debug:
            variance_record = []
            samples_record = []
            count = 0
            for sample in tqdm(self.dataset, desc="Calculating Laplacian variances in debug mode"):
                laplacian_variance = self.calculate_laplacian_variance(sample)
                variance_record.append(laplacian_variance)
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