import numpy as np
from skimage import feature
from tqdm.autonotebook import tqdm
from src.data_filt_setups.data_filt_setups import register_data_filtering_setup
from src.data_filt_setups.utils import SliceDataset


@register_data_filtering_setup()
class EdgeSparsityDetection:
    def __init__(self, file_path: str = None, dataset=None, **kwargs):
        """
        Initialize the EdgeSparsityDetection class.

        * file_path: Path to the dataset JSON file.
        * dataset: Pre-loaded dataset object (if available).
        * kwargs: Additional keyword arguments, including:
            - is_debug: Boolean indicating whether to run in debug mode.
            - max_samples: Maximum number of samples to evaluate (useful for debug or size mode).
            - edge_threshold: The threshold for edge density (used in threshold mode).
            - mode: Either 'threshold' or 'size' to determine filtering logic.
        """
        if not file_path and not dataset:
            raise ValueError("Either file_path or dataset must be provided")
        self.file_path = file_path
        self.dataset = dataset if dataset else SliceDataset(file_path)
        self.max_samples = int(kwargs.get('max_samples', 10000))
        self.edge_threshold = float(kwargs.get('edge_threshold', 0.12))  # Ensure threshold is a float
        self.is_debug = kwargs.get('is_debug', False)
        self.mode = kwargs.get('mode', 'size')  # Default mode is 'size'

    def calculate_edge_density(self, sample):
        """
        Apply Canny edge detection to calculate the edge density of an image.

        * image: 2D array representing the image.
        * return: The edge density of the image.
        """
        image = np.abs(sample[1])/sample[2]['max'] 
        # Apply Canny edge detection to the image
        edge_map = feature.canny(image)
        # Calculate edge density as the ratio of edge pixels to the total number of pixels
        edge_density = np.sum(edge_map) / edge_map.size
        # print(edge_map.size)
        return edge_density
    
    def filter_by_threshold(self):
        """
        Filter the dataset based on the edge density threshold.
        """
        filtered_samples = []
        for sample in tqdm(self.dataset, desc="Filtering sparse edge images by threshold"):
            if self.calculate_edge_density(sample)<self.edge_threshold:
                filtered_samples.append((sample[-2], sample[-1]))  # Store file name and slice number
        return filtered_samples
    
    def filter_by_size(self):
        """
        Filter the dataset by selecting the top samples with the lowest edge density, up to max_samples.
        """
        samples_with_density = []
        for sample in tqdm(self.dataset, desc="Calculating edge densities"):
            edge_density = self.calculate_edge_density(sample)
            samples_with_density.append((sample[-2], sample[-1], edge_density))  # Store file name, slice number, and density
        # Sort samples by edge density (lowest to highest)
        samples_with_density.sort(key=lambda x: x[2])

        # Calculate how many samples need to be removed to reach the target size
        num_samples_to_remove = len(samples_with_density) - self.max_samples
        if num_samples_to_remove <= 0:
            # If dataset already has fewer or exactly max_samples, no need to filter
            print("No samples need to be removed; dataset size is already less than or equal to max_samples.")
            return []
        # Select the samples to remove (those with the lowest edge density)
        filtered_samples = [(fname, slice_num) for fname, slice_num, _ in samples_with_density[:num_samples_to_remove]]

        return filtered_samples

    def filter(self):
        """
        Perform edge sparsity detection over the entire dataset.

        * return: A list of filtered samples as (filename, slice_num) tuples in non-debug mode.
                  In debug mode, return a list of samples and their edge densities.
        """
        if self.is_debug:
            density_record = []
            samples_record = []
            count = 0
            for sample in tqdm(self.dataset, desc="Detecting edge sparsity"):
                density_record.append(self.calculate_edge_density(sample))
                samples_record.append((sample[-2], sample[-1]))  # Storing file name and slice number
                count+=1
                if count==10000:
                    break
            return samples_record, density_record
        else:
            if self.mode == "threshold":
                return self.filter_by_threshold()
            elif self.mode == "size":
                return self.filter_by_size()
            else:
                raise ValueError(f"Unknown mode: {self.mode}")
