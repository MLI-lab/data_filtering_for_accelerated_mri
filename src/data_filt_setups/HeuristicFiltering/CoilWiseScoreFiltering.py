import numpy as np
from skimage import feature
from math import sqrt
from tqdm.autonotebook import tqdm
from src.data_filt_setups.data_filt_setups import register_data_filtering_setup
from src.data_filt_setups.utils import SliceDataset


@register_data_filtering_setup()
class CoilWiseScoreFiltering:
    def __init__(self, file_path: str = None, dataset=None, **kwargs):
        """
        Initialize the CoilWiseScoreFiltering class.

        * file_path: Path to the dataset JSON file.
        * dataset: Pre-loaded dataset object (if available).
        * kwargs: Additional keyword arguments, including:
            - is_debug: Boolean indicating whether to run in debug mode.
            - max_samples: Maximum number of samples to evaluate (useful for debug or size mode).
            - threshold: The threshold for the chosen score to classify data.
            - score_type: The type of score to calculate (e.g., 'edge_density', 'high_frequency').
            - cutoff_frequency: The cutoff frequency for high-pass filtering (if high_frequency is used).
            - mode: Either 'size' or 'threshold' to determine filtering logic.
        """
        if not file_path and not dataset:
            raise ValueError("Either file_path or dataset must be provided")
        self.file_path = file_path
        self.dataset = dataset if dataset else SliceDataset(file_path)
        self.max_samples = int(kwargs.get('max_samples', 10000))
        self.threshold = float(kwargs.get('threshold', 0.1))  # Default threshold
        self.is_debug = kwargs.get('is_debug', False)
        self.mode = kwargs.get('mode', 'size')  # Default mode is 'size'
        self.score_type = kwargs.get('score_type', 'high_frequency')  # Default score function is 'edge_density'
        self.cutoff_frequency = float(kwargs.get('cutoff_frequency', 30.0))  # Default cutoff frequency for high-pass filter

        # Cached high-pass filter and its dimensions
        self.high_pass_filter = None
        self.cached_shape = None

    def compute_high_pass_filter(self, shape):
        """
        Compute the Ideal High-Pass Filter (IHPF) for a given shape.

        * shape: Shape of the k-space data (H, W).
        * return: High-pass filter matrix.
        """
        H, W = shape
        center_u, center_v = H // 2, W // 2

        # Generate meshgrid for distances from the center
        u = np.arange(H)[:, None]  # column vector for rows
        v = np.arange(W)[None, :]  # row vector for columns
        D_uv = np.sqrt((u - center_u)**2 + (v - center_v)**2)

        # Create High-Pass Filter (0 inside cutoff, 1 outside)
        IHPF = np.ones((H, W))
        IHPF[D_uv <= self.cutoff_frequency] = 0

        return IHPF

    def apply_high_pass_filter(self, kspace):
        """
        Apply the High-Pass Filter to the k-space data, dynamically handling size changes.

        * kspace: 2D array representing the k-space data.
        * return: Filtered k-space after applying the high-pass filter.
        """
        # Check if the kspace shape has changed and update the filter if needed
        if self.high_pass_filter is None or kspace.shape != self.cached_shape:
            self.high_pass_filter = self.compute_high_pass_filter(kspace.shape)
            self.cached_shape = kspace.shape  # Cache the current shape

        # Apply high-pass filter (element-wise multiplication)
        ksp_filtered = kspace * self.high_pass_filter

        return ksp_filtered

    def calculate_high_frequency_ratio(self, kspace):
        """
        Calculate the high-frequency energy ratio for a given k-space data.

        * kspace: The k-space data of the coil.
        * return: The high-frequency ratio of the sample.
        """
        # Apply high-pass filter
        ksp_filtered = self.apply_high_pass_filter(kspace)

        # Calculate high-frequency ratio
        numerator = np.sum(np.abs(ksp_filtered) ** 2)
        denominator = np.sum(np.abs(kspace) ** 2)

        return numerator / denominator

    def calculate_edge_density(self, image):
        """
        Calculate the edge density of an image using Canny edge detection.

        * image: 2D array representing the image.
        * return: The edge density of the image.
        """
        edge_map = feature.canny(image)
        edge_density = np.sum(edge_map) / edge_map.size
        return edge_density

    def calculate_coil_wise_score(self, sample):
        """
        Calculate the score of the image for each coil based on the selected scoring method.

        * sample: 3D array representing k-space data for each coil (sample[0]).
        * return: Weighted score based on each coil's energy.
        """
        kspace_data = sample[0]  # k-space data of shape (#coil, height, width)
        num_coils = kspace_data.shape[0]
        coil_images = []

        # Step 1: Perform IFFT on each coil to get the image domain data
        for i in range(num_coils):
            image = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace_data[i])))
            coil_images.append(np.abs(image))  # Save the absolute value of the image
        
        # Step 2: Find the maximum pixel value across all coil images
        max_pixel_value = np.max([np.max(image) for image in coil_images])

        # Step 3: Normalize each coil image by the global max pixel value
        if max_pixel_value > 0:
            coil_images = [image / max_pixel_value for image in coil_images]

        # Step 4: Calculate the score for each coil, and accumulate a weighted score
        weighted_score = 0  # Initialize weighted sum of score
        for i in range(num_coils):
            # Choose scoring method based on score_type
            if self.score_type == 'edge_density':
                score = self.calculate_edge_density(coil_images[i])
            elif self.score_type == 'high_frequency':
                score = self.calculate_high_frequency_ratio(kspace_data[i])
            else:
                raise ValueError(f"Unknown score type: {self.score_type}")

            # Compute coil energy (sum of the absolute values of the k-space data for coil i)
            coil_energy = np.sum(np.abs(kspace_data[i])**2)

            # Accumulate weighted score (weighted by coil energy)
            weighted_score += coil_energy * score

        # Normalize the score by the total coil energy
        total_energy = np.sum(np.abs(kspace_data)**2)
        normalized_score = weighted_score / total_energy

        return normalized_score

    def filter_by_threshold(self):
        """
        Filter the dataset by removing samples where the score is below the threshold.
        """
        filtered_samples = []
        for sample in tqdm(self.dataset, desc="Filtering samples by threshold"):
            score = self.calculate_coil_wise_score(sample)

            if score < self.threshold:
                filtered_samples.append((sample[-2], sample[-1]))  # Store file name and slice number
        return filtered_samples

    def filter_by_size(self):
        """
        Filter the dataset by selecting the top samples with the lowest score, up to max_samples.
        """
        samples_with_score = []
        for sample in tqdm(self.dataset, desc="Calculating scores"):
            score = self.calculate_coil_wise_score(sample)
            samples_with_score.append((sample[-2], sample[-1], score))  # Store file name, slice number, and score
        
        # Sort samples by score (lowest to highest)
        samples_with_score.sort(key=lambda x: x[2])

        # Calculate how many samples need to be removed to reach the target size
        num_samples_to_remove = len(samples_with_score) - self.max_samples
        if num_samples_to_remove <= 0:
            print("No samples need to be removed; dataset size is already less than or equal to max_samples.")
            return []

        # Select the samples to remove (those with the lowest score)
        filtered_samples = [(fname, slice_num) for fname, slice_num, _ in samples_with_score[:num_samples_to_remove]]

        return filtered_samples

    def filter(self):
        """
        Perform coil-wise score-based filtering over the entire dataset.

        * return: A list of filtered samples as (filename, slice_num) tuples in non-debug mode.
                  In debug mode, return a list of samples and their scores.
        """
        if self.is_debug:
            score_record = []
            samples_record = []
            count = 0
            for sample in tqdm(self.dataset, desc="Calculating scores in debug mode"):
                score_record.append(self.calculate_coil_wise_score(sample))
                samples_record.append((sample[-2], sample[-1]))  # Store file name and slice number
                count += 1
                if count == 3000:
                    break
            return samples_record, score_record
        else:
            if self.mode == "threshold":
                return self.filter_by_threshold()
            elif self.mode == "size":
                return self.filter_by_size()
           