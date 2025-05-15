import numpy as np
from math import sqrt
from tqdm.autonotebook import tqdm
from src.data_filt_setups.data_filt_setups import register_data_filtering_setup
from src.data_filt_setups.utils import SliceDataset


@register_data_filtering_setup()
class HighFrequencyFiltering:
    def __init__(self, file_path: str = None, dataset=None, **kwargs):
        """
        Initialize the HighFrequencyFiltering class.

        * file_path: Path to the dataset JSON file.
        * dataset: Pre-loaded dataset object (if available).
        * kwargs: Additional keyword arguments, including:
            - is_debug: Boolean indicating whether to run in debug mode.
            - max_samples: Maximum number of samples to evaluate (useful for debug or size mode).
            - threshold: The threshold for the high-frequency ratio to classify samples.
            - cutoff_frequency: The cutoff frequency for the high-pass filter.
            - mode: Either 'size' or 'threshold' to determine filtering logic.
        """
        if not file_path and not dataset:
            raise ValueError("Either file_path or dataset must be provided")
        self.file_path = file_path
        self.dataset = dataset if dataset else SliceDataset(file_path)
        self.max_samples = int(kwargs.get('max_samples', 10000))
        self.threshold = float(kwargs.get('threshold', 0.1))  # Default threshold for high-frequency ratio
        self.cutoff_frequency = float(kwargs.get('cutoff_frequency', 30.0))  # Default cutoff frequency for high-pass filter
        self.is_debug = kwargs.get('is_debug', False)
        self.mode = kwargs.get('mode', 'size')  # Default mode is 'size'

    def apply_high_pass_filter(self, kspace):
        """
        Apply the Discrete Fourier Transform (DFT) and Ideal High-Pass Filter (IHPF) to the k-space data.

        * kspace: 2D array representing the k-space data.
        * return: Filtered k-space after applying the high-pass filter.
        """
        H, W = kspace.shape[-2], kspace.shape[-1]
        center_u, center_v = H // 2, W // 2

        # Initialize the high-pass filter
        IHPF = np.ones((H, W))
        for u in range(H):
            for v in range(W):
                D_uv = sqrt((u - center_u) ** 2 + (v - center_v) ** 2)
                if D_uv <= self.cutoff_frequency:
                    IHPF[u, v] = 0  # Zero out low-frequency components
        
        # Apply high-pass filter to frequency domain representation
        ksp_filtered = kspace * IHPF
        
        return ksp_filtered

    def calculate_high_frequency_ratio(self, sample):
        """
        Calculate the high-frequency energy ratio for a given sample.

        * sample: The sample containing k-space data.
        * return: The high-frequency ratio of the sample.
        """
        kspace = sample[0]  # Assuming k-space data is at index 0
        ksp_filtered = self.apply_high_pass_filter(kspace)
        
        numerator = np.sum(np.abs(ksp_filtered) ** 2)
        denominator = np.sum(np.abs(kspace) ** 2)
        
        return numerator / denominator
    
    def filter_by_threshold(self):
        """
        Filter the dataset by removing samples where the high-frequency ratio is above the threshold.
        """
        filtered_samples = []
        for sample in tqdm(self.dataset, desc="Filtering samples by high-frequency threshold"):
            high_freq_ratio = self.calculate_high_frequency_ratio(sample)

            if high_freq_ratio > self.threshold:
                filtered_samples.append((sample[-2], sample[-1]))  # Store file name and slice number
        return filtered_samples

    def filter_by_size(self):
        """
        Filter the dataset by selecting the top samples with the highest high-frequency ratio, up to max_samples.
        """
        samples_with_ratio = []
        for sample in tqdm(self.dataset, desc="Calculating high-frequency ratios"):
            high_freq_ratio = self.calculate_high_frequency_ratio(sample)
            samples_with_ratio.append((sample[-2], sample[-1], high_freq_ratio))  # Store file name, slice number, and high-frequency ratio
        
        # Sort samples by high-frequency ratio (highest to lowest)
        samples_with_ratio.sort(key=lambda x: x[2], reverse=True)

        # Calculate how many samples need to be removed to reach the target size
        num_samples_to_remove = len(samples_with_ratio) - self.max_samples
        if num_samples_to_remove <= 0:
            print("No samples need to be removed; dataset size is already less than or equal to max_samples.")
            return []

        # Select the samples to remove (those with the highest high-frequency ratio)
        filtered_samples = [(fname, slice_num) for fname, slice_num, _ in samples_with_ratio[:num_samples_to_remove]]

        return filtered_samples

    def filter(self):
        """
        Perform high-frequency filtering over the entire dataset.

        * return: A list of filtered samples as (filename, slice_num) tuples in non-debug mode.
                  In debug mode, return a list of samples and their high-frequency ratios.
        """
        if self.is_debug:
            ratio_record = []
            samples_record = []
            count = 0
            for sample in tqdm(self.dataset, desc="Calculating high-frequency ratios in debug mode"):
                high_freq_ratio = self.calculate_high_frequency_ratio(sample)
                ratio_record.append(high_freq_ratio)
                samples_record.append((sample[-2], sample[-1]))  # Storing file name and slice number
                count += 1
                if count == 10000:  # Limit to 1000 samples in debug mode
                    break
            return samples_record, ratio_record
        else:
            if self.mode == "threshold":
                return self.filter_by_threshold()
            elif self.mode == "size":
                return self.filter_by_size()
            else:
                raise ValueError(f"Unknown mode: {self.mode}")