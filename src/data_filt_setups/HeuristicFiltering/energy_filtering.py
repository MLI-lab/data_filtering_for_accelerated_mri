import numpy as np
from tqdm.autonotebook import tqdm
from src.data_filt_setups.data_filt_setups import register_data_filtering_setup
from src.data_filt_setups.utils import SliceDataset


@register_data_filtering_setup()
class EnergyFiltering:
    def __init__(self, file_path: str = None, dataset=None, **kwargs):
        """
        Initialize the EnergyFiltering class.

        * file_path: Path to the dataset JSON file.
        * dataset: Pre-loaded dataset object (if available).
        * kwargs: Additional keyword arguments, including:
            - is_debug: Boolean indicating whether to run in debug mode.
            - max_samples: Maximum number of samples to evaluate (useful for debug or size mode).
            - threshold: The threshold for signal ratio to classify low signal samples.
            - mode: Either 'size' or 'threshold' to determine filtering logic.
        """
        if not file_path and not dataset:
            raise ValueError("Either file_path or dataset must be provided")
        self.file_path = file_path
        self.dataset = dataset if dataset else SliceDataset(file_path)
        self.max_samples = int(kwargs.get('max_samples', 10000))
        self.threshold = float(kwargs.get('threshold', 0.1))  # Ensure threshold is a float
        self.is_debug = kwargs.get('is_debug', False)
        self.mode = kwargs.get('mode', 'size')  # Default mode is 'size'

    def calculate_signal_ratio(self, sample):
        """
        Calculate the signal ratio of a sample based on the maximum value in the slice.

        * sample: The sample data to be evaluated.
        * return: The signal ratio of the sample.
        """
        target = np.abs(sample[1])
        maxval = sample[2]['max_mvue']
        return np.max(target) / maxval
    
    def filter_by_threshold(self):
        """
        Filter the dataset by removing samples where the signal ratio is below the threshold.
        """
        filtered_samples = []
        for sample in tqdm(self.dataset, desc="Filtering low signal samples by threshold"):
            if self.calculate_signal_ratio(sample) < self.threshold:
                filtered_samples.append((sample[-2], sample[-1]))  # Store file name and slice number
        return filtered_samples
    
    def filter_by_size(self):
        """
        Filter the dataset by selecting the top samples with the lowest signal ratio, up to max_samples.
        """
        samples_with_ratio = []
        for sample in tqdm(self.dataset, desc="Calculating signal ratios"):
            signal_ratio = self.calculate_signal_ratio(sample)
            samples_with_ratio.append((sample[-2], sample[-1], signal_ratio))  # Store file name, slice number, and signal ratio
        
        # Sort samples by signal ratio (lowest to highest)
        samples_with_ratio.sort(key=lambda x: x[2])

        # Calculate how many samples need to be removed to reach the target size
        num_samples_to_remove = len(samples_with_ratio) - self.max_samples
        if num_samples_to_remove <= 0:
            print("No samples need to be removed; dataset size is already less than or equal to max_samples.")
            return []
        
        # Select the samples to remove (those with the lowest signal ratio)
        filtered_samples = [(fname, slice_num) for fname, slice_num, _ in samples_with_ratio[:num_samples_to_remove]]

        return filtered_samples

    def filter(self):
        """
        Perform low signal detection over the entire dataset.

        * return: A list of filtered samples as (filename, slice_num) tuples in non-debug mode.
                  In debug mode, return a list of samples and their signal ratios.
        """
        if self.is_debug:
            ratio_record = []
            samples_record = []
            count = 0
            for sample in tqdm(self.dataset, desc="Calculating signal ratios in debug mode"):
                ratio_record.append(self.calculate_signal_ratio(sample))
                samples_record.append((sample[-2], sample[-1]))  # Storing file name and slice number
                count += 1
                if count == 10000:
                    break
            return samples_record, ratio_record
        else:
            if self.mode == "threshold":
                return self.filter_by_threshold()
            elif self.mode == "size":
                return self.filter_by_size()
            else:
                raise ValueError(f"Unknown mode: {self.mode}")