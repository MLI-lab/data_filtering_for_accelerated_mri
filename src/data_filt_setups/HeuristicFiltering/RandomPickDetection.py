import numpy as np
from tqdm.autonotebook import tqdm
from src.data_filt_setups.data_filt_setups import register_data_filtering_setup
from src.data_filt_setups.utils import SliceDataset
import random


@register_data_filtering_setup()
class RandomPickDetection:
    def __init__(self, file_path: str = None, dataset=None, **kwargs):
        """
        Initialize the RandomPickDetection class.

        * file_path: Path to the dataset JSON file.
        * dataset: Pre-loaded dataset object (if available).
        * kwargs: Additional keyword arguments, including:
            - is_debug: Boolean indicating whether to run in debug mode.
            - max_samples: Maximum number of samples to evaluate (useful for size mode).
        """
        if not file_path and not dataset:
            raise ValueError("Either file_path or dataset must be provided")
        self.file_path = file_path
        self.dataset = dataset if dataset else SliceDataset(file_path)
        self.max_samples = int(kwargs.get('max_samples', 10000))  # Default to 10000
        self.is_debug = kwargs.get('is_debug', False)

    def generate_random_value(self):
        """
        Generate a random value for each sample to simulate random selection.
        
        * return: A random number between 0 and 1.
        """
        return np.random.random()

    def filter_by_size(self):
        """
        Randomly select a number of samples up to max_samples.
        """
        samples_with_random_value = []
        
        # Step 1: Generate random value for each sample and store it
        for sample in tqdm(self.dataset, desc="Selecting random samples"):
            random_value = self.generate_random_value()
            samples_with_random_value.append((sample[-2], sample[-1], random_value))  # Store file name, slice number, and random value
        
        # Step 2: Randomly shuffle samples to simulate random selection
        random.shuffle(samples_with_random_value)

        # Step 3: Select samples to keep (those with the highest random value until max_samples)
        num_samples_to_remove = len(samples_with_random_value) - self.max_samples
        if num_samples_to_remove <= 0:
            print("No samples need to be removed; dataset size is already less than or equal to max_samples.")
            return []
        
        # Select the samples to remove (those randomly selected)
        filtered_samples = [(fname, slice_num) for fname, slice_num, _ in samples_with_random_value[:num_samples_to_remove]]

        return filtered_samples

    def filter(self):
        """
        Perform random sample selection over the entire dataset.

        * return: A list of filtered samples as (filename, slice_num) tuples in non-debug mode.
                  In debug mode, return a list of samples and their randomly generated values.
        """
        if self.is_debug:
            random_value_record = []
            samples_record = []
            count = 0
            for sample in tqdm(self.dataset, desc="Generating random values for samples"):
                random_value = self.generate_random_value()
                random_value_record.append(random_value)
                samples_record.append((sample[-2], sample[-1]))  # Storing file name and slice number
                count += 1
                if count == 10000:
                    break
            return samples_record, random_value_record
        else:
            return self.filter_by_size()