import socket

from contextlib import closing
from typing import Optional
from tqdm import tqdm

from torch.utils.data import IterableDataset, Dataset

def find_free_port():
    """ https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number &
        https://stackoverflow.com/questions/66498045/how-to-solve-dist-init-process-group-from-hanging-or-deadlocks
    """

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])

class InMemoryDataset(Dataset):
    def __init__(self, iterable_ds: IterableDataset, use_tqdm : bool = False, device : Optional[str] = None, repeat_dataset : int = 1):
        device = device if device is not None else "cpu"
        it = iterable_ds if not use_tqdm else tqdm(iterable_ds, desc=f"Caching tensors in {device}")
        self.data = []
        for _ in range(repeat_dataset):
            cntr = 0
            for x in it:
                # debug
                #if cntr > 100:
                    #break

                self.data.append(x.to(device))

                cntr += 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

def cache_iterable_in_memory(iterable_ds: IterableDataset, use_tqdm : bool = False, device : Optional[str] = None, repeat_dataset : int = 1):
    return InMemoryDataset(iterable_ds, use_tqdm, device, repeat_dataset)
