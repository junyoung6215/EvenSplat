from typing import Iterator, Optional

import torch
from torch.utils.data import Dataset, IterableDataset


class ValidationWrapper(Dataset):
    """Wraps a dataset so that PyTorch Lightning's validation step can be turned into a
    visualization step.
    """

    dataset: Dataset
    dataset_iterator: Optional[Iterator]
    length: int

    def __init__(self, dataset: Dataset, length: int) -> None:
        super().__init__()
        self.dataset = dataset
        self.length = length
        self.dataset_iterator = None

    def __len__(self):
        return self.length

    def __getitem__(self, index: int):
        if isinstance(self.dataset, IterableDataset):
            if self.dataset_iterator is None:
                self.dataset_iterator = iter(self.dataset)
            try:
                return next(self.dataset_iterator)
            except StopIteration:
                # Lightning reuses this wrapper across validation checks. Small
                # subsets must cycle instead of silently yielding no batches.
                self.dataset_iterator = iter(self.dataset)
                try:
                    return next(self.dataset_iterator)
                except StopIteration as error:
                    # Do not hide an empty/fully filtered dataset or loop forever.
                    self.dataset_iterator = None
                    raise RuntimeError(
                        "Validation dataset yielded no samples. Check the validation "
                        "index, dataset split, and scene/image filters."
                    ) from error

        random_index = torch.randint(0, len(self.dataset), tuple())
        return self.dataset[random_index.item()]
