import torch
import pandas as pd

class StartingDataset(torch.utils.data.Dataset):
    """
    Dataset that contains 100000 3x224x224 black images (all zeros).
    """

    def __init__(self, image_ids, labels):
        self.image_ids = image_ids
        self.labels = labels

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        label = self.labels[index]
        both = (image_id, label)

        return both

    def __len__(self):
        return len(self.image_ids)
