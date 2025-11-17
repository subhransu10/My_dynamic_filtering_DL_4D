import os
import numpy as np
import torch
from torch.utils.data import Dataset

from .config import Config

#loads the npz files *in numpy library npz fle is a zipped archive that stores multiple numpy arrays for motion segmentation (a pytorch dataset),loads much faster than text files and can store multiple arrays
#in a single file
class RMOSNPZDataset(Dataset):
    def __init__(self, root, list_file):
        self.root = root
        with open(list_file, "r") as f:
            self.files = [l.strip() for l in f if l.strip()]
        assert len(self.files) > 0, f"No samples found in {list_file}"

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.root, self.files[idx])
        data = np.load(path)
        coords = data["coords"].astype(np.int32)
        feats = data["feats"].astype(np.float32)
        labels = data["labels"].astype(np.int32)

        # Return numpy; collate will convert to tensors
        return coords, feats, labels
