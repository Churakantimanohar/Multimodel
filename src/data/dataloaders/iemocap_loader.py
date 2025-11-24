import torch
from torch.utils.data import Dataset

class IEMOCAPDataset(Dataset):
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.samples = []

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        raise NotImplementedError("Implement IEMOCAP parsing")
