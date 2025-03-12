import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import glob
import numpy as np
import os


class WSSPeakDataset(Dataset):
    def __init__(self, root_dir, 
                 encode_size=12800, decode_size=3600,
                 labels=['x', 'y', 'z', 'wss', 'wss_x', 'wss_y', 'wss_z']):
        self.files = glob.glob(f"{root_dir}/*.csv")
        self.data = [read_ansys_csv(f) for f in self.files]
        self.normalize()
        self.encode_size = min(encode_size, min(data.shape[0] for data in self.data))
        self.decode_size = min(decode_size, min(data.shape[0] for data in self.data))
        self.labels = labels
        self.sampled_indices = []
        self.resample()

    def normalize(self):
        group_mean = []
        group_std = []
        for idx, tensor_ in enumerate(self.data):
            group_mean.append(tensor_.mean(dim=0))
            group_std.append(tensor_.std(dim=0))
        group_mean = torch.mean(torch.stack(group_mean), dim=0)
        group_std = torch.mean(torch.stack(group_std), dim=0)
        group_mean[:3] = torch.mean(group_mean[:3], dim=0)
        group_std[:3] = torch.mean(group_std[:3], dim=0)
        self.group_mean, self.group_std = group_mean, group_std
        for idx, tensor in enumerate(self.data):
            self.data[idx] = (tensor - group_mean) / group_std

    def resample(self):
        self.sampled_indices = []
        self.recon_sampled_indices = []
        for idx, tensor in enumerate(self.data):
            total_rows = tensor.shape[0]
            sampled_indices = torch.randperm(total_rows)[:self.encode_size]  # Random sampling without replacement
            recon_sampled_indices = torch.randperm(total_rows)[:self.decode_size]
            self.sampled_indices.append(sampled_indices)
            self.recon_sampled_indices.append(recon_sampled_indices)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tensor = self.data[idx]
        coords = tensor[:, :3]
        # feats = tensor[:, 3:]
        feats = tensor[:, 3:4]
        sampled_indices = self.sampled_indices[idx]
        recon_sampled_indices = self.recon_sampled_indices[idx]
        return {"coords": coords[sampled_indices], "feats": feats[sampled_indices],
                "recon_coords": coords[recon_sampled_indices], "recon_feats": feats[recon_sampled_indices]}
    


def read_ansys_csv(file):
    # Read the CSV file, skipping the first three rows
    df = pd.read_csv(file, skiprows=5)
    tensor_ = torch.tensor(df.values, dtype=torch.float32)
    return tensor_




class ResamplingCSVLoader(Dataset):
    def __init__(self, folder_path, resample_size=None, seed=None):
        
        self.data = [torch.tensor(pd.read_csv(f).values, dtype=torch.float32) for f in self.files]
        self.resample_size = resample_size
        self.seed = seed
        self.resample_indices()

    def resample_indices(self):
        """Randomly sample indices from the loaded data."""
        if self.seed is not None:
            np.random.seed(self.seed)
        
        self.sampled_indices = []
        for idx, tensor in enumerate(self.data):
            total_rows = tensor.shape[0]
            sample_size = min(self.resample_size, total_rows) if self.resample_size else total_rows
            sampled_indices = np.random.choice(total_rows, sample_size, replace=False)
            self.sampled_indices.append((idx, sampled_indices))

    def __len__(self):
        return sum(len(indices) for _, indices in self.sampled_indices)

    def __getitem__(self, idx):
        cumulative_idx = 0
        for file_idx, indices in self.sampled_indices:
            if idx < cumulative_idx + len(indices):
                relative_idx = idx - cumulative_idx
                return self.data[file_idx][indices[relative_idx]]
            cumulative_idx += len(indices)
        raise IndexError(f"Index {idx} out of range")

