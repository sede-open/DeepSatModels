from __future__ import print_function, division
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch.utils.data
import pickle
import warnings
warnings.filterwarnings("ignore")



def get_distr_dataloader(paths_file, root_dir, rank, world_size, transform=None, batch_size=32, num_workers=4,
                         shuffle=True, return_paths=False):
    """
    return a distributed dataloader
    """
    dataset = SatImDataset(csv_file=paths_file, root_dir=root_dir, transform=transform, return_paths=return_paths)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                             pin_memory=True, sampler=sampler)
    return dataloader


def get_dataloader(paths_file, root_dir, transform=None, batch_size=32, num_workers=4, shuffle=True,
                   return_paths=False, my_collate=None):
    dataset = SatImDataset(csv_file=paths_file, root_dir=root_dir, transform=transform, return_paths=return_paths)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                             collate_fn=my_collate)
    return dataloader


class SatImDataset(Dataset):
    """Satellite Images dataset."""

    def __init__(self, csv_file, root_dir, transform=None, multilabel=False, return_paths=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if type(csv_file) == str:
            self.data_paths = pd.read_csv(csv_file, header=None)
        elif type(csv_file) in [list, tuple]:
            self.data_paths = pd.concat([pd.read_csv(csv_file_, header=None) for csv_file_ in csv_file], axis=0).reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform
        self.multilabel = multilabel
        self.return_paths = return_paths

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.data_paths.iloc[idx, 0])

        if not os.path.isfile(img_name):
            print(f"PASTIS24/dataloader.py:63 -- `img_name` (='{img_name}') CANNOT be found on the local system\n")
            from azureml.fsspec import AzureMachineLearningFileSystem
            uri = 'azureml://subscriptions/8b829ed0-f80d-4b85-8636-a7967eaf7aed/resourcegroups/semantic_segmentation/workspaces/bioversml/datastores/covercropatlas'
            folder_path = os.path.split(img_name.replace("../", ""))[0]
            filename = os.path.split(img_name)[-1]
            source_file_path = os.path.join("dataset", folder_path, filename)
            dest_folder_path = os.path.join("download_files",folder_path, filename)
            print(f"Downloading `{source_file_path}` to `{dest_folder_path}`...")
            fs = AzureMachineLearningFileSystem(uri)
            fs.download(rpath=source_file_path, lpath=dest_folder_path, recursive=False)
            raise Exception()
        else:
            print(f"PASTIS24/dataloader.py:63 -- `img_name` (='{img_name}') DOES exist on the local system\n")

        with open(img_name, 'rb') as handle:
            sample = pickle.load(handle, encoding='latin1')

        if self.transform:
            message = "\n"
            message += f"Data type: {sample['img'].dtype}\n"  # int16
            message += f"Shape: {sample['img'].shape}\n"  # (43, 10, 24, 24)
            # message += f"Sample data: {sample['img'][:2]}\n"
            message += f"Max value in img: {np.max(sample['img'])}\n"
            message += f"Min value in img: {np.min(sample['img'])}\n"

            # Check for NaNs or infs in the data
            if np.any(np.isnan(sample['img'])) or np.any(np.isinf(sample['img'])):  # NaN and/or Infinity NOT found in sample['img']
                message += "NaN and/or Infinity FOUND in sample['img']\n"
            else:
                message += "NaN and/or Infinity NOT found in sample['img']\n"
            
            print(message)
            sample = self.transform(sample)

        if self.return_paths:
            return sample, img_name
        
        return sample

    def read(self, idx, abs=False):
        """
        read single dataset sample corresponding to idx (index number) without any data transform applied
        """
        if type(idx) == int:
            img_name = os.path.join(self.root_dir,
                                    self.data_paths.iloc[idx, 0])
        if type(idx) == str:
            if abs:
                img_name = idx
            else:
                img_name = os.path.join(self.root_dir, idx)
        with open(img_name, 'rb') as handle:
            sample = pickle.load(handle, encoding='latin1')
        return sample
    
    
def my_collate(batch):
    "Filter out sample where mask is zero everywhere"
    idx = [b['unk_masks'].sum(dim=(0, 1, 2)) != 0 for b in batch]
    batch = [b for i, b in enumerate(batch) if idx[i]]
    return torch.utils.data.dataloader.default_collate(batch)
