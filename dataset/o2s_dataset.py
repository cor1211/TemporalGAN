from torch.utils.data import Dataset
import os
from torchvision.transforms import Compose, Normalize, ToPILImage
import numpy as np
import yaml
from pathlib import Path
import sys
import random
import torch

S1_FOLDER_NAME = 's1_256_vv'
S2_FOLDER_NAME = 's2_256'
LC_FOLDER_NAME = 'lc_2048'


def load_config(config_path: str):
    try:
        with open(Path(config_path), 'r') as file:
            config = yaml.safe_load(file)
            return config
    except FileNotFoundError as e:
        print(f'{e}: Your config path is not valid.')
        sys.exit(1)


class O2SDataset(Dataset):
    def __init__(self, train: bool = False, valid:bool = False, test:bool = False, transform_rgb = None, transform_gray = None, cfg_data:dict = None):
        self.is_train = train
        root = cfg_data['root']
        if self.is_train:
            root = os.path.join(root, cfg_data['train_folder_name'])
        elif valid:
            root = os.path.join(root, cfg_data['valid_folder_name'])
        else:
            root = os.path.join(root, cfg_data['test_folder_name'])
        
        self.transform_rgb = transform_rgb
        self.transform_gray = transform_gray
        self.modalities = cfg_data['modalities']

        self.s2_path = []
        self.lc_path = []
        self.s1_path = []

        # Modalities is a list, e.g., ['s1', 's2', 'lc'] or ['s1_256_vv', 's2_256', 'lc_2048']
        import torch
        for folder in self.modalities:
            folder_path = os.path.join(root, folder)
            if not os.path.exists(folder_path):
                print(f"Directory missing: {folder_path}")
                continue
                
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                # Flexible matching
                if 's1' in folder.lower():
                    self.s1_path.append(file_path)
                elif 's2' in folder.lower():
                    self.s2_path.append(file_path)
                elif 'lc' in folder.lower():
                    self.lc_path.append(file_path)
                else:
                    raise ValueError(f'Folder name "{folder}" of modalities is not valid. Check at base_config.yaml')
    
        # Sort file paths to ensure correspondence between modalities
        self.s1_path.sort()
        self.s2_path.sort()
        self.lc_path.sort()
        
        # Check data consistency
        assert len(self.s1_path) == len(self.s2_path) == len(self.lc_path), f"Mismatch data length: S1({len(self.s1_path)}), S2({len(self.s2_path)}), LC({len(self.lc_path)})"
    
    
    def __len__(self):
        return len(self.s2_path)
    
    def __getitem__(self, index):
        # 1. Load numpy arrays
        s2 = np.load(self.s2_path[index])
        lc = np.load(self.lc_path[index])
        s1 = np.load(self.s1_path[index])

        # 2. Ensure Arrays have C channel to prepare for transformations (H, W, C)
        if s1.ndim == 2:
            s1 = np.expand_dims(s1, axis=-1)
        if s2.ndim == 2:
            s2 = np.expand_dims(s2, axis=-1)
        if lc.ndim == 2:
            lc = np.expand_dims(lc, axis=-1)

        # 3. Apply numpy-level random augmentations for Training set
        if self.is_train:
            # Horizontal flip
            if random.random() > 0.5:
                s2 = s2[:, ::-1, :].copy()
                lc = lc[:, ::-1, :].copy()
                s1 = s1[:, ::-1, :].copy()
                
            # Random Rotate 90, 180, 270 degrees
            rot_k = random.choice([0, 1, 2, 3])
            if rot_k > 0:
                s2 = np.rot90(s2, k=rot_k, axes=(0, 1)).copy()
                lc = np.rot90(lc, k=rot_k, axes=(0, 1)).copy()
                s1 = np.rot90(s1, k=rot_k, axes=(0, 1)).copy()

        # 4. Extract required features
        # For SAR, if there are multi bands, usually channel 0 is Amplitude/Intensity
        if s1.shape[-1] > 1:
            s1 = s1[..., 0:1]

        # 5. Convert to PyTorch Tensors
        s2_t = torch.from_numpy(s2).float()
        lc_t = torch.from_numpy(lc).float()
        s1_t = torch.from_numpy(s1).float()

        # 6. Change axis from HWC -> CHW
        s2_t = s2_t.permute(2, 0, 1)
        lc_t = lc_t.permute(2, 0, 1)
        s1_t = s1_t.permute(2, 0, 1)

        # 7. Apply Torchvision normalization pipelines (usually passed via Compose)
        if self.transform_rgb:
            # ToTensor is no longer needed in config since we explicitly parsed it here.
            s2_t = self.transform_rgb(s2_t)
            lc_t = self.transform_rgb(lc_t)
        if self.transform_gray:
            s1_t = self.transform_gray(s1_t)

        return (s2_t, lc_t), s1_t



if __name__ == '__main__':
    index = 0
    config = load_config('/mnt/data1tb/vinh/TemporalGAN/config/base_config.yaml')
    
    transform_rgb_test = Compose([Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    transform_gray_test = Compose([Normalize(mean=0.5, std=0.5)])
    train_set = O2SDataset(
        train=True,
        cfg_data=config['data'],
        # transform_rgb=transform_rgb_test,
        # transform_gray=transform_gray_test
    )
    valid_set = O2SDataset(
        valid=True,
        cfg_data=config['data'],
        # transform_rgb=transform_rgb_test,
        # transform_gray=transform_gray_test
    )
    toPIL = ToPILImage()
    print(len(valid_set))
    (s2, lc), s1 = train_set[index]
    toPIL(s2).show()
    toPIL(lc).show()
    toPIL(s1).show()
    # Simple check
    print("S2 Tensor:", s2.shape, s2.min(), s2.max(), s2.dtype)
    print("LC Tensor:", lc.shape, lc.min(), lc.max(), lc.dtype)
    print("S1 Tensor:", s1.shape, s1.min(), s1.max(), s1.dtype)
