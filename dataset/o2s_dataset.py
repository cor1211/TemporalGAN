from torch.utils.data import Dataset
import os
from torchvision.transforms import Compose, Normalize, ToTensor, ToPILImage
import torchvision.transforms.functional as F
from PIL import Image
import yaml
from pathlib import Path
import sys
import random

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

        for folder in self.modalities: # s1_256_vv | s2_256 | lc_2048
            folder_path = os.path.join(root, folder)
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                if folder.strip() == S1_FOLDER_NAME:
                    self.s1_path.append(file_path)
                elif folder.strip() == S2_FOLDER_NAME:
                    self.s2_path.append(file_path)
                elif folder.strip() == LC_FOLDER_NAME:
                    self.lc_path.append(file_path)
                else:
                    raise ValueError('Folder name of modalities is not valid. Check at base_config.yaml')
    
        # Sort file paths to ensure correspondence between modalities
        self.s1_path.sort()
        self.s2_path.sort()
        self.lc_path.sort()
        
        # Check data consistency
        assert len(self.s1_path) == len(self.s2_path) == len(self.lc_path), f"Mismatch data length: S1({len(self.s1_path)}), S2({len(self.s2_path)}), LC({len(self.lc_path)})"
    
    
    def __len__(self):
        return len(self.s2_path)
    
    def __getitem__(self, index):
        s2 = Image.open(self.s2_path[index]).convert('RGB')
        lc = Image.open(self.lc_path[index]).convert('RGB')
        s1 = Image.open(self.s1_path[index]).convert('L')

        lc = lc.resize(size=(256,256), resample=Image.Resampling.BICUBIC)

        # Apply augmentations only for the training set
        if self.is_train:
            # Random horizontal flip with 50% probability, applied consistently
            if random.random() > 0.5:
                s2 = F.hflip(s2)
                lc = F.hflip(lc)
                s1 = F.hflip(s1)

        if self.transform_rgb:
            s2 = self.transform_rgb(s2)
            lc = self.transform_rgb(lc)
        if self.transform_gray:
            s1 = self.transform_gray(s1)

        return (s2, lc), s1



if __name__ == '__main__':
    index = 3
    config = load_config('/mnt/data1tb/vinh/TemporalGAN/config/base_config.yaml')
    
    transform_rgb_test = Compose([ToTensor(), Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    transform_gray_test = Compose([ToTensor(), Normalize(mean=0.5, std=0.5)])
    train_set = O2SDataset(
        train=True,
        cfg_data=config['data'],
        transform_rgb=transform_rgb_test,
        transform_gray=transform_gray_test
    )
    valid_set = O2SDataset(
        valid=True,
        cfg_data=config['data'],
        transform_rgb=transform_rgb_test,
        transform_gray=transform_gray_test
    )

    print(len(train_set))
    (s2, lc), s1 = train_set[index]
    to_pil = ToPILImage()
    to_pil((s2+1) *0.5 ).show()
    to_pil((lc+1)*0.5).show()
    to_pil((s1+1)*0.5).show()
