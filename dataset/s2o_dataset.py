from torch.utils.data import Dataset
import os
from torchvision.transforms import Compose, Normalize, ToTensor, RandomHorizontalFlip
from PIL import Image
import yaml
from pathlib import Path
import sys

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


class S2ODataset(Dataset):
    def __init__(self, root, train: bool = False, valid:bool = False, test:bool = False, transform = None, cfg_data:dict = None):
        if train:
            root = os.path.join(root, cfg_data['train_folder_name'])
        elif valid:
            root = os.path.join(root, cfg_data['valid_folder_name'])
        else:
            root = os.path.join(root, cfg_data['test_folder_name'])
        
        self.transform = transform
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
    
    def __len__(self):
        return len(self.s2_path)
    
    def __getitem__(self, index):
        s2 = Image.open(self.s2_path[index]).convert('RGB')
        lc = Image.open(self.lc_path[index]).convert('RGB')
        s1 = Image.open(self.s1_path[index]).convert('L')

        lc = lc.resize(size=(256,256), resample=Image.Resampling.BICUBIC)

        flip = RandomHorizontalFlip(1)
        s2 = flip(s2)
        lc = flip(lc)
        s1 = flip(s1)

        if self.transform: 
            s2 = self.transform(s2)
            lc = self.transform(lc)
            s1 = self.transform(s1)
        
        return (s2, lc), s1



if __name__ == '__main__':
    index = 3
    config = load_config('/mnt/data1tb/vinh/TemporalGAN/config/base_config.yaml')
    train_set = S2ODataset('/mnt/data1tb/vinh/TemporalGAN/dataset/o2s', train = True, cfg_data=config['data'])

    (s2, lc), s1 = train_set[index]
    s2.show()
    lc.show()
    s1.show()

