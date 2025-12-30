from temporalgan.gen_s2_lc_v1_0 import Generator
from temporalgan.disc_s2_lc_v1_0 import Discriminator
from dataset.o2s_dataset import O2SDataset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
import sys
from pathlib import Path
import yaml
from argparse import ArgumentParser
import torch
from torch.optim import Adam
from datetime import datetime
import os
from torch.utils.tensorboard import SummaryWriter
from trainer import Trainer
import random
import numpy as np

def load_config(config_path: str):
    try:
        with open(Path(config_path), 'r') as file:
            config = yaml.safe_load(file)
            return config
    except FileNotFoundError as e:
        print(f'{e}: Your config path is not valid.')
        sys.exit(1)


def set_seed(seed=42):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    set_seed(42) # Set seed
    parser = ArgumentParser(prog="Model Training")
    parser.add_argument('--config_path', type=str, required=True)
    
    args = parser.parse_args()
    #---------Load yaml file---------------
    config_dict = load_config(args.config_path)
    cfg_data = config_dict['data']
    cfg_train = config_dict['train']

    #---------Load checkpoint---------------
    ckp_path = cfg_train['resume_path']
    if ckp_path:
        ckp_path = Path(ckp_path)
        if not os.path.exists(ckp_path):
            raise FileNotFoundError(f'Resume_path: {ckp_path} is not valid!')
        try: 
            run_name = ckp_path.parent.name
        except Exception as e:
            print(f'Cannot extract run_name from {ckp_path}')
            sys.exit(1)
    else:
        run_name = f'exp_{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    
    #-----------Init Summary Writer to log-------------
    logdir = os.path.join('runs', run_name)
    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(log_dir = logdir)
    print(f'TensorBoard logs will be saved in {logdir}')


    #-------------Dataset, Dataloader-----------
    transform_RGB = Compose(transforms=[
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])
    ])

    transform_gray = Compose(transforms=[
        ToTensor(),
        Normalize(mean=[0.5], std=[0.5])
    ])

        #------Dataset--------------
    train_set = O2SDataset(train=True, transform_rgb=transform_RGB, transform_gray= transform_gray, cfg_data=cfg_data)
    valid_set = O2SDataset(valid=True, transform_rgb=transform_RGB, transform_gray=transform_gray, cfg_data=cfg_data)
    test_set = O2SDataset(test=True, transform_rgb=transform_RGB, transform_gray=transform_gray, cfg_data=cfg_data)


        #--------Dataloader----------
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=cfg_data['train_batch_size'],
        shuffle=True,
        num_workers=cfg_data['num_workers'],
        drop_last=True
    )

    valid_loader = DataLoader(
        dataset=valid_set,
        batch_size=cfg_data['valid_batch_size'],
        shuffle=False,
        num_workers=cfg_data['num_workers'],
        drop_last=True
    )


    #---------Configure model------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    netG = Generator(s2_in_channels=3, lc_in_channels=3, out_channels=1, features=64)
    netD = Discriminator(s2_in_channels=3, lc_in_channels=3, s1_out_channels=1)
    optG = Adam(netG.parameters(), lr = cfg_train['lr'], betas=tuple(cfg_train['betas']))
    optD = Adam(netD.parameters(), lr = cfg_train['lr'], betas=tuple(cfg_train['betas']))


    #----------Train-----------
    train = Trainer(netG, netD, optG, optD, train_loader, valid_loader, device, config_dict, writer, run_name, ckp_path)
    train.run()
