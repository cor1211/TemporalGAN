# from temporalgan.gen_s2_lc_v1_0 import Generator
from temporalgan import gen_s2_lc_v1_0, gen_s2_lc_v1_1
# from temporalgan.disc_s2_lc_v1_0 import Discriminator
from temporalgan import disc_s2_lc_v1_0
from dataset.o2s_dataset import O2SDataset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize
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

# Mapping from config model to Object
GENERATORS = {
    'gen_s2_lc_v1_0': gen_s2_lc_v1_0,
    'gen_s2_lc_v1_1': gen_s2_lc_v1_1,
}

DISCRIMINATORS = {
    'disc_s2_lc_v1_0': disc_s2_lc_v1_0
}


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
    torch.cuda.empty_cache()
    parser = ArgumentParser(prog="Model Training")
    parser.add_argument('--config_path', type=str, default='/mnt/data1tb/vinh/TemporalGAN/config/base_config.yaml')
    parser.add_argument('--use_wandb', action='store_true', help='Enable Weights & Biases logging')
    args = parser.parse_args()

    #---------Load yaml file---------------
    config_dict = load_config(args.config_path)
    cfg_data = config_dict['data']
    cfg_train = config_dict['train']
    cfg_model = config_dict['model']

    #--------Set seed--------------
    set_seed(cfg_train.get('seed', 42)) 

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
    writer = SummaryWriter(log_dir=logdir)
    print(f'TensorBoard logs will be saved in {logdir}')

    #-----------Init WandB (optional)------------------
    use_wandb = args.use_wandb
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project="TemporalGAN-O2S",
                name=run_name,
                config=config_dict,
            )
            print(f'WandB initialized: project=TemporalGAN-O2S, run={run_name}')
        except ImportError:
            print("Warning: wandb not installed. Falling back to TensorBoard only.")
            use_wandb = False
        except Exception as e:
            print(f"Warning: wandb init failed: {e}. Falling back to TensorBoard only.")
            use_wandb = False


    #-------------Dataset, Dataloader-----------
    transform_RGB = Compose(transforms=[
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    transform_gray = Compose(transforms=[
        Normalize(mean=[0.5], std=[0.5])
    ])

        #------Dataset--------------
    train_set = O2SDataset(train=True, transform_rgb=transform_RGB, transform_gray=transform_gray, cfg_data=cfg_data)
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
    if not cfg_train['device']:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        try:
            device = torch.device(cfg_train['device'])
        except Exception as e:
            print(f'Explain error.\nIn yaml base_config, device: {cfg_train["device"]}: {e}')
            sys.exit(1)

    # Validate and load generator and discriminator from config
    gen_module_name = cfg_model['generator']
    disc_module_name = cfg_model['discriminator']
    
    if gen_module_name not in GENERATORS:
        raise ValueError(f"Generator '{gen_module_name}' not found. Available: {list(GENERATORS.keys())}")
    
    if disc_module_name not in DISCRIMINATORS:
        raise ValueError(f"Discriminator '{disc_module_name}' not found. Available: {list(DISCRIMINATORS.keys())}")
    
    generator_module = GENERATORS[gen_module_name]
    discriminator_module = DISCRIMINATORS[disc_module_name]

    #--------------Initialize Models---------------
    netG = generator_module.Generator(s2_in_channels=3, lc_in_channels=3, out_channels=1, features=64)
    netD = discriminator_module.Discriminator(s2_in_channels=3, lc_in_channels=3, s1_out_channels=1)
    optG = Adam(netG.parameters(), lr=cfg_train['lr'], betas=tuple(cfg_train['betas']))
    optD = Adam(netD.parameters(), lr=cfg_train['lr'], betas=tuple(cfg_train['betas']))


    #----------Train-----------
    trainer = Trainer(netG, netD, optG, optD, train_loader, valid_loader, device, config_dict, writer, run_name, 
                      resume_path=ckp_path, 
                      strict_netG=cfg_train['strict_netG'],
                      strict_netD=cfg_train['strict_netD'],
                      use_wandb=use_wandb)
    trainer.run()
