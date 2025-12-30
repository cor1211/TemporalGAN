"""
Infer output with inputs {s2, lc}
"""

from temporalgan.gen_s2_lc_v1_0 import Generator
from torchvision.transforms import Compose, ToTensor, Normalize, ToPILImage
import torch
from argparse import ArgumentParser
from PIL import Image
import yaml
from pathlib import Path
import sys


def load_config_file(config_path):
    try:
        with open(Path(config_path), 'r') as file:
            config = yaml.safe_load(file)
            return config
    except FileNotFoundError:
        print(f'Infer config path: {config_path} is not valid')
        sys.exit(1)


def denorm(img: torch.Tensor):
    """
    denorm from [-1, 1] to [0, 1]
    """
    img = (img * 0.5 + 0.5).clamp(0, 1)
    return img


if __name__ == '__main__':
    parser = ArgumentParser(prog="Inference")
    parser.add_argument('--config_path', type = str, required = True, help='inference config file')
    args = parser.parse_args()


    #----------Load config file------------
    config_dict = load_config_file(args.config_path)
    ckp_path = config_dict['ckp_path']
    cfg_input = config_dict['input']
    cfg_output = config_dict['output']


    #-----------Configure------------------
    if not config_dict['device']:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(f"{config_dict['device']}")


    transform_rgb = Compose([
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5],
                  std = [0.5, 0.5, 0.5])
    ])

    toPil = ToPILImage()

    #--------Init model------------
    netG = Generator(s2_in_channels=3, lc_in_channels=3, out_channels=1).to(device)
    ckp = torch.load(Path(ckp_path))
    netG.load_state_dict(ckp['netG_state_dict'])


    #-------Read input------------
    s2 = Image.open(cfg_input['s2_path']).convert('RGB')
    lc = Image.open(cfg_input['lc_path']).convert('RGB').resize(size=(256,256), resample=Image.Resampling.BICUBIC)
    s2_transformed, lc_transformed = transform_rgb(s2), transform_rgb(lc)
    s2_transformed = s2_transformed.unsqueeze(0).to(device) # Add batch dimens at first -> [B, C ,H, W]
    lc_transformed = lc_transformed.unsqueeze(0).to(device) #
    
    
    #------Infer-----------
    netG.eval()
    with torch.no_grad():
        s1_fake = netG(s2_transformed, lc_transformed) # s1_fake in [-1, 1], [B, C, H, W]
        s1_denormed = denorm(s1_fake).squeeze(0) # [0, 1], [C, H, W]
        s1_pil = toPil(s1_denormed) # [0 ,255], [H, W, C]

        s1_pil.show()


    if cfg_output['s1_path']: # Show S1 target
        s1_real = Image.open(cfg_output['s1_path']).convert('L')
        s1_real.show()
