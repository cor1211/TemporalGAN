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
import os
from tqdm import tqdm
import torch.nn as nn

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
    Denorm from [-1, 1] to [0, 1]
    """
    img = (img * 0.5 + 0.5).clamp(0, 1)
    return img


def enable_dropout(module):
    if isinstance(module, nn.Dropout):
        module.train()


if __name__ == '__main__':
    parser = ArgumentParser(prog="Inference")
    parser.add_argument('--config_path', type = str, required = True, help='inference config file')
    args = parser.parse_args()


    #----------Load config file------------
    config_dict = load_config_file(args.config_path)
    ckpt_path = config_dict['ckpt_path']
    cfg_input = config_dict['input']
    cfg_output = config_dict['output']
    cfg_batch_infer = config_dict['batch_infer']


    #-----------Configure------------------
    if not config_dict['device']:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        try:
            device = torch.device(config_dict['device'])
        except Exception as e:
            print(f'Explain error.\nIn yaml inference_config, device: {config_dict["device"]}: {e}')
            sys.exit(1)


    transform_rgb = Compose([
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5],
                  std = [0.5, 0.5, 0.5])
    ])

    toPil = ToPILImage()

    #--------Init model------------
    netG = Generator(s2_in_channels=3, lc_in_channels=3, out_channels=1).to(device)
    ckpt = torch.load(Path(ckpt_path))
    netG.load_state_dict(ckpt['netG_state_dict'])

    netG.eval() # Set eval mode globally first (fixes BN layers)

    # Apply dropout for infer phrase
    if config_dict['dropout']:
        netG.apply(enable_dropout)

    #--------------------MAIN--------------------
    if not cfg_batch_infer['action']:
    #--------Infer 1 img----------

        #-------Read input------------
        s2 = Image.open(cfg_input['s2_path']).convert('RGB')
        lc = Image.open(cfg_input['lc_path']).convert('RGB').resize(size=(256,256), resample=Image.Resampling.BICUBIC)
        s2_transformed, lc_transformed = transform_rgb(s2), transform_rgb(lc)
        s2_transformed = s2_transformed.unsqueeze(0).to(device) # Add batch dimens at first -> [B, C ,H, W]
        lc_transformed = lc_transformed.unsqueeze(0).to(device) #
        
        
        #------Infer-----------
        with torch.no_grad():
            s1_fake = netG(s2_transformed, lc_transformed) # s1_fake in [-1, 1], [B, C, H, W]
            s1_denormed = denorm(s1_fake).squeeze(0) # [0, 1], [C, H, W]
            s1_pil = toPil(s1_denormed) # [0 ,255], [H, W, C]

            s1_pil.show()


        if cfg_output['s1_path']: # Show S1 target
            s1_real = Image.open(cfg_output['s1_path']).convert('L')
            s1_real.show()

    else:
    #-----------Infer 1 folder img---------------
        print(f'{10*"-"}Infer folder processing...{10*"-"}')
        save_dir = cfg_batch_infer['save_dir']
        if not save_dir:
            raise ValueError('In yaml inference_config -> Save_dir is not None') 
        else: 
            print(f'Images saved to folder: {save_dir}')

        os.makedirs(save_dir, exist_ok=True) # Make folder to save

    
        folder_path = cfg_batch_infer['image_folder_path']
        if not folder_path:
            raise ValueError(f'In yaml inference_config, image_folder_path: "{folder_path}" - null')

        # Path to input folders
        s2_folder_path = os.path.join(folder_path, cfg_batch_infer['input_folders'][0])
        lc_folder_path = os.path.join(folder_path, cfg_batch_infer['input_folders'][1])
        s1_folder_path = os.path.join(folder_path, cfg_batch_infer['output_folders']) if cfg_batch_infer['output_folders'] else None

        # Len of each folders
        len_s2 = len(os.listdir(s2_folder_path))
        len_lc = len(os.listdir(lc_folder_path))
        assert len_s2 == len_lc, f'Mismatch data length: S2({len_s2}) != LC({len_lc})'

        # The quantity of images need to infer
        data_len = cfg_batch_infer['data_len']
        total = data_len if data_len > 0 else len_s2

        # Iteration 
        count = 0
        for image_name in tqdm(sorted(os.listdir(s2_folder_path))):
            os.makedirs(os.path.join(save_dir, image_name.replace('.png', '')), exist_ok = True)
            count += 1
   
            # Path
            s2_path = os.path.join(s2_folder_path, image_name)
            lc_path = os.path.join(lc_folder_path, image_name)
            s1_path = os.path.join(s1_folder_path, image_name) if s1_folder_path else None

            # Open image
            s2_img = Image.open(s2_path).convert('RGB')
            lc_img = Image.open(lc_path).convert('RGB').resize(size=(256,256), resample=Image.Resampling.BICUBIC)
            s1_img = Image.open(s1_path).convert('L') if s1_path else None

            s2_transformed, lc_transformed = transform_rgb(s2_img), transform_rgb(lc_img)
            s2_transformed = s2_transformed.unsqueeze(0).to(device) # Add batch dimens at first -> [B, C ,H, W]
            lc_transformed = lc_transformed.unsqueeze(0).to(device) #
            
            # Save
            s2_img.save(os.path.join(save_dir, image_name.replace('.png', ''), 's2.png'))
            lc_img.save(os.path.join(save_dir, image_name.replace('.png', ''), 'lc.png'))
            if s1_img:
                s1_img.save(os.path.join(save_dir, image_name.replace('.png', ''), 's1.png'))

            #-----------Infer----------
            with torch.no_grad():
                s1_gen = netG(s2_transformed, lc_transformed) # Forward
                # Denorm
                s1_denormed = denorm(s1_gen.squeeze(0))
                s1_pil = toPil(s1_denormed)
                s1_pil.save(os.path.join(save_dir, image_name.replace('.png', ''), 's1_gen.png'))

            if count >= total:
                print(f'Successful infer [{count}/{total}]')
                break