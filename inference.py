"""
Infer s1 output with inputs {s2, lc}
"""

from temporalgan import gen_s2_lc_v1_0, gen_s2_lc_v1_1 
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
import numpy as np


# Mapping from config model to Object
GENERATORS = {
    'gen_s2_lc_v1_0': gen_s2_lc_v1_0,
    'gen_s2_lc_v1_1': gen_s2_lc_v1_1,
}



def load_config_file(config_path):
    try:
        with open(Path(config_path), 'r') as file:
            config = yaml.safe_load(file)
            return config
    except FileNotFoundError:
        print(f'Infer config path: {config_path} is not valid')
        sys.exit(1)

def load_whitelist(whitelist_path):
    """
    Loads a whitelist file. The file should contain one filename per line.
    Returns a list of strings for substring checking.
    """
    if not whitelist_path or not os.path.exists(whitelist_path):
        print(f"Whitelist path '{whitelist_path}' not found or not specified. Skipping.")
        return None
    
    try:
        with open(whitelist_path, 'r') as f:
            # Read lines, strip whitespace
            whitelist_items = [line.split('.')[0].strip() for line in f.readlines() if line.strip().endswith('.tif')]
        print(whitelist_items)
        if not whitelist_items:
            print("Whitelist file is empty. Skipping.")
            return None
            
        return whitelist_items
    except Exception as e:
        print(f"Error reading whitelist file {whitelist_path}: {e}")
        sys.exit(1)

def denorm(img: torch.Tensor):
    """
    Denorm from [-1, 1] to [0, 1]
    """
    img = (img * 0.5 + 0.5).clamp(0, 1)
    return img


def load_npy_as_tensor(path, is_s1=False):
    """
    Mirror the formatting logic from o2s_dataset.py exactly
    """
    arr = np.load(path)
    
    # 2. Ensure Arrays have C channel to prepare for transformations (H, W, C)
    if arr.ndim == 2:
        arr = np.expand_dims(arr, axis=-1)
        
    # 4. Extract required features for SAR
    if is_s1 and arr.shape[-1] > 1:
        arr = arr[..., 0:1]
        
    # 5. Convert to PyTorch Tensors
    tensor = torch.from_numpy(arr).float()
    
    # 6. Change axis from HWC -> CHW
    tensor = tensor.permute(2, 0, 1)
    return tensor


def enable_dropout(module):
    if isinstance(module, nn.Dropout):
        module.train()


def save_raw_output(tensor: torch.Tensor, save_path: str, format: str = 'npy'):
    """
    Save raw tensor output for downstream training.
    
    Args:
        tensor: PyTorch tensor to save [C, H, W] or [B, C, H, W]
        save_path: Path to save the file (without extension)
        format: 'pt' (PyTorch), 'npy' (NumPy), or 'npz' (NumPy compressed)
    """
    if format == 'pt':
        torch.save(tensor, f"{save_path}.pt")
    elif format == 'npy':
        np.save(f"{save_path}.npy", tensor.cpu().numpy())
    elif format == 'npz':
        np.savez_compressed(f"{save_path}.npz", data=tensor.cpu().numpy())
    else:
        raise ValueError(f"Unknown format: {format}. Supported: 'pt', 'npy', 'npz'")


if __name__ == '__main__':
    parser = ArgumentParser(prog="Inference")
    parser.add_argument('--config_path', type = str, default= '/mnt/data1tb/vinh/TemporalGAN/config/inference_config.yaml', help='inference config file')
    args = parser.parse_args()


    #----------Load config file------------
    config_dict = load_config_file(args.config_path)
    ckpt_path = config_dict['ckpt_path']
    cfg_input = config_dict['input']
    cfg_output = config_dict['output']
    cfg_batch_infer = config_dict['batch_infer']
    cfg_model = config_dict['model']
    # Save raw output for downstream training
    save_raw_cfg = cfg_batch_infer.get('save_raw', {})


    #-----------Configure------------------
    if not config_dict['device']:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        try:
            device = torch.device(config_dict['device'])
        except Exception as e:
            print(f'Explain error.\nIn yaml inference_config, device: {config_dict["device"]}: {e}')
            sys.exit(1)

    #---------Init transform---------
    # For PIL Images (e.g. PNGs)
    transform_rgb = Compose([
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5],
                  std = [0.5, 0.5, 0.5])
    ])
    
    # For Numpy Arrays (already in [0, 1], no ToTensor needed)
    transform_norm = Normalize(mean=[0.5, 0.5, 0.5],
                               std=[0.5, 0.5, 0.5])

    toPil = ToPILImage()

    # Validate and load generator and discriminator from config
    gen_module_name = cfg_model['generator']    
    if gen_module_name not in GENERATORS:
        raise ValueError(f"Generator '{gen_module_name}' not found. Available: {list(GENERATORS.keys())}")
        
    generator_module = GENERATORS[gen_module_name]

    #--------Init model------------
    netG = generator_module.Generator(s2_in_channels=3, lc_in_channels=3, out_channels=1).to(device)
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
        is_npy = cfg_input['s2_path'].endswith('.npy')
        
        if is_npy:
            s2_tensor = load_npy_as_tensor(cfg_input['s2_path'])
            lc_tensor = load_npy_as_tensor(cfg_input['lc_path'])
            # Only Normalize, no ToTensor
            s2_transformed = transform_norm(s2_tensor).unsqueeze(0).to(device)
            lc_transformed = transform_norm(lc_tensor).unsqueeze(0).to(device)
        else:
            s2 = Image.open(cfg_input['s2_path']).convert('RGB')
            lc = Image.open(cfg_input['lc_path']).convert('RGB')
            lc = lc.resize(size=tuple(config_dict['new_size_lc']), resample=Image.Resampling.BICUBIC) if config_dict['new_size_lc'] else lc
            s2_transformed, lc_transformed = transform_rgb(s2), transform_rgb(lc)
            s2_transformed = s2_transformed.unsqueeze(0).to(device) # Add batch dimens at first -> [B, C ,H, W]
            lc_transformed = lc_transformed.unsqueeze(0).to(device)
        
        
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

    
        s2_folder_path = cfg_batch_infer.get('s2_folder_path')
        lc_folder_path = cfg_batch_infer.get('lc_folder_path')
        if not s2_folder_path or not lc_folder_path:
            raise ValueError('In yaml inference_config, both s2_folder_path and lc_folder_path must be specified')

        s1_folder_path = cfg_batch_infer.get('s1_folder_path')

        # --- Whitelist Logic ---
        all_s2_images = sorted(os.listdir(s2_folder_path))
        images_to_process = all_s2_images

        whitelist_path = cfg_batch_infer.get('whitelist_path')
        if whitelist_path:
            print(f"Attempting to load whitelist from: {whitelist_path}")
            whitelist_items = load_whitelist(whitelist_path)
            print(f'Loaded {len(whitelist_items)} items from whitelist')

            if whitelist_items:
                images_to_process = [
                    img for img in all_s2_images 
                    if any(wl_item in img for wl_item in whitelist_items)
                ]
                print(f"Whitelist applied: {len(images_to_process)} images will be processed out of {len(all_s2_images)} total.")
        # --- End Whitelist Logic ---

        # Len of each folders
        len_s2 = len(all_s2_images)
        len_lc = len(os.listdir(lc_folder_path))
        assert len_s2 == len_lc, f'Mismatch data length in source folders: S2({len_s2}) != LC({len_lc})'

        # The quantity of images need to infer
        data_len = cfg_batch_infer['data_len']
        total = data_len if data_len > 0 and data_len < len(images_to_process) else len(images_to_process)

        # Iteration 
        count = 0
        batch_s2 = []
        batch_lc = []
        batch_img_name = []

        for image_name in tqdm(images_to_process, total=total, desc="Inferring"):
            if count >= total:
                break

            count += 1

            # Path
            s2_path = os.path.join(s2_folder_path, image_name)
            # lc_path = os.path.join(lc_folder_path, image_name.split('.')[0]+"_colored.png")
            lc_path = os.path.join(lc_folder_path, image_name)
            s1_path = os.path.join(s1_folder_path, image_name) if s1_folder_path else None

            # Read & Transform
            is_npy = image_name.endswith('.npy')
            
            if is_npy:
                s2_tensor = load_npy_as_tensor(s2_path)
                lc_tensor = load_npy_as_tensor(lc_path)
                s1_tensor = load_npy_as_tensor(s1_path, is_s1=True) if s1_path else None
                
                s2_transformed = transform_norm(s2_tensor)
                lc_transformed = transform_norm(lc_tensor)
                
                if cfg_batch_infer['save_png']:
                    # Convert to PIL for saving input views
                    s2_img = toPil(s2_tensor)
                    lc_img = toPil(lc_tensor)
                    s1_img = toPil(s1_tensor) if s1_tensor is not None else None
            else:
                s2_img = Image.open(s2_path).convert('RGB')
                lc_img = Image.open(lc_path).convert('RGB')
                lc_img = lc_img.resize(size=tuple(config_dict['new_size_lc']), resample=Image.Resampling.BICUBIC) if config_dict['new_size_lc'] else lc_img            
                s1_img = Image.open(s1_path).convert('L') if s1_path else None

                s2_transformed, lc_transformed = transform_rgb(s2_img), transform_rgb(lc_img)
            
            # Save: If save_png: True -> save (s2, lc, sr)
            if cfg_batch_infer['save_png']:
                save_base = image_name.rsplit('.', 1)[0]
                os.makedirs(os.path.join(save_dir, save_base), exist_ok = True)
                s2_img.save(os.path.join(save_dir, save_base, 's2.png'))
                lc_img.save(os.path.join(save_dir, save_base, 'lc.png'))
                if s1_img:
                    s1_img.save(os.path.join(save_dir, save_base, 's1.png'))
            
            # Add to batch
            batch_s2.append(s2_transformed)
            batch_lc.append(lc_transformed)
            batch_img_name.append(image_name)
            
            if len(batch_s2) < cfg_batch_infer['batch_size'] and count < total:
                continue

            #-----------Infer----------
            with torch.no_grad():
                # Convert list of tensors [C, H, W] to a tensor [B, C, H, W]
                # Move batch to device here for better efficiency (reduce CPU-GPU communication overhead)
                s2_tensor_batch = torch.stack(tensors=batch_s2, dim=0).to(device)
                lc_tensor_batch = torch.stack(tensors=batch_lc, dim=0).to(device)
                
                # Forward
                s1_gen_batch = netG(s2_tensor_batch, lc_tensor_batch)
                
                # Denorm output
                s1_denormed_batch = denorm(s1_gen_batch).cpu() # Move to CPU once for the whole batch before converting to PIL
                s1_pil_list = [toPil(s1_denormed.squeeze(0)) for s1_denormed in s1_denormed_batch.chunk(chunks=len(batch_s2), dim=0)]

                # Save s1_gen
                for i in range(len(s1_pil_list)):
                    base_name = batch_img_name[i].rsplit('.', 1)[0]
                    save_path = os.path.join(save_dir, base_name, 's1_gen.png')
                    if cfg_batch_infer['save_png']: # Check to save png format
                        s1_pil_list[i].save(save_path)
                        print(f'Saved at {save_path}')
                    

                    if save_raw_cfg.get('enabled', False):
                        raw_format = save_raw_cfg.get('format', 'npy')
                        save_denormed = save_raw_cfg.get('save_denormed', False)
                        
                        # Get the tensor for this sample
                        if save_denormed:
                            raw_tensor = s1_denormed_batch[i]  # [C, H, W], range [0, 1]
                        else:
                            raw_tensor = s1_gen_batch[i].cpu()  # [C, H, W], range [-1, 1]
                        
                        raw_save_path = os.path.join(save_dir, base_name)
                        save_raw_output(raw_tensor, raw_save_path, raw_format)
                        print(f'Saved raw at {raw_save_path}.{raw_format}')

            # Reset
            batch_s2 = []
            batch_lc = []
            batch_img_name = []

            if count >= total:
                print(f'\nSuccessful infer [{count}/{total}]')
                break