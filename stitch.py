import torch
from temporalgan.gen_s2_lc_v1_0 import Generator
from argparse import ArgumentParser
from tqdm import tqdm
import yaml
from pathlib import Path
import sys
import rasterio
from rasterio.plot import show
import numpy as np
from PIL import Image
from torchvision import transforms as F

def load_yaml_config(config_path:str)->dict:
    try:
        with open(Path(config_path), 'r') as file:
            config = yaml.safe_load(file)
            return config
    except FileNotFoundError:
        print(f'Config path is not valid')
        sys.exit(1)


def read_tiff(img_path):
    with rasterio.open(img_path) as img:
        # print(f'Width: {img.width}')
        # print(f'Height: {img.height}')
        # print(f'Number of bands: {img.count}')
        # print(f'Order of bands {img.colorinterp}')
        obj = img.read()
        array = np.array(obj)
        # print(f'Shape: {array.shape}')
        # print(f'Data type: {array.dtype}')
        # for _ in range(array.shape[0]):
        #     print(f'Band {_+1}: Max = {max(array[_].reshape(-1))}, Min = {min(array[_].reshape(-1))}')

    return array


def infer_large_image(model, device, s2_tensor: torch.Tensor, lc_tensor: torch.Tensor, whole_patch: int = 256, overlap: int = 32):
    assert s2_tensor.shape == lc_tensor.shape, f'Mismatch shape of S2: {s2_tensor.shape} and LC: {lc_tensor.shape}'
    real_patch = whole_patch - 2 * overlap
    model.to(device)
    s2_tensor.to(device)
    lc_tensor.to(device)

    #-------Make null output canvas------
    B, C, H, W = s2_tensor.shape
    output = torch.zeros(size=(B, C, H, W), device='cpu')

    for y in range(0, H, real_patch):
        for x in range(0, W, real_patch):
            y_start = max(0, y - overlap)
            x_start = max(0, x - overlap)
            y_end = min(H, y + real_patch + overlap)
            x_end = min(W, x + real_patch + overlap)

            s2_patch = s2_tensor[:, :, y_start: y_end, x_start: x_end]
            lc_patch = lc_tensor[:, :, y_start: y_end, x_start: x_end]

            #----------------Forward---------------
            model.eval()
            with torch.no_grad():
                output_patch = model(s2_patch, lc_patch)         

            in_y_rel_start = 0 if y == 0 else overlap
            in_x_rel_start = 0 if x == 0 else overlap
            in_y_rel_end = s2_patch.shape[2] if y + real_patch + overlap >= H else s2_patch.shape[2] - overlap
            in_x_rel_end = s2_patch.shape[3] if x + real_patch + overlap >= W else s2_patch.shape[3] - overlap

            output_real_patch = output_patch[:, :, in_y_rel_start: in_y_rel_end, in_x_rel_start: in_x_rel_end]
            
            h_out_real_patch, w_out_real_patch = output_real_patch.shape[2], output_real_patch.shape[3]
            output[:, :, y: y+h_out_real_patch, x: x+w_out_real_patch] = output_real_patch.cpu()
    
    return output



if __name__ == '__main__':
    parser = ArgumentParser(prog = "Stitching Argument")
    parser.add_argument('--config_path', type=str, default = '/mnt/data1tb/vinh/TemporalGAN/config/stitch_config.yaml')
    args = parser.parse_args()


    #----------Load yaml config--------
    config = load_yaml_config(args.config_path)
    whole_patch = config['whole_patch']
    overlap = config['overlap']
    s2_path = config['s2_path']
    lc_path = config['lc_path']
    ckpt_path = config['ckpt_path']


    #---------Init device----------
    if not config['device']:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        try:
            device = torch.device(config['device'])
        except:
            print(f'Error! Check your device configure: {config["device"]} in stitch_config.yaml')
            sys.exit(1)


    #-----------Load checkpoint----------
    if not ckpt_path:
        print('Must be put checkpoint at yaml stitch config')
        sys.exit(1)
    try:
        checkpoint = torch.load(Path(ckpt_path), weights_only=True)
    except FileNotFoundError:
        print(f'Checkpoint path: {ckpt_path} is not valid at yaml stitch config')
        sys.exit(1)
    

    #--------Init model----------
    netG = Generator(s2_in_channels = 3, lc_in_channels = 3, out_channels = 1, features = 64)
    netG.load_state_dict(checkpoint['netG_state_dict'])

    # output = infer_large_image(netG, device, input, whole_patch, overlap)
    lc_arr = read_tiff(lc_path)[0:3, :, :]
    print(lc_arr.shape)

    s2_array = read_tiff(s2_path)
    print(s2_array.shape)

    lc_tensor = F.ToTensor()(lc_arr).unsqueeze(0).transpose(0, 2, 3, 1)
    print(type(lc_tensor))
    print(lc_tensor.shape)
    # lc_tensor = F.Resize(size = (lc_tensor.shape[2] // 8, lc_tensor.shape[3]// 8), interpolation=F.InterpolationMode.BICUBIC)
    # print(lc_tensor.shape)
    # s2_tensor = F.ToTensor()(s2_array).unsqueeze(0)
    # print(s2_tensor.shape)
    