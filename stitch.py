import torch
from temporalgan.gen_s2_lc_v1_0 import Generator
from argparse import ArgumentParser
# from tqdm import tqdm
import yaml
from pathlib import Path
import sys
import rasterio
# from rasterio.plot import show
import numpy as np
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F


def enable_dropout(module):
    """
    Enable dropout in Inference phrase
    Use after model.eval()
    """
    if isinstance(module, torch.nn.Dropout):
        module.train()


def denorm(tensor: torch.Tensor)->torch.Tensor:
    """
    Docstring for denorm
    
    :param tensor: [B, C, H, W] / [C, H, W] / [H, W] & [-1, 1]
    :type tensor: torch.Tensor
    :return: [B, C, H, W] & [0, 1]
    :rtype: Tensor
    """
    return (tensor * 0.5 + 0.5).clamp(0, 1)



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


def infer_large_image(model, device, s2_tensor: torch.Tensor, lc_tensor: torch.Tensor, whole_patch: int = 256, overlap: int = 32, dropout: bool = False):
    assert s2_tensor.shape == lc_tensor.shape, f'Mismatch shape of S2: {s2_tensor.shape} and LC: {lc_tensor.shape}'
    real_patch = whole_patch - 2 * overlap
    model.to(device)
    s2_tensor = s2_tensor.to(device)
    lc_tensor = lc_tensor.to(device)
    
    # Set model to eval mode to infers
    model.eval()
    if dropout:
        model.apply(enable_dropout)

    #-------Make null output canvas------
    B, C, H, W = s2_tensor.shape
    output = torch.zeros(size=(B, C, H, W), device='cpu')
    count = 0 # Debug
    for y in range(0, H, real_patch):
        for x in range(0, W, real_patch):

            shift_y_start = False
            shift_x_start = False

            y_start = max(0, y - overlap)
            x_start = max(0, x - overlap)
            y_end = min(H, y_start + whole_patch)
            x_end = min(W, x_start + whole_patch)

            print(f'y_start: {y_start}, y_end: {y_end}')
            print(f'x_start: {x_start}, x_end: {x_end}')

            if y_end == H:
                shift_y_start = True
                y_start = H - whole_patch
                print(f'Change y_start: {y_start}')

            if x_end == W:  
                shift_x_start = True
                x_start = W - whole_patch
                print(f'Change x_start: {x_start}')

            s2_patch = s2_tensor[:, :, y_start: y_end, x_start: x_end]
            lc_patch = lc_tensor[:, :, y_start: y_end, x_start: x_end]
            print(f'The shape of S2_patch: {s2_patch.shape}')
            print(f'The shape of LC_patch: {lc_patch.shape}')

            if s2_patch.shape[2] != 256 or s2_patch.shape[3] != 256:
                print('SKIP')
                continue


            #----------------Forward---------------
            with torch.no_grad():
                output_patch = model(s2_patch, lc_patch)      

            output_pil = transforms.ToPILImage()(denorm(output_patch.squeeze(0)))
            # output_pil.show()
            count += 1 # Debug
            print(f'DONE {count}') # Debug

            in_y_rel_start = 0 if y == 0 else overlap
            in_x_rel_start = 0 if x == 0 else overlap
            in_y_rel_end = s2_patch.shape[2] if y + real_patch >= H else s2_patch.shape[2] - overlap
            in_x_rel_end = s2_patch.shape[3] if x + real_patch >= W else s2_patch.shape[3] - overlap

            if shift_y_start:
                in_y_rel_start = y - y_start
            if shift_x_start:
                in_x_rel_start = x - x_start

            output_real_patch = output_patch[:, :, in_y_rel_start: in_y_rel_end, in_x_rel_start: in_x_rel_end]
            
            h_out_real_patch, w_out_real_patch = output_real_patch.shape[2], output_real_patch.shape[3]
            output[:, :, y: y+h_out_real_patch, x: x+w_out_real_patch] = output_real_patch.cpu()

            # Show
            # output_real_pil = transforms.ToPILImage()(denorm(output_real_patch.squeeze(0)))
            # output_real_pil.show()
    return output




def infer_large_image_gemini(model, device, s2_tensor: torch.Tensor, lc_tensor: torch.Tensor, whole_patch: int = 256, overlap: int = 32, dropout: bool = False):
    """
    Inference ảnh lớn bằng chiến lược Sliding Window.
    Fix lỗi: Hardcode, Small Image Crash, và tối ưu vùng ghép nối.
    """
    assert s2_tensor.shape == lc_tensor.shape, f'Mismatch shape: S2 {s2_tensor.shape}, LC {lc_tensor.shape}'
    
    # 1. Handle trường hợp ảnh nhỏ hơn kích thước patch
    # Nếu ảnh nhỏ hơn patch, ta pad ảnh để đủ ít nhất 1 patch
    orig_H, orig_W = s2_tensor.shape[2], s2_tensor.shape[3]
    pad_h = max(0, whole_patch - orig_H)
    pad_w = max(0, whole_patch - orig_W)
    
    if pad_h > 0 or pad_w > 0:
        # Pad reflect hoặc constant tùy bài toán (ở đây dùng constant 0 cho an toàn)
        s2_tensor = torch.nn.functional.pad(s2_tensor, (0, pad_w, 0, pad_h))
        lc_tensor = torch.nn.functional.pad(lc_tensor, (0, pad_w, 0, pad_h))
    
    # Cập nhật lại H, W sau khi pad (nếu có)
    B, C, H, W = s2_tensor.shape
    
    real_patch = whole_patch - 2 * overlap
    stride = real_patch # Bước nhảy

    model.to(device)
    model.eval()
    if dropout:
        model.apply(enable_dropout)

    # Output canvas (giữ kích thước gốc ban đầu)
    output = torch.zeros(size=(B, C, orig_H, orig_W), device='cpu') 
    
    # Move input to device one time if memory allows, otherwise keep inside loop
    # Ở đây giả sử ảnh rất lớn nên để input ở CPU, cắt patch mới đẩy lên GPU
    
    # Lưới toạ độ
    y_grids = list(range(0, H - whole_patch + 1, stride))
    if (H - whole_patch) % stride != 0:
        y_grids.append(H - whole_patch) # Thêm patch cuối cùng (Shift logic)

    x_grids = list(range(0, W - whole_patch + 1, stride))
    if (W - whole_patch) % stride != 0:
        x_grids.append(W - whole_patch) # Thêm patch cuối cùng (Shift logic)

    for y in y_grids:
        for x in x_grids:
            # 2. Cắt patch (Input luôn đảm bảo đúng size whole_patch nhờ logic grid ở trên)
            s2_patch = s2_tensor[:, :, y : y + whole_patch, x : x + whole_patch].to(device)
            lc_patch = lc_tensor[:, :, y : y + whole_patch, x : x + whole_patch].to(device)

            # Forward
            with torch.no_grad():
                output_patch = model(s2_patch, lc_patch)
                output_patch = output_patch.cpu() # Đưa về CPU ngay để ghép

            # 3. Tính toán vùng Valid (vùng trung tâm) để ghép
            # Mặc định lấy phần giữa, bỏ qua overlap
            valid_y_start_src = overlap
            valid_y_end_src = whole_patch - overlap
            valid_x_start_src = overlap
            valid_x_end_src = whole_patch - overlap

            # Xử lý biên (Edge Cases): Giữ lại phần rìa nếu là patch đầu hoặc cuối
            if y == 0: 
                valid_y_start_src = 0
            if y == y_grids[-1]: # Patch cuối (Bottom edge)
                valid_y_end_src = whole_patch

            if x == 0:
                valid_x_start_src = 0
            if x == x_grids[-1]: # Patch cuối (Right edge)
                valid_x_end_src = whole_patch

            # 4. Trích xuất vùng Valid từ Output Patch
            output_valid = output_patch[:, :, valid_y_start_src:valid_y_end_src, valid_x_start_src:valid_x_end_src]

            # 5. Tính vị trí đặt vào Canvas lớn
            # Vị trí trên canvas = vị trí patch (y) + offset valid (valid_y_start_src)
            # Cần clamp lại để không ghi ra ngoài kích thước gốc (orig_H, orig_W) do padding ban đầu
            
            y_start_canvas = y + valid_y_start_src
            x_start_canvas = x + valid_x_start_src
            
            h_valid, w_valid = output_valid.shape[2], output_valid.shape[3]
            
            # Cắt bớt phần thừa nếu nó lòi ra khỏi ảnh gốc (do padding lúc đầu)
            valid_h_slice = slice(0, min(h_valid, orig_H - y_start_canvas))
            valid_w_slice = slice(0, min(w_valid, orig_W - x_start_canvas))
            
            if valid_h_slice.stop <= 0 or valid_w_slice.stop <= 0: continue

            output[:, :, y_start_canvas : y_start_canvas + valid_h_slice.stop, 
                         x_start_canvas : x_start_canvas + valid_w_slice.stop] = \
                output_valid[:, :, valid_h_slice, valid_w_slice]

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
    dropout = config['dropout']

    #---------Init device----------
    if not config['device']:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        try:
            device = torch.device(config['device'])
        except:
            print(f'Error! Check your device configure: {config["device"]} in stitch_config.yaml')
            sys.exit(1)
    print(f'{20 * "-"}\nUse device: {device}\n{20 * "-"}')


    #-----------Load checkpoint----------
    if not ckpt_path:
        print('Must be put checkpoint at yaml stitch config')
        sys.exit(1)
    try:
        checkpoint = torch.load(Path(ckpt_path), weights_only=False)
    except FileNotFoundError:
        print(f'Checkpoint path: {ckpt_path} is not valid at yaml stitch config')
        sys.exit(1)
    

    #--------Init model----------
    netG = Generator(s2_in_channels = 3, lc_in_channels = 3, out_channels = 1, features = 64)
    netG.load_state_dict(checkpoint['netG_state_dict'])

    toPil = transforms.ToPILImage()

    # output = infer_large_image(netG, device, input, whole_patch, overlap)
    lc_arr = read_tiff(lc_path)[0:3, :, :]
    print(lc_arr.shape)
    s2_array = read_tiff(s2_path)
    print(s2_array.shape)

    # Convert to Tensor [C, H, W] - [0, 1]
    s2_tensor = torch.Tensor(np.float32(s2_array/255.0)).unsqueeze(0)
    lc_tensor = torch.Tensor(np.float32(lc_arr/255.0)).unsqueeze(0)
    
    # Normalize input to [-1, 1] to match training/inference logic
    norm = transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std = [0.5, 0.5, 0.5]
    )
    s2_tensor = norm(s2_tensor)
    lc_tensor = norm(lc_tensor)

    # lc_tensor = torch.resize_as_(input=lc_tensor, the_template=s2_tensor) # Downsample lc
    lc_tensor = F.interpolate(input=lc_tensor, size = (s2_tensor.shape[2], s2_tensor.shape[3]), mode= 'bicubic' )
    print(f'Shape of S2_tensor: {s2_tensor.shape}')
    print(f'Shape of LC_tensor: {lc_tensor.shape}')

    # Show
    # s2_pil = toPil(s2_tensor.squeeze(0))
    # lc_pil = toPil(lc_tensor.squeeze(0))

    # s2_pil.show()
    # lc_pil.show()

    #------------Infer large image-------------
    output = infer_large_image_gemini(netG, device, s2_tensor, lc_tensor, whole_patch, overlap, dropout)
    output_denormed = denorm(output).squeeze(0) # -> [C, H, W] & [0, 1]
    output_pil = toPil(output_denormed) # ->[H, W, C] & [0, 255]
    output_pil.show()

    # output = infer_large_image(netG, device, s2_tensor, lc_tensor, whole_patch, overlap, dropout)
    # output_denormed = denorm(output).squeeze(0) # -> [C, H, W] & [0, 1]
    # output_pil = toPil(output_denormed) # ->[H, W, C] & [0, 255]
    # output_pil.show()

    