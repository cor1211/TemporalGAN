import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import yaml
from pathlib import Path
import sys
from argparse import ArgumentParser


def load_yaml_config(config_path: str) -> dict:
    try:
        with open(Path(config_path), 'r') as file:
            config = yaml.safe_load(file)
            return config
    except FileNotFoundError:
        print('Config path is not valid')
        sys.exit(1)


if __name__ == '__main__':
    parser = ArgumentParser(prog='Plot infer')
    parser.add_argument('--config_path', type=str, default = '/mnt/data1tb/vinh/TemporalGAN/config/plot_config.yaml')  
    args = parser.parse_args()

    #------------Load yaml config------------
    config_dict = load_yaml_config(args.config_path)
    root_dir = config_dict['root_dir']
    quantity = config_dict['quantity']
    save_dict = config_dict['save']
    save_action = save_dict['action']
    root_save = save_dict['root_save']
    modalities = config_dict['modalities']
    show = config_dict['show']
    quantity = quantity if quantity > 0 else 99999999999
    
    count = 0
    for subfolder in sorted(os.listdir(root_dir)):
        subfolder_path = os.path.join(root_dir, subfolder)

        if not os.path.isdir(subfolder_path):
            continue
        
        images = []
        for fn in modalities:
            if fn.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(subfolder_path, fn)
                if os.path.exists(img_path):
                    images.append(img_path)

        if not images:
            print(f"Skip {subfolder}: No images found")
            continue

        n_imgs = len(images)
        fig, axes = plt.subplots(1, n_imgs, figsize=(4 * n_imgs, 4))
        if n_imgs == 1:
            axes = [axes]

        for ax, img_path in zip(axes, images):
            img = Image.open(img_path).convert('RGB')
            # if len(np.asarray(img).shape) == 2:
            #     img = img.convert('RGB')
            ax.imshow(img)
            ax.axis("off")
            ax.text(x = 0, y= -4, s = os.path.basename(img_path).replace('.png', ''), fontsize = 14)
        
        plt.title(subfolder)
        # fig.suptitle(subfolder)

        # Setting
        plt.subplots_adjust(
        left=0, right=0.945, top=0.94, bottom=0,
        wspace=0, hspace=0
        )

        # Save fig
        if save_action:
            os.makedirs(root_save, exist_ok=True)
            save_path = os.path.join(root_save, subfolder)
            plt.savefig(save_path)
            print(f'Saved figure to {save_path}')

        # Show
        if show:
            plt.show()

        count+=1

        if count == quantity:
            break
        
    print('Finished!')