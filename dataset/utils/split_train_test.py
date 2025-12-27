import random
import os
import sys
import shutil
import re
from collections import defaultdict
import random


def load_black_list(black_list_path) -> list:
    black_list = []
    try:
        with open(black_list_path, 'r') as file:
            for line in file.readlines():
                black_list.append(os.path.basename(line.strip()).split('.')[0])
        return black_list
    
    except FileNotFoundError as e:
        print(f'{e}: Black list path isnt valid')
        sys.exit(1)

def in_black_list(filename ,black_list):
    for fn in black_list:
        if fn in filename:
            return True
    return False


def count_quantity_file(source_data, black_list:list):
    count = 0
    flag = 0
    for fn in os.listdir(source_data):
        flag = 0
        for ban in black_list:
            if ban in fn: flag = 1
        if not flag:
            count+=1
    return count
    
def split_train_test(train_percent, valid_percent, source_data, target_data, black_list):
    # test_percent = 1 - train_percent - valid_percent
    MODALTITES = ['s1_256_vvv', 's2_256', 'lc_2048']

    # create folders
    for split in ['train', 'test', 'valid']:
        for m in MODALTITES:
            os.makedirs(os.path.join(target_data, split, m), exist_ok=True)
    
    # collect images by sample
    samples = defaultdict(dict)
    for m in MODALTITES:
        folder = os.path.join(source_data, m)
        for fn in os.listdir(folder):
            if  not fn.endswith('.png') or in_black_list(fn, black_list):
                continue
            if m == 'lc_2048':
                sample_id = fn.split('_colored')[0]+'.png'
            else:
                sample_id = fn
            
            samples[sample_id][m] = sample_id
    
    # check
    for key, value in samples.items():
        if len(value) != 3:
            print(f'Error collects image by sample at {key}')
    

    sample_label = {}
    for sample_id in samples:
        name = sample_id.replace('.png', "")
        match = re.search(r'([A-Z].*)', name)
        if not match:
            raise ValueError(f'Cannot get label from {sample_id}')
        sample_label[sample_id] = match.group(1)

    label_samples = defaultdict(list)
    for sid, label in sample_label.items():
        label_samples[label].append(sid)
    
    splits = {
        'train': [],
        'valid': [],
        'test': []
    }
    for label, sample_ids in label_samples.items():
        random.shuffle(sample_ids)
        n = len(sample_ids)
        t_end = int(n * train_percent)
        v_end = int(n * (train_percent + valid_percent))

        splits['train'].extend(sample_ids[:t_end])
        splits['valid'].extend(sample_ids[t_end:v_end])
        splits['test'].extend(sample_ids[v_end:])


    for split, sample_ids in splits.items():
        for sid in sample_ids:
            for m, fname in samples[sid].items():
                if m == 'lc_2048':
                    src = os.path.join(source_data, m, fname.split('.')[0]+'_colored.png')
                else:
                    src = os.path.join(source_data, m, fname)
                dst = os.path.join(target_data, split, m, fname)
                print(f'Copy from {src} to {dst}')
                shutil.copy(src, dst)

if __name__ == '__main__':
    random.seed(42)
    BLACK_LIST_PATH = '/mnt/data1tb/vinh/TemporalGAN/dataset/o2s/not_corresponding_dimens.txt'
    black_list = load_black_list(BLACK_LIST_PATH)
    source_data = '/mnt/data1tb/vinh/s2_s1_lc_O2S'
    target_data = '/mnt/data1tb/vinh/TemporalGAN/dataset/o2s'

    split_train_test(train_percent=0.7, valid_percent=0.2, source_data=source_data, target_data=target_data, black_list=black_list)

    
    # print(count_quantity_file(s1_root, black_list))

# import os
# import re
# import random
# import shutil
# from collections import defaultdict

# # ================= CONFIG =================
# BASE_DIR = "/mnt/data1tb/vinh/s2_s1_lc_O2S"        # chứa s1/, s2/, lc/
# TARGET_DIR = "/mnt/data1tb/vinh/TemporalGAN/dataset/o2s"

# MODALITIES = ["s1_256_vvv", "s2_256", "lc_2048"]
# IMAGE_EXT = ".png"

# TRAIN_RATIO = 0.7
# VALID_RATIO = 0.15
# TEST_RATIO  = 0.15

# RANDOM_SEED = 42
# # ==========================================

# random.seed(RANDOM_SEED)

# # tạo folder output
# for split in ["train", "valid", "test"]:
#     for m in MODALITIES:
#         os.makedirs(os.path.join(TARGET_DIR, split, m), exist_ok=True)

# # =====================================================
# # 1. Gom ảnh theo SAMPLE (theo TÊN FILE)
# # =====================================================
# samples = defaultdict(dict)
# # sample_id (filename) -> {s1: file, s2: file, lc: file}

# for m in MODALITIES:
#     folder = os.path.join(BASE_DIR, m)

#     for fname in os.listdir(folder):
#         if not fname.endswith(IMAGE_EXT):
#             continue
#         if m == 'lc_2048':
#             sample_id = fname.split('_colored')[0]+'.png'
#         else:
#             sample_id = fname  # FULL filename, không cắt gì cả
#         samples[sample_id][m] = fname

# # kiểm tra sample thiếu modality
# for sid, files in samples.items():
#     if len(files) != 3:
#         print(f"Sample thiếu ảnh: {sid} -> {files}")

# # =====================================================
# # 2. Lấy LABEL cho mỗi sample
# #    (từ chữ IN HOA đầu tiên tới hết)
# # =====================================================
# sample_labels = {}

# for sample_id in samples:
#     name = sample_id.replace(IMAGE_EXT, "")
#     match = re.search(r'([A-Z].*)', name)
#     if not match:
#         raise ValueError(f"❌ Không tìm thấy label trong {sample_id}")
#     sample_labels[sample_id] = match.group(1)

# # =====================================================
# # 3. Gom sample theo label
# # =====================================================
# label_samples = defaultdict(list)
# for sid, label in sample_labels.items():
#     label_samples[label].append(sid)

# # =====================================================
# # 4. Chia train / valid / test (stratified theo label)
# # =====================================================
# splits = {"train": [], "valid": [], "test": []}

# for label, sample_ids in label_samples.items():
#     random.shuffle(sample_ids)

#     n = len(sample_ids)
#     t_end = int(n * TRAIN_RATIO)
#     v_end = int(n * (TRAIN_RATIO + VALID_RATIO))

#     splits["train"].extend(sample_ids[:t_end])
#     splits["valid"].extend(sample_ids[t_end:v_end])
#     splits["test"].extend(sample_ids[v_end:])

#     print(f"{label}: {n} samples")

# # =====================================================
# # 5. Copy ảnh ra folder đích
# # =====================================================
# for split, sample_ids in splits.items():
#     for sid in sample_ids:
#         for m, fname in samples[sid].items():
#             src = os.path.join(BASE_DIR, m, fname)
#             dst = os.path.join(TARGET_DIR, split, m, fname)
#             shutil.copy(src, dst)

# print("\n✅ Hoàn tất chia dataset!")


