import os
import shutil
import random

def split_dataset(
    source_dir,
    output_dir,
    train_ratio=0.7,
    val_ratio=0.2,
    test_ratio=0.1
):
    classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    
    for cls in classes:
        cls_source = os.path.join(source_dir, cls)
        images = [f for f in os.listdir(cls_source) if os.path.isfile(os.path.join(cls_source, f))]
        random.shuffle(images)
        
        n_total = len(images)
        n_train = int(train_ratio * n_total)
        n_val = int(val_ratio * n_total)
        n_test = n_total - n_train - n_val  # 保證全用上

        splits = {
            'train': images[:n_train],
            'val': images[n_train:n_train + n_val],
            'test': images[n_train + n_val:]
        }

        for split_name, split_files in splits.items():
            split_dir = os.path.join(output_dir, split_name, cls)
            os.makedirs(split_dir, exist_ok=True)
            for fname in split_files:
                src = os.path.join(cls_source, fname)
                dst = os.path.join(split_dir, fname)
                shutil.copyfile(src, dst)

if __name__ == '__main__':
    random.seed(42)  # 為了可重現
    source_folder = r'D:/ultralytics/data/classification/PetImages'     # 原始路徑
    output_folder = r'D:/ultralytics/data/classification'      # 輸出路徑
    split_dataset(source_folder, output_folder)