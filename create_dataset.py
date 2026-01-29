import argparse
import os
import yaml
import cv2
import shutil
from sklearn.model_selection import train_test_split
import numpy as np


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_empty_mask(image_shape):
    return np.zeros(image_shape[:2], dtype=np.uint8)


def _read_image_shape(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    return image.shape


def resolve_cropped_folder(base_folder):
    if base_folder is None:
        return None
    cropped = os.path.join(base_folder, 'cropped')
    if os.path.isdir(cropped):
        return cropped
    return base_folder


def list_image_files(folder):
    return [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]


def create_dataset_structure(config):
    base_path = os.path.join(config['dataset']['output_folder'], config['dataset']['category_name'])
    
    folders = {
        'train_good': os.path.join(base_path, 'train', 'good'),
        'test_good': os.path.join(base_path, 'test', 'good'),
        'test_anomaly': os.path.join(base_path, 'test', config['dataset']['anomaly_type'])
    }
    
    for folder in folders.values():
        os.makedirs(folder, exist_ok=True)
    
    return folders


def process_pass_images(config, folders):
    pass_base = config['dataset']['pass_folder']
    pass_folder = resolve_cropped_folder(pass_base)
    image_files = [(pass_folder, f) for f in list_image_files(pass_folder)]
    
    overkill_base = config['dataset'].get('overkill_folder')
    if (not config.get('pass_only', False)) and overkill_base is not None:
        overkill_folder = resolve_cropped_folder(overkill_base)
        if overkill_folder is not None and os.path.isdir(overkill_folder):
            image_files.extend([(overkill_folder, f) for f in list_image_files(overkill_folder)])
    
    train_split = config['dataset']['train_split']
    train_files, test_files = train_test_split(image_files, train_size=train_split, random_state=42)
    
    print(f"Processing Pass images: {len(train_files)} for train, {len(test_files)} for test")
    
    for filename in train_files:
        src = os.path.join(filename[0], filename[1])
        out_name = filename[1]
        dst = os.path.join(folders['train_good'], out_name)
        shutil.copy2(src, dst)
    
    for filename in test_files:
        src = os.path.join(filename[0], filename[1])
        out_name = filename[1]
        dst = os.path.join(folders['test_good'], out_name)
        shutil.copy2(src, dst)


def process_reject_images(config, folders):
    dataset_cfg = config['dataset']
    reject_base = dataset_cfg.get('reject_folder')
    if reject_base is None:
        reject_base = dataset_cfg.get('reject_cropped_folder')
    reject_folder = resolve_cropped_folder(reject_base)
    if reject_folder is None:
        print("Warning: No reject folder configured")
        return
    
    if not os.path.exists(reject_folder):
        print(f"Warning: Reject cropped folder '{reject_folder}' does not exist")
        return
    
    image_files = list_image_files(reject_folder)
    print(f"Processing Reject images: {len(image_files)} images")
    
    for filename in image_files:
        src = os.path.join(reject_folder, filename)
        dst = os.path.join(folders['test_anomaly'], filename)
        shutil.copy2(src, dst)


def process_overkill_images(config, folders):
    dataset_cfg = config['dataset']
    overkill_base = dataset_cfg.get('overkill_folder')
    if overkill_base is None:
        return
    
    overkill_folder = resolve_cropped_folder(overkill_base)
    if not os.path.exists(overkill_folder):
        print(f"Warning: Overkill folder '{overkill_folder}' does not exist")
        return
    
    image_files = list_image_files(overkill_folder)
    print(f"Processing Overkill images: {len(image_files)} images (as good)")
    
    for filename in image_files:
        src = os.path.join(overkill_folder, filename)
        dst = os.path.join(folders['test_good'], filename)
        shutil.copy2(src, dst)


def main():
    parser = argparse.ArgumentParser(description='Create MVTec-AD format dataset')
    parser.add_argument('--config', default='configs/mask_generation.yaml', help='Path to config file')
    args = parser.parse_args()
    
    if not os.path.isfile(args.config):
        print(f"Error: Config file '{args.config}' does not exist")
        return
    
    config = load_config(args.config)
    
    pass_folder = resolve_cropped_folder(config['dataset']['pass_folder'])
    if not os.path.isdir(pass_folder):
        print(f"Error: Pass folder '{pass_folder}' does not exist")
        return
    
    folders = create_dataset_structure(config)
    
    print(f"Creating dataset in: {config['dataset']['output_folder']}")
    print(f"Category: {config['dataset']['category_name']}")
    print(f"Anomaly type: {config['dataset']['anomaly_type']}")
    print()
    
    process_pass_images(config, folders)
    print()
    process_reject_images(config, folders)
    
    print()
    print("Dataset creation complete!")
    print(f"Dataset structure:")
    base_path = os.path.join(config['dataset']['output_folder'], config['dataset']['category_name'])
    print(f"  {base_path}/")
    print(f"    train/good/ - {len(os.listdir(folders['train_good']))} images")
    print(f"    test/good/ - {len(os.listdir(folders['test_good']))} images")
    print(f"    test/{config['dataset']['anomaly_type']}/ - {len(os.listdir(folders['test_anomaly']))} images")


if __name__ == '__main__':
    main()
