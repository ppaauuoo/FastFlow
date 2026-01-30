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


def collect_good_images(config):
    """Collect all pass and overkill images together"""
    good_images = []
    
    # Collect pass images
    pass_base = config['dataset']['pass_folder']
    pass_folder = resolve_cropped_folder(pass_base)
    pass_files = [(pass_folder, f, 'pass') for f in list_image_files(pass_folder)]
    good_images.extend(pass_files)
    
    # Collect overkill images if available
    if not config['dataset'].get('pass_only', False):
        overkill_base = config['dataset'].get('overkill_folder')
        if overkill_base:
            overkill_folder = resolve_cropped_folder(overkill_base)
            if os.path.exists(overkill_folder):
                overkill_files = [(overkill_folder, f, 'overkill') for f in list_image_files(overkill_folder)]
                good_images.extend(overkill_files)
    
    return good_images


def process_good_images(config, folders):
    """Process pooled pass and overkill images with train/test split"""
    good_images = collect_good_images(config)
    
    if not good_images:
        print("No good images found")
        return
    
    train_split = config['dataset']['train_split']
    train_files, test_files = train_test_split(good_images, train_size=train_split, random_state=42)
    
    # Count pass vs overkill for reporting
    train_pass = sum(1 for f in train_files if f[2] == 'pass')
    train_overkill = sum(1 for f in train_files if f[2] == 'overkill')
    test_pass = sum(1 for f in test_files if f[2] == 'pass')
    test_overkill = sum(1 for f in test_files if f[2] == 'overkill')
    
    print(f"Processing Good images (Pass + Overkill):")
    print(f"  Train: {len(train_files)} total (pass: {train_pass}, overkill: {train_overkill})")
    print(f"  Test: {len(test_files)} total (pass: {test_pass}, overkill: {test_overkill})")
    
    # Copy train files
    for folder_path, filename, source_type in train_files:
        src = os.path.join(folder_path, filename)
        dst = os.path.join(folders['train_good'], filename)
        shutil.copy2(src, dst)
    
    # Copy test files
    for folder_path, filename, source_type in test_files:
        src = os.path.join(folder_path, filename)
        dst = os.path.join(folders['test_good'], filename)
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
    
    # Process pooled pass and overkill images
    process_good_images(config, folders)
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
