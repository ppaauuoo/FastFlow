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


def create_dataset_structure(config):
    base_path = os.path.join(config['dataset']['output_folder'], config['dataset']['category_name'])
    
    folders = {
        'train_good': os.path.join(base_path, 'train', 'good'),
        'test_good': os.path.join(base_path, 'test', 'good'),
        'test_anomaly': os.path.join(base_path, 'test', config['dataset']['anomaly_type']),
        'ground_truth_good': os.path.join(base_path, 'ground_truth', 'good'),
        'ground_truth_anomaly': os.path.join(base_path, 'ground_truth', config['dataset']['anomaly_type'])
    }
    
    for folder in folders.values():
        os.makedirs(folder, exist_ok=True)
    
    return folders


def process_pass_images(config, folders):
    pass_folder = config['dataset']['pass_folder']
    image_files = [f for f in os.listdir(pass_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    train_split = config['dataset']['train_split']
    train_files, test_files = train_test_split(image_files, train_size=train_split, random_state=42)
    
    print(f"Processing Pass images: {len(train_files)} for train, {len(test_files)} for test")
    
    for filename in train_files:
        src = os.path.join(pass_folder, filename)
        dst = os.path.join(folders['train_good'], filename)
        shutil.copy2(src, dst)
    
    for filename in test_files:
        src = os.path.join(pass_folder, filename)
        dst = os.path.join(folders['test_good'], filename)
        shutil.copy2(src, dst)
        
        base_name = filename.rsplit('.', 1)[0]
        mask_filename = f"{base_name}_mask.png"
        mask_path = os.path.join(folders['ground_truth_good'], mask_filename)
        
        image = cv2.imread(src)
        if image is not None:
            mask = create_empty_mask(image.shape)
            cv2.imwrite(mask_path, mask)


def process_reject_images(config, folders):
    reject_folder = config['dataset']['reject_cropped_folder']
    
    if not os.path.exists(reject_folder):
        print(f"Warning: Reject cropped folder '{reject_folder}' does not exist")
        return
    
    image_files = [f for f in os.listdir(reject_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Processing Reject images: {len(image_files)} images")
    
    for filename in image_files:
        src = os.path.join(reject_folder, filename)
        dst = os.path.join(folders['test_anomaly'], filename)
        shutil.copy2(src, dst)
        
        base_name = filename.rsplit('.', 1)[0]
        mask_filename = f"{base_name}_mask.png"
        
        mask_src = os.path.join(reject_folder, 'masks', mask_filename)
        mask_dst = os.path.join(folders['ground_truth_anomaly'], mask_filename)
        
        if os.path.exists(mask_src):
            shutil.copy2(mask_src, mask_dst)
            print(f"Copied mask for {filename}")
        else:
            print(f"Warning: No mask found for {filename}")
            image = cv2.imread(src)
            if image is not None:
                mask = create_empty_mask(image.shape)
                cv2.imwrite(mask_dst, mask)


def main():
    parser = argparse.ArgumentParser(description='Create MVTec-AD format dataset')
    parser.add_argument('--config', default='configs/mask_generation.yaml', help='Path to config file')
    args = parser.parse_args()
    
    if not os.path.isfile(args.config):
        print(f"Error: Config file '{args.config}' does not exist")
        return
    
    config = load_config(args.config)
    
    pass_folder = config['dataset']['pass_folder']
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
    print(f"    ground_truth/good/ - {len(os.listdir(folders['ground_truth_good']))} masks")
    print(f"    ground_truth/{config['dataset']['anomaly_type']}/ - {len(os.listdir(folders['ground_truth_anomaly']))} masks")


if __name__ == '__main__':
    main()
