import argparse
import os
import yaml
import cv2
import numpy as np
from skimage.feature import local_binary_pattern


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def variance_filter(gray, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
    mean_sq = cv2.filter2D((gray.astype(np.float32))**2, -1, kernel)
    variance = mean_sq - mean**2
    return variance


def sobel_filter(gray):
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    return magnitude


def contrast_filter(gray, block_size):
    mean = cv2.blur(gray, (block_size, block_size))
    contrast = cv2.absdiff(gray, mean)
    return contrast


def lbp_filter(gray):
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    return lbp.astype(np.float32)


def normalize_image(img):
    if img.max() > 0:
        return (img - img.min()) / (img.max() - img.min())
    return img


def generate_mask(image, config):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    combined = np.zeros_like(gray, dtype=np.float32)
    total_weight = 0
    
    methods = config['mask_generation']['methods']
    
    if methods['variance']['enabled']:
        var = variance_filter(gray, methods['variance']['kernel_size'])
        var = normalize_image(var)
        combined += var * methods['variance']['weight']
        total_weight += methods['variance']['weight']
    
    if methods['sobel']['enabled']:
        sobel = sobel_filter(gray)
        sobel = normalize_image(sobel)
        combined += sobel * methods['sobel']['weight']
        total_weight += methods['sobel']['weight']
    
    if methods['contrast']['enabled']:
        contrast = contrast_filter(gray, 15)
        contrast = normalize_image(contrast)
        combined += contrast * methods['contrast']['weight']
        total_weight += methods['contrast']['weight']
    
    if methods['lbp']['enabled']:
        lbp = lbp_filter(gray)
        lbp = normalize_image(lbp)
        combined += lbp * methods['lbp']['weight']
        total_weight += methods['lbp']['weight']
    
    if total_weight > 0:
        combined /= total_weight
    
    combined = (combined * 255).astype(np.uint8)
    
    threshold_config = config['mask_generation']['threshold']
    
    if threshold_config['method'] == 'adaptive':
        binary = cv2.adaptiveThreshold(
            combined,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            threshold_config['block_size'],
            threshold_config['C']
        )
    elif threshold_config['method'] == 'otsu':
        _, binary = cv2.threshold(combined, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(combined, int(threshold_config['manual_value'] * 255), 255, cv2.THRESH_BINARY)
    
    morph_config = config['mask_generation']['morphology']
    kernel_size = morph_config['kernel_size']
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    if morph_config['opening']:
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    if morph_config['closing']:
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    mask = (binary > 0).astype(np.uint8)
    
    return mask


def process_reject_images(reject_folder, config):
    mask_folder = os.path.join(reject_folder, 'masks')
    os.makedirs(mask_folder, exist_ok=True)
    
    image_files = [f for f in os.listdir(reject_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    success_count = 0
    failure_count = 0
    
    for filename in image_files:
        image_path = os.path.join(reject_folder, filename)
        mask_path = os.path.join(mask_folder, filename.replace('.jpg', '_mask.png').replace('.jpeg', '_mask.png').replace('.png', '_mask.png'))
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load {filename}")
            failure_count += 1
            continue
        
        mask = generate_mask(image, config)
        cv2.imwrite(mask_path, mask * 255)
        print(f"Generated mask for {filename}")
        success_count += 1
    
    print(f"\nMask generation: {success_count} successful, {failure_count} failed")
    print(f"Masks saved to: {mask_folder}")
    
    return mask_folder


def main():
    parser = argparse.ArgumentParser(description='Generate anomaly masks for images using traditional CV methods')
    parser.add_argument('--input', required=True, help='Input folder containing images to process')
    parser.add_argument('--config', default='configs/mask_generation.yaml', help='Path to config file')
    args = parser.parse_args()
    
    if not os.path.isdir(args.input):
        print(f"Error: Input folder '{args.input}' does not exist")
        return
    
    if not os.path.isfile(args.config):
        print(f"Error: Config file '{args.config}' does not exist")
        return
    
    config = load_config(args.config)
    process_reject_images(args.input, config)


if __name__ == '__main__':
    main()
