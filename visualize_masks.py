import argparse
import os
import cv2
import numpy as np
import glob


def visualize_mask_overlay(image_path, mask_path, output_path=None):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return None
    
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Failed to load mask: {mask_path}")
        return None
    
    mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
    
    mask_colored = np.zeros_like(image)
    mask_colored[:, :, 0] = mask_resized
    
    overlay = cv2.addWeighted(image, 0.7, mask_colored, 0.3, 0)
    
    result = np.hstack([image, mask_resized[..., np.newaxis], overlay])
    
    if output_path:
        cv2.imwrite(output_path, result)
    
    return result


def visualize_dataset(dataset_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    anomaly_types = [d for d in os.listdir(os.path.join(dataset_path, 'test')) if d != 'good']
    
    for anomaly_type in anomaly_types:
        test_folder = os.path.join(dataset_path, 'test', anomaly_type)
        gt_folder = os.path.join(dataset_path, 'ground_truth', anomaly_type)
        output_anomaly_folder = os.path.join(output_folder, anomaly_type)
        os.makedirs(output_anomaly_folder, exist_ok=True)
        
        image_files = glob.glob(os.path.join(test_folder, '*.jpg')) + \
                      glob.glob(os.path.join(test_folder, '*.jpeg')) + \
                      glob.glob(os.path.join(test_folder, '*.png'))
        
        print(f"Processing {anomaly_type}: {len(image_files)} images")
        
        for image_path in image_files:
            filename = os.path.basename(image_path)
            base_name = filename.rsplit('.', 1)[0]
            mask_filename = f"{base_name}_mask.png"
            mask_path = os.path.join(gt_folder, mask_filename)
            
            if os.path.exists(mask_path):
                output_path = os.path.join(output_anomaly_folder, f"{base_name}_visualization.png")
                visualize_mask_overlay(image_path, mask_path, output_path)
                print(f"  Generated visualization for {filename}")
            else:
                print(f"  Warning: No mask found for {filename}")


def visualize_masks_simple(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    image_files = glob.glob(os.path.join(input_folder, '*.jpg')) + \
                  glob.glob(os.path.join(input_folder, '*.jpeg')) + \
                  glob.glob(os.path.join(input_folder, '*.png'))
    
    mask_folder = os.path.join(input_folder, 'masks')
    
    print(f"Processing {len(image_files)} images from {input_folder}")
    
    count = 0
    for image_path in image_files:
        filename = os.path.basename(image_path)
        base_name = filename.rsplit('.', 1)[0]
        mask_filename = f"{base_name}_mask.png"
        mask_path = os.path.join(mask_folder, mask_filename)
        
        if os.path.exists(mask_path):
            output_path = os.path.join(output_folder, f"{base_name}_visualization.png")
            visualize_mask_overlay(image_path, mask_path, output_path)
            print(f"  Generated visualization for {filename}")
            count += 1
    
    print(f"\nGenerated {count} visualizations in {output_folder}")


def main():
    parser = argparse.ArgumentParser(description='Visualize generated masks')
    parser.add_argument('--input', required=True, help='Input folder (dataset path or image folder)')
    parser.add_argument('--output', required=True, help='Output folder for visualizations')
    parser.add_argument('--dataset', action='store_true', help='Input is a MVTec-AD format dataset')
    args = parser.parse_args()
    
    if args.dataset:
        visualize_dataset(args.input, args.output)
    else:
        visualize_masks_simple(args.input, args.output)


if __name__ == '__main__':
    main()
