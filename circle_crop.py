import argparse
import os
import yaml
import cv2
import numpy as np


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def detect_circle_at_position(image, x, y, min_radius, max_radius, param1, param2, min_dist):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    roi_size = max(min_radius * 2, 50)
    x_min = max(0, x - roi_size)
    x_max = min(image.shape[1], x + roi_size)
    y_min = max(0, y - roi_size)
    y_max = min(image.shape[0], y + roi_size)
    
    roi = gray[y_min:y_max, x_min:x_max]
    
    if roi.size == 0:
        return None
    
    circles = cv2.HoughCircles(
        roi,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=min_dist,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius
    )
    
    if circles is None:
        return None
    
    circles = np.round(circles[0, :]).astype("int")
    
    best_circle = None
    min_distance = float('inf')
    
    for (cx, cy, r) in circles:
        actual_x = cx + x_min
        actual_y = cy + y_min
        distance = np.sqrt((actual_x - x)**2 + (actual_y - y)**2)
        
        if distance < min_distance and distance < r * 0.5:
            min_distance = distance
            best_circle = (actual_x, actual_y, r)
    
    return best_circle


def crop_around_circle(image, circle):
    cx, cy, r = circle
    
    x1 = max(0, int(cx - r))
    y1 = max(0, int(cy - r))
    x2 = min(image.shape[1], int(cx + r))
    y2 = min(image.shape[0], int(cy + r))
    
    return image[y1:y2, x1:x2]


def process_images(input_folder, config):
    supported_formats = config['images']['supported_formats']
    circle_config = config['circle']
    detection_config = config['detection']
    
    image_files = []
    for filename in os.listdir(input_folder):
        if any(filename.lower().endswith(ext) for ext in supported_formats):
            image_files.append(filename)
    
    if not image_files:
        print(f"No supported images found in {input_folder}")
        return
    
    output_folder = os.path.join(input_folder, 'cropped')
    os.makedirs(output_folder, exist_ok=True)
    
    success_count = 0
    failure_count = 0
    
    for filename in image_files:
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        
        image = cv2.imread(input_path)
        if image is None:
            print(f"Failed to load {filename}")
            failure_count += 1
            continue
        
        circle = detect_circle_at_position(
            image,
            circle_config['x'],
            circle_config['y'],
            circle_config['min_radius'],
            circle_config['max_radius'],
            detection_config['param1'],
            detection_config['param2'],
            detection_config['min_dist']
        )
        
        if circle is None:
            print(f"No circle detected at ({circle_config['x']}, {circle_config['y']}) in {filename}")
            failure_count += 1
            continue
        
        cropped = crop_around_circle(image, circle)
        
        if cropped.size == 0:
            print(f"Failed to crop {filename} - empty result")
            failure_count += 1
            continue
        
        cv2.imwrite(output_path, cropped)
        print(f"Processed {filename} - circle at ({circle[0]}, {circle[1]}, r={circle[2]})")
        success_count += 1
    
    print(f"\nSummary: {success_count} successful, {failure_count} failed")
    print(f"Cropped images saved to: {output_folder}")


def main():
    parser = argparse.ArgumentParser(description='Detect and crop circles at specified positions in images')
    parser.add_argument('--input', required=True, help='Input folder containing images')
    parser.add_argument('--config', default='configs/circle_crop.yaml', help='Path to config file')
    args = parser.parse_args()
    
    if not os.path.isdir(args.input):
        print(f"Error: Input folder '{args.input}' does not exist")
        return
    
    if not os.path.isfile(args.config):
        print(f"Error: Config file '{args.config}' does not exist")
        return
    
    config = load_config(args.config)
    process_images(args.input, config)


if __name__ == '__main__':
    main()
