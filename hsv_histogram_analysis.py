import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_hsv_histograms(image_folders, images_per_plot=30):
    """
    Plot HSV histograms from images in the specified folders.
    Creates multiple plots with images_per_plot images each.
    
    Args:
        image_folders: List of folder paths containing images
        images_per_plot: Number of images per plot (default: 30)
    """
    image_paths = []
    for folder in image_folders:
        folder_path = Path(folder)
        if folder_path.exists():
            image_paths.extend(list(folder_path.glob("*.jpg")))
            image_paths.extend(list(folder_path.glob("*.png")))
    
    total_images = len(image_paths)
    print(f"Processing {total_images} images...")
    
    n_cols = 5
    n_rows = 6
    
    all_h_hists = []
    all_s_hists = []
    all_v_hists = []
    
    num_plots = int(np.ceil(total_images / images_per_plot))
    
    for plot_num in range(num_plots):
        start_idx = plot_num * images_per_plot
        end_idx = min(start_idx + images_per_plot, total_images)
        batch_images = image_paths[start_idx:end_idx]
        
        print(f"\nPlot {plot_num + 1}/{num_plots}: Processing images {start_idx + 1} to {end_idx}")
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 24))
        fig.suptitle(f'HSV Histograms - Batch {plot_num + 1}/{num_plots} (Images {start_idx + 1}-{end_idx})', 
                     fontsize=16, fontweight='bold')
        
        for idx, img_path in enumerate(batch_images):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]
            
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            h_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180]).flatten()
            s_hist = cv2.calcHist([hsv], [1], None, [256], [0, 256]).flatten()
            v_hist = cv2.calcHist([hsv], [2], None, [256], [0, 256]).flatten()
            
            h_hist = h_hist / h_hist.sum()
            s_hist = s_hist / s_hist.sum()
            v_hist = v_hist / v_hist.sum()
            
            all_h_hists.append(h_hist)
            all_s_hists.append(s_hist)
            all_v_hists.append(v_hist)
            
            ax.plot(h_hist, 'r-', label='H', alpha=0.8)
            ax.plot(np.linspace(0, 180, 256), s_hist, 'g-', label='S', alpha=0.8)
            ax.plot(np.linspace(0, 180, 256), v_hist, 'b-', label='V', alpha=0.8)
            ax.set_title(img_path.name[:15], fontsize=8)
            ax.set_xlim([0, 180])
            ax.tick_params(axis='both', labelsize=6)
            
            if idx == 0:
                ax.legend(fontsize=6)
        
        for idx in range(len(batch_images), n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        output_filename = f'hsv_histograms_batch_{plot_num + 1}.png'
        plt.savefig(output_filename, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_filename}")
        plt.show()
    
    print(f"\nTotal images processed: {len(all_h_hists)}")


if __name__ == "__main__":
    crop_folders = [
        r"data\Overkill\cropped",
        r"data\Pass\cropped",
        r"data\Reject\cropped"
    ]
    
    plot_hsv_histograms(crop_folders)
    
    print("\n" + "="*50)
    print("Analysis Complete!")
    print("="*50)
