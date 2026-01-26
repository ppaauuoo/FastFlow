# Ground Truth Mask Generation for Anomaly Detection

This pipeline generates ground truth masks for anomaly detection and creates MVTec-AD format datasets.

## Workflow

1. **Generate masks** for Reject images using traditional CV methods
2. **Create dataset** in MVTec-AD format
3. **Visualize** results for validation

## Usage

### Step 1: Generate Masks

```bash
python generate_masks.py --input data/Reject/cropped --config configs/mask_generation.yaml
```

This generates masks for all images in the cropped folder and saves them to `data/Reject/cropped/masks/`.

### Step 2: Create Dataset

```bash
python create_dataset.py --config configs/mask_generation.yaml
```

This creates the dataset structure:
```
data/dataset/
└── inspection/
    ├── train/good/          # 80% of Pass images
    ├── test/good/           # 20% of Pass images
    ├── test/anomaly/        # Reject images
    └── ground_truth/
        ├── good/            # Empty masks for test/good
        └── anomaly/         # Generated masks for test/anomaly
```

### Step 3: Visualize Masks

```bash
# Visualize masks in Reject folder
python visualize_masks.py --input data/Reject/cropped --output visualizations

# Visualize complete dataset
python visualize_masks.py --input data/dataset/inspection --output visualizations --dataset
```

## Configuration

Edit `configs/mask_generation.yaml` to adjust:

- **Mask generation methods**: Enable/disable and weight different detection methods
- **Threshold method**: adaptive, otsu, or manual
- **Morphology**: Opening and closing operations
- **Dataset structure**: Folder paths, split ratio, category names

## Files

- `configs/mask_generation.yaml` - Configuration file
- `generate_masks.py` - Mask generation using traditional CV
- `create_dataset.py` - Dataset creation script
- `visualize_masks.py` - Visualization tool

## Notes

- Only 3 Reject images in current dataset - consider collecting more for training
- Traditional CV methods may need tuning for your specific patterns
- Use visualization tool to validate mask quality
