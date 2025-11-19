# 🌳 Botanical Tree Reconstruction

A Python toolkit for 3D tree reconstruction from aerial imagery using COLMAP structure-from-motion and deep learning segmentation.

## 🐆 Quick Start Guide

### 1. Installation and Setup

**Install Python dependencies:**
```bash
cd BotanicalTreeReconstruction
pip install -r requirements.txt
```

**Load COLMAP module (on HPC systems):**
```bash
module load colmap/3.8-cuda-11.8-avx512
```

**Verify COLMAP is available:**
```bash
colmap --help
```

### 2. Command-Line Interface

All scripts support `--help` to show available options:

```bash
cd /work3/s204201/BotanicalTreeReconstruction/scripts
python script_name.py --help
```

### Basic Workflow Commands

#### 1. Generate Segmentation Masks
```bash
# Generate both rough and fine masks (recommended)
python segmentation_inference.py \
    --model /path/to/trained_model.pth \
    --input ../data/raw \
    --output ../data/segmentation_masks \
    --mask-type both

# Generate only rough masks
python segmentation_inference.py \
    --model /path/to/model.pth \
    --input ../data/raw \
    --output ../data/segmentation_masks/rough \
    --mask-type rough

# Generate only fine masks
python segmentation_inference.py \
    --model /path/to/model.pth \
    --input ../data/raw \
    --output ../data/segmentation_masks/fine \
    --mask-type fine
```

#### 2. Create COLMAP 3D Model
```bash
# Basic reconstruction
python colmap_pipeline.py ../data/raw

# With custom settings
python colmap_pipeline.py ../data/raw \
    --output ../models/my_reconstruction \
    --max-features 30000 \
    --log-level INFO
```

#### 3. Filter Points with Segmentation Masks
```bash
# Using both rough and fine masks with 70% visibility threshold
python filter_colmap_with_masks.py \
    --colmap ../models/colmap_output/sparse/0 \
    --images ../data/raw \
    --rough-mask ../data/segmentation_masks \
    --fine-mask ../data/segmentation_masks \
    --visibility-threshold 0.7 \
    --combine or \
    --output ../models/filtered_model

# More conservative filtering (80% threshold)
python filter_colmap_with_masks.py \
    --colmap ../models/colmap_output/sparse/0 \
    --images ../data/raw \
    --rough-mask ../data/segmentation_masks \
    --fine-mask ../data/segmentation_masks \
    --visibility-threshold 0.8 \
    --combine and
```

### Complete Example Workflow
```bash
cd /work3/s204201/BotanicalTreeReconstruction

# 1. Add your aerial images
cp /path/to/your/drone/images/* data/raw/

# 2. Generate segmentation masks (both types)
cd scripts
python segmentation_inference.py \
    --model /path/to/deeplabv3_model.pth \
    --input ../data/raw \
    --output ../data/segmentation_masks \
    --mask-type both

# 3. Create initial COLMAP reconstruction  
python colmap_pipeline.py ../data/raw \
    --output ../models/initial_reconstruction \
    --max-features 25000

# 4. Filter using vegetation masks
python filter_colmap_with_masks.py \
    --colmap ../models/initial_reconstruction/sparse/0 \
    --images ../data/raw \
    --rough-mask ../data/segmentation_masks \
    --fine-mask ../data/segmentation_masks \
    --visibility-threshold 0.7 \
    --output ../models/filtered_tree_model

# 5. Results are in models/filtered_tree_model/
```

## 📁 Project Structure

```
BotanicalTreeReconstruction/
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── configs/              # Configuration files
├── data/                 # Data directory
│   ├── raw/              # Original images
│   └── segmentation_masks/  # Segmentation masks
│       ├── rough/        # Rough segmentation masks  
│       └── fine/         # Fine segmentation masks
├── models/               # COLMAP models and outputs
└── scripts/              # Processing scripts
    ├── colmap_pipeline.py   # COLMAP reconstruction script
    ├── filter_colmap_with_masks.py  # Point filtering with masks
    ├── segmentation_inference.py   # Generate segmentation masks
    └── README.md           # Scripts documentation
```

## 🛠️ Prerequisites

- Python 3.7+
- COLMAP installed and available in PATH
- CUDA-capable GPU (recommended)
- PyTorch and torchvision (for segmentation)

## 📖 Available Scripts

### segmentation_inference.py

Generates tree segmentation masks from aerial images using DeepLabV3.

**Key Arguments:**
- `--model`: Path to trained segmentation model (.pth file) 
- `--input`: Directory containing input images
- `--output`: Output directory for masks
- `--mask-type`: `rough` (fast, full-image), `fine` (detailed, patch-based), or `both` (default: both)
- `--tile-size`: Patch size for fine masks (default: 512)
- `--overlap`: Overlap between patches for fine masks (default: 64)
- `--input-size`: Input size for rough masks (default: 512)

**Note:** The script automatically uses:
- **Full-image mode** for rough masks (fast, lower resolution)
- **Patch-based mode** for fine masks (detailed, high resolution)

### colmap_pipeline.py

Creates a COLMAP 3D reconstruction model from an image directory.

**Key Arguments:**
- `image_dir`: Path to directory containing input images (required)
- `-o, --output`: Output directory (default: ../models/colmap_output)
- `--max-features`: Max features per image (default: 20000)
- `--log-level`: Logging level (DEBUG, INFO, WARNING, ERROR)

### filter_colmap_with_masks.py

Filters COLMAP 3D points using segmentation masks with visibility-based thresholding.

**Key Arguments:**
- `--colmap`: Path to COLMAP model directory (required)
- `--images`: Path to original images directory (required)
- `--rough-mask`: Path to rough segmentation masks (required)
- `--fine-mask`: Path to fine segmentation masks (required)
- `--output`: Output directory for filtered model (default: auto-generated)
- `--visibility-threshold`: Only keep points visible in ≥this fraction of images (default: 0.7)
- `--threshold`: Mask pixel threshold value (default: 10)
- `--combine`: Mask combination: "or" or "and" (default: or)
- `--examples`: Number of visualization examples (default: 5)

## 📊 Output

The reconstruction creates:
```
models/colmap_output/
├── database.db          # COLMAP feature database
└── sparse/              # 3D reconstruction
    └── 0/               # Model files
        ├── cameras.bin  # Camera parameters
        ├── images.bin   # Image poses
        └── points3D.bin # 3D points
```

## 📝 Notes

- The script uses GPU acceleration by default when available
- Processing time depends on number and resolution of images
- Large image sets may require significant memory
- For best results, use images with good overlap and coverage
- Rough masks use full-image inference (fast)
- Fine masks use patch-based inference (detailed)
- The visibility threshold in filtering helps preserve structurally important points

