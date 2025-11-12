# 🌳 Botanical Tree Reconstruction

A Python toolkit for 3D tree reconstruction from aerial imagery using COLMAP structure-from-motion.

## 📁 Project Structure

```
BotanicalTreeReconstruction/
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── configs/              # Configuration files
├── data/                 # Data directory
│   ├── raw/              # Original images
│   └── segmentation_masks/  # Segmentation masks
├── models/               # COLMAP models and outputs
└── scripts/              # Processing scripts
    ├── colmap_pipeline.py   # COLMAP reconstruction script
    ├── filter_colmap_with_masks.py  # Point filtering with masks
    └── README.md           # Scripts documentation
```

## 🐆 Quick Start

### Prerequisites

- Python 3.7+
- COLMAP installed and available in PATH
- CUDA-capable GPU (recommended)

On HPC systems, load COLMAP:
```bash
module load colmap/3.8-cuda-11.8-avx512
```

### Basic Usage

1. **Place your aerial images** in the `data/raw/` directory

2. **Run COLMAP reconstruction**:
   ```bash
   cd scripts
   python colmap_pipeline.py /path/to/your/images
   ```

3. **View results** in the `models/` directory

## 📖 Available Scripts

### colmap_pipeline.py

Creates a COLMAP 3D reconstruction model from an image directory.

**Basic usage:**
```bash
python colmap_pipeline.py /path/to/images
```

**With custom settings:**
```bash
python colmap_pipeline.py /path/to/images \
    --output ../models/my_model \
    --max-features 30000 \
    --log-level INFO
```

**Arguments:**
- `image_dir`: Path to directory containing input images (required)
- `-o, --output`: Output directory (default: ../models/colmap_output)
- `--max-features`: Max features per image (default: 20000)
- `--log-level`: Logging level (DEBUG, INFO, WARNING, ERROR)

### filter_colmap_with_masks.py

Filters COLMAP 3D points using segmentation masks to remove points outside vegetation areas.

**Basic usage:**
```bash
python filter_colmap_with_masks.py \
    --colmap ../models/sparse/0 \
    --images ../data/raw \
    --rough-mask /path/to/rough/masks \
    --fine-mask /path/to/fine/masks
```

**With custom settings:**
```bash
python filter_colmap_with_masks.py \
    --colmap ../models/sparse/0 \
    --images ../data/raw \
    --rough-mask /path/to/rough/masks \
    --fine-mask /path/to/fine/masks \
    --threshold 25 \
    --combine and \
    --output ../models/filtered_model \
    --examples 10
```

**Arguments:**
- `--colmap`: Path to COLMAP model directory (required)
- `--images`: Path to original images directory (required)
- `--rough-mask`: Path to rough segmentation masks (required)
- `--fine-mask`: Path to fine segmentation masks (required)
- `--output`: Output directory for filtered model (default: auto-generated)
- `--threshold`: Mask threshold value (default: 10)
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

## 🔧 Example Workflow

### Basic 3D Reconstruction
```bash
# 1. Navigate to project
cd /work3/s204201/BotanicalTreeReconstruction

# 2. Add your images to data/raw/
cp /path/to/your/aerial/images/* data/raw/

# 3. Run reconstruction
cd scripts
python colmap_pipeline.py ../data/raw --output ../models/tree_reconstruction

# 4. Your 3D model is now in models/tree_reconstruction/sparse/0/
```

### Advanced Workflow with Mask Filtering
```bash
# 1. Create initial COLMAP model
cd scripts
python colmap_pipeline.py ../data/raw --output ../models/initial_model

# 2. Filter points using segmentation masks (if you have masks)
python filter_colmap_with_masks.py \
    --colmap ../models/initial_model/sparse/0 \
    --images ../data/raw \
    --rough-mask ../data/segmentation_masks \
    --fine-mask ../data/segmentation_masks \
    --output ../models/filtered_model \
    --threshold 10 \
    --combine or

# 3. Your filtered model is in models/filtered_model/
# 4. Visualizations are saved in models/filtered_model_visualizations/
```

## 🛠️ Dependencies

Install required packages:
```bash
pip install -r requirements.txt
```

Main dependencies:
- OpenCV
- NumPy
- Matplotlib
- COLMAP (external)

## 📝 Notes

- The script uses GPU acceleration by default when available
- Processing time depends on number and resolution of images
- Large image sets may require significant memory
- For best results, use images with good overlap and coverage

