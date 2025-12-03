# 🌳 Botanical Tree Reconstruction

A Python toolkit for 3D tree reconstruction from aerial imagery using COLMAP structure-from-motion, deep learning segmentation, and ray-based point densification.

## 🐆 Quick Start (Recommended)

### 1. Setup
```bash
# Create Python virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Load COLMAP (on HPC systems)
module load colmap/3.8-cuda-11.8-avx512

# Add your aerial images to the data/raw/ folder
cp /path/to/your/drone/images/* data/raw/
```

### 2. Run Complete Pipeline
```bash
cd scripts

# Basic usage (full 5-step pipeline)
python run_pipeline.py --images ../data/raw --model ../models/model.pth
```

### 3. HPC Batch Execution
```bash
# Submit complete pipeline to cluster
bsub < run_pipeline.sh
```

## 📊 Pipeline Overview

The complete pipeline consists of 5 automated steps:

1. **🎯 Segmentation** - Generate tree masks using DeepLabV3
2. **🏗️ COLMAP Reconstruction** - Create initial 3D point cloud from images
3. **🔍 Point Filtering** - Remove non-tree points using mask visibility  
4. **✨ Ray Enhancement** - Add more points to sparse model via ray casting
5. **📊 Visualization** - Create comparison plots

**Input**: Raw aerial images (+ pre-trained segmentation model)  
**Output**: Enhanced sparse 3D point cloud + visualization + processing summary

## 🗂️ Project Structure

```
BotanicalTreeReconstruction/
├── README.md              # This documentation
├── requirements.txt       # Python dependencies
├── .gitignore            # Git ignore patterns
├── .gitattributes        # Git LFS configuration
├── data/                 # Datasets (raw images, segmentation masks)
├── models/               # Pre-trained segmentation model, Colmap models
├── outputs/              # Generated visualizations and summaries
└── scripts/              # All processing scripts
    ├── run_pipeline.py               # 🎯 Complete automation script
    ├── run_pipeline.sh               # 🚀 HPC batch script
    ├── segmentation_inference.py     # 🎭 Tree segmentation with DeepLabV3
    ├── colmap_reconstruction.py       # 🏗️ COLMAP reconstruction
    ├── mask_based_filtering.py       # 🔍 Point cloud filtering
    ├── ray_based_enhancement.py      # ✨ Ray-based point enhancement
    ├── visualize_results.py          # 📊 Comprehensive visualization
    ├── read_write_model.py           # 🛠️ COLMAP model I/O utilities
    └── logs/                         # HPC job output logs
```

## 🎯 Main Pipeline Script

**`run_pipeline.py`** - Complete automation script that orchestrates all pipeline steps.

### Key Arguments:
- `--images DIR` - Input images directory (required)
- `--model FILE` - Path to segmentation model (.pth file)
- `--mask-type {rough,fine,both}` - Type of masks to generate (default: both)
- `--max-features INT` - Maximum features for COLMAP (default: 20000)  
- `--visibility-threshold FLOAT` - Point filtering threshold 0.0-1.0 (default: 0.5)
- `--combine-masks {or,and}` - How to combine rough/fine masks (default: or)

### Skip Options:
- `--skip-segmentation` - Use existing masks in data/segmentation_masks/
- `--skip-colmap` - Use existing model in models/colmap_reconstruction/
- `--skip-filtering` - Use existing filtered model in models/colmap_filtered/
- `--skip-rays` - Skip ray enhancement (faster, less dense)
- `--skip-visualization` - Skip final visualization

### Advanced Options:
- `--colmap-model DIR` - Path to existing COLMAP model (when skipping COLMAP)
- `--filtered-model DIR` - Path to existing filtered model (when skipping filtering)
- `--continue-on-error` - Continue pipeline even if a step fails
- `--debug` - Enable detailed error output

## 🚀 HPC Batch Script

**`run_pipeline.sh`** - LSF batch script for cluster execution with GPU allocation.

- **Resources**: A100 GPU, 32GB RAM, 6-hour limit
- **Environment**: Loads COLMAP module, uses pip-installed requirements
- **Logs**: Output saved to `logs/JOBID.out` and `logs/JOBID.err`

```bash
# Submit pipeline to cluster
bsub < run_pipeline.sh
```

## 🛠️ Individual Scripts

For advanced users who need fine-grained control over individual pipeline steps.

### segmentation_inference.py
Generate tree segmentation masks using trained DeepLabV3 model.

**Key Arguments:**
- `--model FILE` - Path to trained segmentation model (.pth)
- `--input DIR` - Input images directory
- `--output DIR` - Output masks directory
- `--mask-type {rough,fine,both}` - Type of masks to generate
- `--device {cpu,cuda,auto}` - Processing device
- `--patch-size INT` - Patch size for fine segmentation (default: 512)

**Usage:**
```bash
# Generate both rough and fine masks
python segmentation_inference.py \
    --model ../models/model.pth \
    --input ../data/raw \
    --output ../data/segmentation_masks \
    --mask-type both --device cuda
```

### colmap_reconstruction.py
Create COLMAP 3D reconstruction from images with automatic model merging.

**Key Arguments:**
- `image_dir` - Path to input images directory (required)
- `--output DIR` - Output directory (default: ../models/colmap_output)
- `--max-features INT` - Max features per image (default: 20000)
- `--log-level {DEBUG,INFO,WARNING,ERROR}` - Logging level

**Features:**
- **Automatic GPU/CPU detection** - Uses GPU acceleration when available, falls back to CPU
- **Model merging** - Automatically merges multiple reconstructions when possible
- **Smart model selection** - Pipeline automatically uses the largest/best reconstruction

**Usage:**
```bash
# Basic reconstruction
python colmap_reconstruction.py ../data/raw

# High quality with 50k features
python colmap_reconstruction.py ../data/raw \
    --output ../models/high_quality_colmap \
    --max-features 50000
```

### mask_based_filtering.py
Filter COLMAP 3D points using segmentation masks with visibility-based thresholding.

**Key Arguments:**
- `--colmap DIR` - COLMAP model directory (required)
- `--images DIR` - Original images directory (required)
- `--rough-mask DIR` - Rough segmentation masks directory (required)
- `--fine-mask DIR` - Fine segmentation masks directory (required)
- `--output DIR` - Output directory for filtered model
- `--visibility-threshold FLOAT` - Keep points visible in ≥this fraction of images (default: 0.5)
- `--threshold INT` - Mask pixel threshold value (default: 10)
- `--combine {or,and}` - Mask combination method (default: or)
- `--examples INT` - Number of visualization examples to save (default: 5)

**Usage:**
```bash
# Filter using both rough and fine masks
python mask_based_filtering.py \
    --colmap ../models/colmap_reconstruction/sparse/0 \
    --images ../data/raw \
    --rough-mask ../data/segmentation_masks \
    --fine-mask ../data/segmentation_masks \
    --visibility-threshold 0.5 \
    --combine or
```

**Note:** The pipeline automatically selects the best COLMAP model (merged model if available, otherwise the largest reconstruction).

### ray_based_enhancement.py
Enhance sparse COLMAP model by adding more points using ray-casting and voxel-based sampling.

**Key Arguments:**
- `--colmap_model_dir DIR` - Input COLMAP model directory (required)
- `--rough_mask_dir DIR` - Directory with rough masks (required)
- `--fine_mask_dir DIR` - Directory with fine masks (required)
- `--output_dir DIR` - Base output directory (required)
- `--output-folder-name NAME` - Specific folder name for output
- `--mask_thresh INT` - Mask threshold for combining rough/fine masks (default: 10)
- `--samples_per_image INT` - Sampled mask pixels per image (default: 1000)
- `--depth_samples INT` - Depth samples per ray (default: 50)
- `--voxel_size FLOAT` - Voxel size in meters (default: 0.02)
- `--min_image_support INT` - Minimum images supporting a voxel (default: 3)
- `--chunk_size FLOAT` - Spatial chunk size in meters for processing (smaller = less memory, more runtime)(default: 4)

**Usage:**
```bash
# Enhance sparse model using ray casting
python ray_based_enhancement.py \
    --colmap_model_dir ../models/colmap_filtered \
    --rough_mask_dir ../data/segmentation_masks \
    --fine_mask_dir ../data/segmentation_masks \
    --output_dir ../models \
    --output-folder-name colmap_ray_enhanced \
    --samples_per_image 1000
```

### visualize_results.py
Create comprehensive visualizations showing all pipeline stages.

**Key Arguments:**
- `--images DIR` - Original images directory (required)
- `--original_model DIR` - Original COLMAP model directory (required)
- `--filtered_model DIR` - Filtered COLMAP model directory (required)
- `--ray_model DIR` - Ray-enhanced COLMAP model directory
- `--output FILE` - Output visualization image path (required)
- `--masks DIR` - Segmentation masks directory
- `--n_images INT` - Number of sample images to visualize (default: 3)
- `--point_size FLOAT` - Size of projected points (default: 1.0)
- `--mask_type {rough,fine,both}` - Type of masks to display (default: both)
- `--show_combined` - Show additional column with combined filtered + ray points

**Usage:**
```bash
# Create 5-column visualization (default)
python visualize_results.py \
    --images ../data/raw \
    --masks ../data/segmentation_masks \
    --original_model ../models/colmap_reconstruction/sparse/0 \
    --filtered_model ../models/colmap_filtered \
    --ray_model ../models/colmap_ray_enhanced \
    --n_images 3

# Create 6-column visualization with combined view
python visualize_results.py \
    --images ../data/raw \
    --masks ../data/segmentation_masks \
    --original_model ../models/colmap_reconstruction/sparse/0 \
    --filtered_model ../models/colmap_filtered \
    --ray_model ../models/colmap_ray_enhanced \
    --n_images 3 \
    --show_combined
```

Creates a 5-column visualization showing:
1. **Original Image** - Raw input images
2. **Segmentation Mask** - Combined tree segmentation overlaid on images  
3. **Original COLMAP** - Points from initial reconstruction (red)
4. **Filtered COLMAP** - Points after mask-based filtering (blue)
5. **New Ray Points** - Only newly added points from ray enhancement (green)

With `--show_combined`, adds a 6th column:
6. **Combined (Filtered + Ray)** - All points in final enhanced model (purple)


## 📊 Output Structure

After running the complete pipeline, your outputs will be organized as:

```
data/
└── segmentation_masks/          # Step 1 - Generated masks
    ├── image1_rough.png
    ├── image1_fine.png
    ├── image2_rough.png
    └── image2_fine.png
    
models/
├── colmap_reconstruction/    # Step 2 - Initial COLMAP reconstruction
│   ├── database.db
│   └── sparse/
│       ├── 0/               # Best reconstruction (merged if multiple existed)
│       └── 1/               # Second reconstruction (if multiple)
├── colmap_filtered/         # Step 3 - Filtered points using masks
│   ├── cameras.bin
│   ├── images.bin
│   └── points3D.bin
└── colmap_ray_enhanced/     # Step 4 - Enhanced sparse model from ray casting
    ├── cameras.bin
    ├── images.bin
    └── points3D.bin

outputs/
├── pipeline_visualization.png    # Step 5 - 5-column comparison plot
├── processing_summary.txt       # Detailed processing log
└── point_statistics.txt         # Point counts for each stage

**Notes:**
- **Multiple COLMAP models**: If COLMAP creates disconnected reconstructions, both will be kept and the pipeline will automatically use the largest one
- **Model merging**: When possible, separate reconstructions are merged and the result overwrites the smaller model
- **Automatic selection**: The pipeline selects the best model (largest by points/images) without manual intervention
```

## 🚨 Troubleshooting

### Common Issues:

**"COLMAP not found"**
- Ensure COLMAP is installed and in PATH
- On HPC: `module load colmap` before running
- Test with: `colmap --help`

**"CUDA out of memory"** 
- Reduce `--max-features` (try 10000)
- Use `--mask-type rough` only
- Add `--device cpu` for segmentation

**"Model file not found"**
- Ensure model is tracked with Git LFS: `git lfs pull`
- Check file size is >100MB: `ls -lh models/`

**"No images found"**
- Check image directory structure: images should be directly in specified folder
- Supported formats: .jpg, .jpeg, .png
- Ensure absolute paths or run from correct directory

