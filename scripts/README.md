# COLMAP Pipeline Script

This directory contains scripts for 3D reconstruction using COLMAP.

## colmap_pipeline.py

A simplified Python script to create a COLMAP 3D reconstruction model from an image directory.

### Prerequisites

- COLMAP installed and available in PATH
- Python 3.7+
- Images in a directory (JPG, PNG, TIFF, BMP formats supported)

On HPC systems, load COLMAP:
```bash
module load colmap/3.8-cuda-11.8-avx512
```

### Usage

Basic usage:
```bash
python colmap_pipeline.py /path/to/images
```

With custom output directory:
```bash
python colmap_pipeline.py /path/to/images -o /path/to/output
```

With custom settings:
```bash
python colmap_pipeline.py /path/to/images \
    --output ./my_model \
    --max-features 30000 \
    --log-level DEBUG
```

### Arguments

- `image_dir`: Path to directory containing input images (required)
- `-o, --output`: Output directory for the COLMAP model (default: ./colmap_output)
- `--max-features`: Maximum number of features to extract per image (default: 20000)
- `--log-level`: Logging verbosity (DEBUG, INFO, WARNING, ERROR)

### Output

The script creates the following structure:
```
output_directory/
├── database.db          # COLMAP database with features and matches
└── sparse/             # Sparse 3D reconstruction
    └── 0/              # Reconstruction model
        ├── cameras.bin
        ├── images.bin
        └── points3D.bin
```

### Example

```bash
# Create model from filtered images
python colmap_pipeline.py "/work3/s204201/RisøOakTree - filtered" \
    --output ./tree_model \
    --max-features 25000

# The model will be saved to ./tree_model/sparse/0/
```

### Notes

- The script uses GPU acceleration by default if available
- Processing time depends on the number and resolution of images
- Large image sets may require significant memory and processing time
- The script automatically handles database cleanup and directory creation