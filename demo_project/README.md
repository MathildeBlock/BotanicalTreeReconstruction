# Demo Tree Reconstruction Project

This is a demonstration project created by the test script.

## Quick Start

1. Copy your aerial images to `data/raw/`
2. Run the pipeline: `tree-reconstruct pipeline --config configs/basic.yaml`

## Directory Structure

- `data/raw/`: Input aerial images
- `data/processed/`: Processed images and masks
- `models/`: 3D reconstruction models
- `reports/figures/`: Generated visualizations
- `configs/`: Configuration files
- `logs/`: Processing logs

## Example Commands

```bash
# Run complete pipeline
tree-reconstruct pipeline --config configs/basic.yaml

# Run individual stages
tree-reconstruct segment --input-dir data/raw --output-dir data/processed/masks
tree-reconstruct colmap --input-dir data/raw --output-dir models/sparse
tree-reconstruct filter --input-model models/sparse/sparse/0 --output-model models/filtered
tree-reconstruct visualize --input-model models/filtered --output-dir reports/figures

# Check project status
tree-reconstruct status --project .
```
