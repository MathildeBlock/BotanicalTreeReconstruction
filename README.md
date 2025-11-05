# 🌳 Botanical Tree Reconstruction

A comprehensive Python toolkit for 3D tree reconstruction from aerial imagery using deep learning segmentation and COLMAP structure-from-motion.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 Features

- **🎯 Semantic Segmentation**: AI-powered tree detection and masking from aerial images
- **📐 3D Reconstruction**: COLMAP-based structure-from-motion pipeline
- **🔧 Point Cloud Processing**: Advanced filtering, outlier removal, and optimization
- **📊 Comprehensive Visualizations**: 3D plots, projections, statistics, and comparisons
- **⚙️ Flexible Configuration**: YAML-based configuration for different workflows
- **🖥️ Command Line Interface**: Easy-to-use CLI for all operations
- **🐍 Python API**: Programmatic access to all functionality

## 📦 Installation

### Prerequisites

- Python 3.9 or higher
- CUDA-capable GPU (recommended)
- [COLMAP](https://colmap.github.io/) installed and available in PATH

### Install from source

```bash
git clone https://github.com/MathildeBlock/BotanicalTreeReconstruction.git
cd BotanicalTreeReconstruction
pip install -e .
```

### Install dependencies

```bash
pip install -r requirements.txt
```

## 🏃‍♂️ Quick Start

### 1. Initialize a new project

```bash
tree-reconstruct init --project my_tree_project
cd my_tree_project
```

### 2. Add your aerial images

```bash
cp /path/to/your/images/* data/raw/
```

### 3. Run the complete pipeline

```bash
tree-reconstruct pipeline --config configs/basic.yaml
```

That's it! Your 3D reconstruction will be saved in `models/` and visualizations in `reports/figures/`.

## 📖 Detailed Usage

### Command Line Interface

The toolkit provides several CLI commands:

#### Pipeline Command
Run the complete reconstruction pipeline:

```bash
tree-reconstruct pipeline --config config.yaml [--stage STAGE] [--dry-run]
```

#### Individual Stages

**Segmentation:**
```bash
tree-reconstruct segment \
    --input-dir data/raw \
    --output-dir data/processed/masks \
    --confidence 0.5 \
    --device cuda
```

**COLMAP Reconstruction:**
```bash
tree-reconstruct colmap \
    --input-dir data/raw \
    --output-dir models/sparse \
    --max-features 30000 \
    --gpu
```

**Point Cloud Filtering:**
```bash
tree-reconstruct filter \
    --input-model models/sparse/sparse/0 \
    --output-model models/filtered \
    --filter-type combined \
    --coord-thresh 3.0
```

**Visualization:**
```bash
tree-reconstruct visualize \
    --input-model models/filtered \
    --output-dir reports/figures \
    --plot-type all \
    --num-images 6
```

#### Utility Commands

**Project Status:**
```bash
tree-reconstruct status --project .
```

**Initialize Project:**
```bash
tree-reconstruct init --project new_project --template advanced
```

### Python API

You can also use the toolkit programmatically:

```python
from tree_reconstruction import segmentation, colmap_pipeline, filtering, visualization
from pathlib import Path

# Run segmentation
segmentation.run_segmentation(
    input_dir=Path("data/raw"),
    output_dir=Path("data/masks"),
    confidence=0.5
)

# Run COLMAP reconstruction
colmap_pipeline.run_colmap_pipeline(
    input_dir=Path("data/raw"),
    output_dir=Path("models/sparse"),
    max_features=30000
)

# Filter point cloud
filtering.filter_point_cloud(
    input_model=Path("models/sparse/sparse/0"),
    output_model=Path("models/filtered"),
    filter_type="coordinate",
    coord_threshold=3.0
)

# Create visualizations
visualization.create_visualizations(
    input_model=Path("models/filtered"),
    output_dir=Path("reports/figures"),
    plot_type="all"
)
```

## ⚙️ Configuration

The toolkit uses YAML configuration files to manage pipeline parameters. Here's an example:

```yaml
project_name: "tree_reconstruction"

data:
  input_images: "data/raw"
  output_masks: "data/processed/masks"
  filtered_images: "data/processed/filtered_images"

segmentation:
  model_path: null  # Uses pre-trained model
  confidence: 0.5
  device: "cuda"
  batch_size: 4
  min_vegetation_ratio: 0.1

colmap:
  feature_type: "sift"
  max_features: 30000
  use_gpu: true
  sequential: true
  run_dense: false

filtering:
  filter_type: "combined"  # coordinate, mask, outlier, combined
  coord_threshold: 3.0
  outlier_threshold: 2.0
  min_observations: 3
  add_points: false

visualization:
  plot_types: ["3d", "projected", "stats"]
  num_images: 6
  point_size: 8
  max_points: 50000

output:
  sparse_model: "models/sparse"
  filtered_model: "models/filtered"
  visualizations: "reports/figures"
```

### Configuration Templates

The toolkit includes several configuration templates:

- `configs/basic.yaml`: Standard reconstruction pipeline
- `configs/advanced.yaml`: Advanced options with dense reconstruction

## 📊 Outputs

The toolkit generates several types of outputs:

### 3D Models
- **Sparse Point Cloud**: `models/sparse/` - Initial COLMAP reconstruction
- **Filtered Point Cloud**: `models/filtered/` - Cleaned and optimized model
- **Dense Point Cloud**: `models/dense/` - Dense reconstruction (if enabled)

### Visualizations
- **3D Point Cloud Plot**: Interactive 3D visualization of the point cloud
- **Projected Points**: 2D visualizations showing 3D points projected on images
- **Camera Poses**: 3D visualization of camera positions and orientations
- **Statistics**: Comprehensive statistics and quality metrics
- **Model Comparisons**: Side-by-side comparisons of different models

### Processed Data
- **Segmentation Masks**: Binary masks identifying tree regions
- **Filtered Images**: Images with sufficient vegetation content
- **Quality Reports**: Reconstruction quality assessment

## 🔧 Advanced Features

### Custom Segmentation Models

You can use your own trained segmentation models:

```bash
tree-reconstruct segment \
    --input-dir data/raw \
    --output-dir data/masks \
    --model path/to/your/model.pth \
    --confidence 0.7
```

### Point Cloud Optimization

The toolkit includes advanced point cloud optimization:

```python
from tree_reconstruction.filtering import triangulate_new_points

# Add new triangulated points
new_points = triangulate_new_points(
    cameras, images, points3D, 
    max_new_points=1000
)
```

### Batch Processing

Process multiple datasets:

```python
from tree_reconstruction.config import run_pipeline
import yaml

datasets = ["dataset1", "dataset2", "dataset3"]

for dataset in datasets:
    config = yaml.safe_load(open(f"{dataset}/config.yaml"))
    run_pipeline(config)
```

## 🐛 Troubleshooting

### Common Issues

**COLMAP not found:**
```bash
# Install COLMAP
conda install -c conda-forge colmap
# or
apt-get install colmap
```

**CUDA out of memory:**
- Reduce `batch_size` in segmentation config
- Reduce `max_features` in COLMAP config
- Use `device: "cpu"` for segmentation

**Poor reconstruction quality:**
- Increase `max_features` (up to 50000)
- Use `sequential: false` for exhaustive matching
- Adjust `coord_threshold` for filtering
- Enable `add_points: true` for optimization

**No segmentation model:**
The toolkit uses a pre-trained DeepLabV3 model by default. For better results with tree imagery, train a custom model.

### Performance Tips

1. **Use GPU acceleration** when available
2. **Filter images** before reconstruction to remove low-quality images
3. **Adjust feature count** based on your dataset size and quality
4. **Use appropriate coordinate thresholds** for your scene scale

## 📚 Examples

See the `examples/` directory for complete workflows:

- `examples/basic_workflow.py`: Simple reconstruction pipeline
- `examples/custom_segmentation.py`: Using custom segmentation models
- `examples/batch_processing.py`: Processing multiple datasets
- `examples/quality_assessment.py`: Analyzing reconstruction quality

## 🤝 Contributing

We welcome contributions! Please see `CONTRIBUTING.md` for guidelines.

### Development Setup

```bash
git clone https://github.com/johannebirkchristensen/Botanical-Tree-Reconstruction.git
cd Botanical-Tree-Reconstruction
pip install -e ".[dev]"
pre-commit install
```

### Running Tests

```bash
pytest tests/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use this toolkit in your research, please cite:

```bibtex
@software{botanical_tree_reconstruction,
  title={Botanical Tree Reconstruction: A Python Toolkit for 3D Tree Reconstruction from Aerial Imagery},
  author={Tree Reconstruction Team},
  year={2025},
  url={https://github.com/johannebirkchristensen/Botanical-Tree-Reconstruction}
}
```

## 🙏 Acknowledgments

- [COLMAP](https://colmap.github.io/) for structure-from-motion algorithms
- [PyTorch](https://pytorch.org/) for deep learning framework
- [Open3D](http://www.open3d.org/) for 3D visualization utilities

## 📞 Support

- 📧 Email: your@email.com
- 🐛 Issues: [GitHub Issues](https://github.com/johannebirkchristensen/Botanical-Tree-Reconstruction/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/johannebirkchristensen/Botanical-Tree-Reconstruction/discussions)
