#!/usr/bin/env python3
"""
Configuration Examples

This file contains various configuration examples for different use cases.
"""

import yaml
from pathlib import Path

# Basic configuration for standard tree reconstruction
BASIC_CONFIG = {
    "project_name": "basic_tree_reconstruction",
    "data": {
        "input_images": "data/raw",
        "output_masks": "data/processed/masks",
        "filtered_images": "data/processed/filtered_images"
    },
    "segmentation": {
        "model_path": None,  # Use pre-trained model
        "confidence": 0.5,
        "device": "cuda",
        "batch_size": 4,
        "min_vegetation_ratio": 0.1
    },
    "colmap": {
        "feature_type": "sift",
        "max_features": 30000,
        "use_gpu": True,
        "sequential": True,
        "run_dense": False
    },
    "filtering": {
        "filter_type": "coordinate",
        "coord_threshold": 3.0,
        "outlier_threshold": 2.0,
        "min_observations": 3,
        "add_points": False
    },
    "visualization": {
        "plot_types": ["3d", "projected", "stats"],
        "num_images": 6,
        "point_size": 8,
        "max_points": 50000
    },
    "output": {
        "sparse_model": "models/sparse",
        "filtered_model": "models/filtered", 
        "visualizations": "reports/figures",
        "logs": "logs"
    }
}

# High-quality configuration for research use
HIGH_QUALITY_CONFIG = {
    "project_name": "high_quality_reconstruction",
    "data": {
        "input_images": "data/raw",
        "output_masks": "data/processed/masks",
        "filtered_images": "data/processed/filtered_images"
    },
    "segmentation": {
        "model_path": "models/custom_tree_segmentation.pth",  # Custom model
        "confidence": 0.7,
        "device": "cuda",
        "batch_size": 2,  # Smaller batch for higher quality
        "min_vegetation_ratio": 0.15
    },
    "colmap": {
        "feature_type": "sift",
        "max_features": 50000,  # More features
        "use_gpu": True,
        "sequential": False,  # Exhaustive matching
        "run_dense": True  # Enable dense reconstruction
    },
    "filtering": {
        "filter_type": "combined",  # Use all filtering methods
        "coord_threshold": 5.0,  # Larger scene
        "outlier_threshold": 1.5,  # Stricter outlier removal
        "min_observations": 4,  # Require more observations
        "add_points": True  # Enable point triangulation
    },
    "visualization": {
        "plot_types": ["3d", "projected", "poses", "stats", "comparison"],
        "num_images": 12,
        "point_size": 6,
        "max_points": 100000
    },
    "output": {
        "sparse_model": "models/sparse",
        "filtered_model": "models/filtered",
        "dense_model": "models/dense",
        "visualizations": "reports/figures",
        "logs": "logs"
    }
}

# Fast configuration for quick testing
FAST_CONFIG = {
    "project_name": "fast_test_reconstruction",
    "data": {
        "input_images": "data/raw",
        "output_masks": "data/processed/masks", 
        "filtered_images": "data/processed/filtered_images"
    },
    "segmentation": {
        "model_path": None,
        "confidence": 0.4,  # Lower threshold for more detection
        "device": "cuda",
        "batch_size": 8,  # Larger batch for speed
        "min_vegetation_ratio": 0.05
    },
    "colmap": {
        "feature_type": "sift",
        "max_features": 10000,  # Fewer features for speed
        "use_gpu": True,
        "sequential": True,
        "run_dense": False
    },
    "filtering": {
        "filter_type": "coordinate",  # Simple filtering only
        "coord_threshold": 2.0,
        "outlier_threshold": 3.0,
        "min_observations": 2,
        "add_points": False
    },
    "visualization": {
        "plot_types": ["3d", "stats"],  # Minimal plots
        "num_images": 3,
        "point_size": 10,
        "max_points": 20000
    },
    "output": {
        "sparse_model": "models/sparse",
        "filtered_model": "models/filtered",
        "visualizations": "reports/figures",
        "logs": "logs"
    }
}

# CPU-only configuration (no GPU required)
CPU_CONFIG = {
    "project_name": "cpu_reconstruction",
    "data": {
        "input_images": "data/raw",
        "output_masks": "data/processed/masks",
        "filtered_images": "data/processed/filtered_images"
    },
    "segmentation": {
        "model_path": None,
        "confidence": 0.5,
        "device": "cpu",  # CPU only
        "batch_size": 1,  # Small batch for CPU
        "min_vegetation_ratio": 0.1
    },
    "colmap": {
        "feature_type": "sift",
        "max_features": 15000,
        "use_gpu": False,  # No GPU
        "sequential": True,
        "run_dense": False
    },
    "filtering": {
        "filter_type": "coordinate",
        "coord_threshold": 3.0,
        "outlier_threshold": 2.0,
        "min_observations": 3,
        "add_points": False
    },
    "visualization": {
        "plot_types": ["3d", "projected", "stats"],
        "num_images": 6,
        "point_size": 8,
        "max_points": 30000
    },
    "output": {
        "sparse_model": "models/sparse",
        "filtered_model": "models/filtered",
        "visualizations": "reports/figures",
        "logs": "logs"
    }
}

def save_config_examples():
    """Save all configuration examples to files"""
    
    configs_dir = Path("configs")
    configs_dir.mkdir(exist_ok=True)
    
    configs = {
        "basic.yaml": BASIC_CONFIG,
        "high_quality.yaml": HIGH_QUALITY_CONFIG,
        "fast.yaml": FAST_CONFIG,
        "cpu_only.yaml": CPU_CONFIG
    }
    
    for filename, config in configs.items():
        filepath = configs_dir / filename
        with open(filepath, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        print(f"✅ Saved {filepath}")
    
    print(f"\n📁 Configuration examples saved to {configs_dir}/")
    print("\nUsage:")
    print("  tree-reconstruct pipeline --config configs/basic.yaml")
    print("  tree-reconstruct pipeline --config configs/high_quality.yaml")
    print("  tree-reconstruct pipeline --config configs/fast.yaml")
    print("  tree-reconstruct pipeline --config configs/cpu_only.yaml")

def create_custom_config():
    """Interactive configuration creator"""
    
    print("🔧 Custom Configuration Creator")
    print("=" * 35)
    
    config = BASIC_CONFIG.copy()
    
    # Project name
    project_name = input("Project name [tree_reconstruction]: ").strip()
    if project_name:
        config["project_name"] = project_name
    
    # Device selection
    device = input("Device (cuda/cpu) [cuda]: ").strip().lower()
    if device in ["cpu", "cuda"]:
        config["segmentation"]["device"] = device
        config["colmap"]["use_gpu"] = device == "cuda"
    
    # Quality level
    print("\nQuality level:")
    print("1. Fast (fewer features, quick processing)")
    print("2. Standard (balanced quality and speed)")
    print("3. High (maximum quality, slower processing)")
    
    quality = input("Choose quality level (1-3) [2]: ").strip()
    
    if quality == "1":  # Fast
        config["colmap"]["max_features"] = 10000
        config["segmentation"]["batch_size"] = 8
        config["visualization"]["max_points"] = 20000
        
    elif quality == "3":  # High quality
        config["colmap"]["max_features"] = 50000
        config["colmap"]["sequential"] = False
        config["filtering"]["filter_type"] = "combined"
        config["filtering"]["add_points"] = True
        config["visualization"]["max_points"] = 100000
    
    # Save custom config
    output_path = Path("configs") / "custom.yaml"
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"\n✅ Custom configuration saved to {output_path}")
    print(f"Run with: tree-reconstruct pipeline --config {output_path}")

if __name__ == "__main__":
    print("Configuration Examples")
    print("=" * 22)
    print("1. Save all example configurations")
    print("2. Create custom configuration")
    
    choice = input("Choose option (1-2): ").strip()
    
    if choice == "1":
        save_config_examples()
    elif choice == "2":
        create_custom_config()
    else:
        print("Invalid choice. Saving example configurations...")
        save_config_examples()