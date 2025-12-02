#!/bin/bash
#BSUB -J filtering_test
#BSUB -q hpc
#BSUB -W 03:00
#BSUB -R "rusage[mem=8GB]"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err
#BSUB -B
## BSUB -N
## BSUB -u xx@dtu.dk # write email here and uncomment

# Load required modules
# No COLMAP module needed for filtering

# Activate conda environment if needed
source /work3/s204201/miniconda3/etc/profile.d/conda.sh
conda activate colmap_env

# Change to project directory
cd /work3/s204201/BotanicalTreeReconstruction/scripts

echo "=== Running Filtering Tests with Different Visibility Thresholds ==="

# Test with mask threshold 3
echo "Testing mask threshold 20..."
python mask_based_filtering.py \
    --colmap ../models/colmap_reconstruction2611 \
    --images ../data/raw \
    --rough-mask ../data/segmentation_masks \
    --fine-mask ../data/segmentation_masks \
    --threshold 20 \
    --visibility-threshold 0 \
    --combine or \
    --output ../models/filtertest_mask20_vis0 

python mask_based_filtering.py \
    --colmap ../models/colmap_reconstruction2611 \
    --images ../data/raw \
    --rough-mask ../data/segmentation_masks \
    --fine-mask ../data/segmentation_masks \
    --threshold 20 \
    --visibility-threshold 20 \
    --combine or \
    --output ../models/filtertest_mask20_vis20 

python mask_based_filtering.py \
    --colmap ../models/colmap_reconstruction2611 \
    --images ../data/raw \
    --rough-mask ../data/segmentation_masks \
    --fine-mask ../data/segmentation_masks \
    --threshold 20 \
    --visibility-threshold 50 \
    --combine or \
    --output ../models/filtertest_mask20_vis50 

python mask_based_filtering.py \
    --colmap ../models/colmap_reconstruction2611 \
    --images ../data/raw \
    --rough-mask ../data/segmentation_masks \
    --fine-mask ../data/segmentation_masks \
    --threshold 20 \
    --visibility-threshold 70 \
    --combine or \
    --output ../models/filtertest_mask20_vis70    