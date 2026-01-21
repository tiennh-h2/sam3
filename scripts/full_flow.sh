#!/usr/bin/env bash
set -e
export CUDA_VISIBLE_DEVICES=1,2
source /home/tien.nguyen/miniforge3/etc/profile.d/conda.sh
conda activate sam3 \
    && cd /home/tien.nguyen/workspace/project/sam3 \
    && python sam3/train/train.py \
    -c configs/nice/roof_and_house_boundary_segmentation_windoor_slope_factor_box_detection.yaml \
    --use-cluster 0
python3 /home/tien.nguyen/workspace/project/sam3/sam3/infer/infer.py \
    -c /home/tien.nguyen/workspace/project/sam3/experiments/roof_box_and_house_boundary_segmentation_windoor_slope_factor_box_detection_20260119/checkpoints/checkpoint.pt \
    --img-dir /home/tien.nguyen/workspace/project/common/data/nice/cropped_elevation_dataset_roof_and_house_segmentation_windoor_slope_factor_box_detection_for_sam3_20260119/test \
    --masks-save-dir /home/tien.nguyen/workspace/project/sam3/sam3/infer/roof_box_and_house_boundary_segmentation_windoor_slope_factor_box_detection_20260119/masks \
    --viz-save-dir /home/tien.nguyen/workspace/project/sam3/sam3/infer/roof_box_and_house_boundary_segmentation_windoor_slope_factor_box_detection_20260119/visualization
conda activate sam2-unet \
    && cd /home/tien.nguyen/workspace/project/ykkap-ais-models/primus/segmentation/SAM2-UNet \
    && python -m src.evaluation.eval \
    --pred_path "/home/tien.nguyen/workspace/project/sam3/sam3/infer/roof_box_and_house_boundary_segmentation_windoor_slope_factor_box_detection_20260119/masks/roof/" \
    --gt_path "/home/tien.nguyen/workspace/project/common/data/nice/cropped_test_elevation_dataset_roof_segmentation_20260109/masks/"
conda activate sam2-unet \
    && cd /home/tien.nguyen/workspace/project/ykkap-ais-models/primus/segmentation/SAM2-UNet \
    && python -m src.evaluation.eval \
    --pred_path "/home/tien.nguyen/workspace/project/sam3/sam3/infer/roof_box_and_house_boundary_segmentation_windoor_slope_factor_box_detection_20260119/masks/house_boundary/" \
    --gt_path "/home/tien.nguyen/workspace/project/common/data/nice/cropped_test_elevation_dataset_house_boundary_segmentation_20260109/masks/"