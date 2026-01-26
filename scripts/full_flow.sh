#!/usr/bin/env bash
set -e
export CUDA_VISIBLE_DEVICES=4,5
source /home/tien.nguyen/miniforge3/etc/profile.d/conda.sh
conda activate sam3 \
    && cd /home/tien.nguyen/workspace/project/sam3 \
    && python sam3/train/train.py \
    -c configs/yap3/cropped_all_in_one_detection_segmentation_floor_plan_dataset.yaml \
    --use-cluster 0
python3 /home/tien.nguyen/workspace/project/sam3/sam3/infer/infer.py \
    -c /home/tien.nguyen/workspace/project/sam3/experiments/cropped_all_in_one_detection_segmentation_floor_plan_yap3_dataset_20260121/checkpoints/checkpoint.pt \
    --img-dir /home/tien.nguyen/workspace/project/common/data/yap-3/cropped_all_in_one_detection_segmentation_yap3_floor_plan_dataset/test \
    --masks-save-dir /home/tien.nguyen/workspace/project/sam3/sam3/infer/cropped_all_in_one_detection_segmentation_floor_plan_yap3_dataset_20260121_40_epochs/masks \
    --viz-save-dir /home/tien.nguyen/workspace/project/sam3/sam3/infer/cropped_all_in_one_detection_segmentation_floor_plan_yap3_dataset_20260121_40_epochs/visualization \
    --box-detections-save-dir /home/tien.nguyen/workspace/project/sam3/sam3/infer/cropped_all_in_one_detection_segmentation_floor_plan_yap3_dataset_20260121_40_epochs


# set -e
# export CUDA_VISIBLE_DEVICES=2,3
# source /home/tien.nguyen/miniforge3/etc/profile.d/conda.sh
# conda activate sam3 \
#     && cd /home/tien.nguyen/workspace/project/sam3 \
#     && python sam3/train/train.py \
#     -c configs/nice/uncropped_ds_roof_and_house_boundary_segmentation_windoor_slope_factor_box_detection.yaml \
#     --use-cluster 0
# python3 /home/tien.nguyen/workspace/project/sam3/sam3/infer/infer.py \
#     -c /home/tien.nguyen/workspace/project/sam3/experiments/uncropped_ds_roof_box_and_house_boundary_segmentation_windoor_slope_factor_box_detection_20260121/checkpoints/checkpoint.pt \
#     --img-dir /home/tien.nguyen/workspace/project/common/data/yap-3//home/tien.nguyen/workspace/project/common/data/nice/uncropped_elevation_dataset_roof_and_house_segmentation_windoor_slope_factor_box_detection_for_sam3_20260121/test \
#     --masks-save-dir /home/tien.nguyen/workspace/project/sam3/sam3/infer/uncropped_ds_roof_box_and_house_boundary_segmentation_windoor_slope_factor_box_detection_20260121/masks \
#     --viz-save-dir /home/tien.nguyen/workspace/project/sam3/sam3/infer/uncropped_ds_roof_box_and_house_boundary_segmentation_windoor_slope_factor_box_detection_20260121/visualization \
#     --box-detections-save-dir /home/tien.nguyen/workspace/project/sam3/sam3/infer/uncropped_ds_roof_box_and_house_boundary_segmentation_windoor_slope_factor_box_detection_20260121
# conda activate sam2-unet \
#     && cd /home/tien.nguyen/workspace/project/ykkap-ais-models/primus/segmentation/SAM2-UNet \
#     && python -m src.evaluation.eval \
#     --pred_path "/home/tien.nguyen/workspace/project/sam3/sam3/infer/uncropped_ds_roof_box_and_house_boundary_segmentation_windoor_slope_factor_box_detection_20260121/masks/roof/" \
#     --gt_path "/home/tien.nguyen/workspace/project/common/data/nice/uncropped_test_elevation_dataset_roof_segmentation_20260109/masks/"
# conda activate sam2-unet \
#     && cd /home/tien.nguyen/workspace/project/ykkap-ais-models/primus/segmentation/SAM2-UNet \
#     && python -m src.evaluation.eval \
#     --pred_path "/home/tien.nguyen/workspace/project/sam3/sam3/infer/uncropped_ds_roof_box_and_house_boundary_segmentation_windoor_slope_factor_box_detection_20260121/masks/house_boundary/" \
#     --gt_path "/home/tien.nguyen/workspace/project/common/data/nice/uncropped_test_elevation_dataset_house_boundary_segmentation_20260109/masks/"