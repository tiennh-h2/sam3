import os
import json
import random
import shutil

ROOF_CATEGORY_ID = 20
HOUSE_BOUNDARY_CATEGORY_ID = 1

NEW_ROOF_CATEGORY_ID = 0
NEW_HOUSE_BOUNDARY_CATEGORY_ID = 1

SPLIT_RATIO = 0.8
random.seed(42)

raw_data_dir = "/home/tien.nguyen/workspace/project/common/data/nice/cropped_elevation_dataset_20260109"
raw_test_data_dir = "/home/tien.nguyen/workspace/project/common/data/nice/cropped_test_elevation_dataset_20260109"
save_path = "/home/tien.nguyen/workspace/project/common/data/nice/cropped_elevation_dataset_roof_and_house_segmentation_for_sam3_20260109"

train_dir = os.path.join(save_path, "train")
val_dir = os.path.join(save_path, "valid")
test_dir = os.path.join(save_path, "test")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

all_images = []
all_annotations = []
all_images_in_annotations = set()
img_id = 0
ann_id = 0

# -------- MERGE & FILTER ROOF --------
for batch in os.listdir(raw_data_dir):
    for task in os.listdir(os.path.join(raw_data_dir, batch)):
        task_dir = os.path.join(raw_data_dir, batch, task)
        coco = json.load(open(os.path.join(task_dir, "annotations", "instances_default.json")))

        chosen_anns = [a for a in coco["annotations"] if a["category_id"] in [ROOF_CATEGORY_ID, HOUSE_BOUNDARY_CATEGORY_ID]]

        img_id_map = {}
        for img in coco["images"]:
            img_id_map[img["id"]] = img_id
            img["id"] = img_id
            img["src_path"] = os.path.join(task_dir, "images", img["file_name"])
            img["file_name"] = os.path.basename(img["file_name"])
            img_id += 1
            all_images.append(img)

        for ann in chosen_anns:
            ann["id"] = ann_id
            ann["image_id"] = img_id_map[ann["image_id"]]
            all_images_in_annotations.add(ann["image_id"])
            ann["category_id"] = NEW_ROOF_CATEGORY_ID if ann["category_id"] == ROOF_CATEGORY_ID else NEW_HOUSE_BOUNDARY_CATEGORY_ID
            ann_id += 1
            ann["noun_phrase"] = "roof" if ann["category_id"] == NEW_ROOF_CATEGORY_ID else "house boundary"
            all_annotations.append(ann)

all_images = [img for img in all_images if img["id"] in all_images_in_annotations]

all_test_images = []
all_test_annotations = []
all_test_images_in_annotations = set()
for batch in os.listdir(raw_test_data_dir):
    for task in os.listdir(os.path.join(raw_test_data_dir, batch)):
        task_dir = os.path.join(raw_test_data_dir, batch, task)
        coco = json.load(open(os.path.join(task_dir, "annotations", "instances_default.json")))

        chosen_anns = [a for a in coco["annotations"] if a["category_id"] in [ROOF_CATEGORY_ID, HOUSE_BOUNDARY_CATEGORY_ID]]

        img_id_map = {}
        for img in coco["images"]:
            img_id_map[img["id"]] = img_id
            img["id"] = img_id
            img["src_path"] = os.path.join(task_dir, "images", img["file_name"])
            img["file_name"] = os.path.basename(img["file_name"])
            img_id += 1
            all_test_images.append(img)

        for ann in chosen_anns:
            ann["id"] = ann_id
            ann["image_id"] = img_id_map[ann["image_id"]]
            all_test_images_in_annotations.add(ann["image_id"])
            ann["category_id"] = NEW_ROOF_CATEGORY_ID if ann["category_id"] == ROOF_CATEGORY_ID else NEW_HOUSE_BOUNDARY_CATEGORY_ID
            ann_id += 1
            ann["noun_phrase"] = "roof" if ann["category_id"] == NEW_ROOF_CATEGORY_ID else "house boundary"
            all_test_annotations.append(ann)

all_test_images = [img for img in all_test_images if img["id"] in all_test_images_in_annotations]

# -------- SPLIT --------
random.shuffle(all_images)
split = int(len(all_images) * SPLIT_RATIO)

train_imgs = all_images[:split]
val_imgs = all_images[split:]

train_ids = {i["id"] for i in train_imgs}
val_ids = {i["id"] for i in val_imgs}

train_anns = [a for a in all_annotations if a["image_id"] in train_ids]
val_anns = [a for a in all_annotations if a["image_id"] in val_ids]

# -------- COPY IMAGES --------
for img in train_imgs:
    shutil.copy(img["src_path"], os.path.join(train_dir, img["file_name"]))
    img.pop("src_path")

for img in val_imgs:
    shutil.copy(img["src_path"], os.path.join(val_dir, img["file_name"]))
    img.pop("src_path")

for img in all_test_images:
    shutil.copy(img["src_path"], os.path.join(test_dir, img["file_name"]))
    img.pop("src_path")

# -------- WRITE COCO --------
def write_coco(path, images, annotations):
    json.dump(
        {
            "info": {},
            "licenses": [{"id": 0}],
            "categories": [{
                "id": NEW_ROOF_CATEGORY_ID,
                "name": "roof",
                "supercategory": ""
            }, {
                "id": NEW_HOUSE_BOUNDARY_CATEGORY_ID,
                "name": "house_boundary",
                "supercategory": ""
            }],
            "images": images,
            "annotations": annotations,
        },
        open(path, "w")
    )

write_coco(os.path.join(train_dir, "_annotations.coco.json"), train_imgs, train_anns)
write_coco(os.path.join(val_dir, "_annotations.coco.json"), val_imgs, val_anns)
write_coco(os.path.join(test_dir, "_annotations.coco.json"), all_test_images, all_test_annotations)
