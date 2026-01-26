import os
import json
import random
import shutil

CATEGORIES = {
    "info": 1,
    "door_single": 2,
    "door_double": 3,
    "window_single": 4,
    "window_double": 5,
    "room_name": 23,
    "parent-child_door": 32,
    "stair": 34,
    "counter": 39,
    "cabinet": 40,
    "kasagi": 37,
}
CATEGORIES_REV = {v: k for k, v in CATEGORIES.items()}

NEW_CATEGORIES = {
    "info": 0,
    "door_single": 1,
    "door_double": 2,
    "window_single": 3,
    "window_double": 4,
    "room_name": 5,
    "parent-child_door": 6,
    "stair": 7,
    "counter": 8,
    "cabinet": 9,
    "kasagi": 10,
}
NEW_CATEGORIES_REV = {v: k for k, v in NEW_CATEGORIES.items()}

SPLIT_RATIO = 0.8
random.seed(42)

raw_data_dir = "/home/tien.nguyen/workspace/project/common/data/yap-3/cropped_raw_dataset_20260114/train_val"
raw_test_data_dir = "/home/tien.nguyen/workspace/project/common/data/yap-3/cropped_raw_dataset_20260114/test"
save_path = "/home/tien.nguyen/workspace/project/common/data/yap-3/cropped_all_in_one_detection_segmentation_yap3_floor_plan_dataset"

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

none_counters = []

# -------- MERGE & FILTER --------
for batch in os.listdir(raw_data_dir):
    for task in os.listdir(os.path.join(raw_data_dir, batch)):
        task_dir = os.path.join(raw_data_dir, batch, task)
        coco = json.load(open(os.path.join(task_dir, "annotations", "instances_default.json")))

        chosen_anns = [a for a in coco["annotations"] if a["category_id"] in CATEGORIES.values()]

        img_id_map = {}
        for img in coco["images"]:
            img_id_map[img["id"]] = img_id
            img["id"] = img_id
            img["src_path"] = os.path.join(task_dir, "images", img["file_name"])
            img["file_name"] = os.path.basename(img["file_name"])
            img_id += 1
            all_images.append(img)

        for ann in chosen_anns:
            if not ann["segmentation"] and ann["category_id"] in [39]:
                none_counters.append(("/".join(task_dir.split("/")[-3:]), ann["id"]))
            ann["id"] = ann_id
            ann["image_id"] = img_id_map[ann["image_id"]]
            all_images_in_annotations.add(ann["image_id"])
            ann["category_id"] = NEW_CATEGORIES[CATEGORIES_REV[ann["category_id"]]]
            ann["noun_phrase"] = NEW_CATEGORIES_REV[ann["category_id"]].replace("_", " ")
            ann["segmentation"] = ann["segmentation"] if ann["category_id"] == 10 else None
            ann_id += 1
            all_annotations.append(ann)

all_images = [img for img in all_images if img["id"] in all_images_in_annotations]

all_test_images = []
all_test_annotations = []
all_test_images_in_annotations = set()
for batch in os.listdir(raw_test_data_dir):
    for task in os.listdir(os.path.join(raw_test_data_dir, batch)):
        task_dir = os.path.join(raw_test_data_dir, batch, task)
        coco = json.load(open(os.path.join(task_dir, "annotations", "instances_default.json")))

        chosen_anns = [a for a in coco["annotations"] if a["category_id"] in CATEGORIES.values()]

        img_id_map = {}
        for img in coco["images"]:
            img_id_map[img["id"]] = img_id
            img["id"] = img_id
            img["src_path"] = os.path.join(task_dir, "images", img["file_name"])
            img["file_name"] = os.path.basename(img["file_name"])
            img_id += 1
            all_test_images.append(img)

        for ann in chosen_anns:
            if not ann["segmentation"] and ann["category_id"] in [39]:
                none_counters.append(("/".join(task_dir.split("/")[-3:]), ann["id"]))
            ann["id"] = ann_id
            ann["image_id"] = img_id_map[ann["image_id"]]
            all_test_images_in_annotations.add(ann["image_id"])
            ann["category_id"] = NEW_CATEGORIES[CATEGORIES_REV[ann["category_id"]]]
            ann["noun_phrase"] = NEW_CATEGORIES_REV[ann["category_id"]].replace("_", " ")
            ann["segmentation"] = ann["segmentation"] if ann["category_id"] == 10 else None
            ann_id += 1
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
                "id": NEW_CATEGORIES[cat_id],
                "name": cat_id.replace("_", " "),
                "supercategory": ""
            } for cat_id in NEW_CATEGORIES.keys()],
            "images": images,
            "annotations": annotations,
        },
        open(path, "w")
    )

write_coco(os.path.join(train_dir, "_annotations.coco.json"), train_imgs, train_anns)
write_coco(os.path.join(val_dir, "_annotations.coco.json"), val_imgs, val_anns)
write_coco(os.path.join(test_dir, "_annotations.coco.json"), all_test_images, all_test_annotations)


import pandas as pd
df = pd.DataFrame(none_counters, columns=["task_path", "annotation_id"])
df.to_csv(os.path.join(save_path, "none_counters.csv"), index=False)