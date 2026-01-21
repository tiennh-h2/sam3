import os
import json
import random
import shutil

STAIRS_ID = 15
SPECIAL_AREA_ID = 16
BALCONY_ID = 17
ROOF_CATEGORY_ID = 20
HOUSE_BOUNDARY_CATEGORY_ID = 1

NEW_ROOF_CATEGORY_ID = 0
NEW_HOUSE_BOUNDARY_CATEGORY_ID = 1
NEW_STAIRS_ID = 2
NEW_BALCONY_ID = 3
NEW_ENTRANCE_ID = 4
NEW_BATHROOM_ID = 5
NEW_DIRT_FLOOR_ID = 6
NEW_GARAGE_ID = 7

NAMES_MAP = {
    NEW_ROOF_CATEGORY_ID: "roof",
    NEW_HOUSE_BOUNDARY_CATEGORY_ID: "house boundary",
    NEW_STAIRS_ID: "stairs",
    NEW_BALCONY_ID: "balcony",
    NEW_ENTRANCE_ID: "entrance",
    NEW_BATHROOM_ID: "bathroom",
    NEW_DIRT_FLOOR_ID: "dirt floor",
    NEW_GARAGE_ID: "garage",
}

SPLIT_RATIO = 0.8
random.seed(42)

raw_data_dir = "/home/tien.nguyen/workspace/project/common/data/nice/cropped_full_dataset"
raw_test_data_dir = "/home/tien.nguyen/workspace/project/common/data/nice/cropped_test_full_dataset"
save_path = "/home/tien.nguyen/workspace/project/common/data/nice/cropped_full_dataset_all_in_one_segmentation_for_sam3"

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
for task in os.listdir(raw_data_dir):
    task_dir = os.path.join(raw_data_dir, task)
    coco = json.load(open(os.path.join(task_dir, "annotations", "instances_default.json")))

    chosen_anns = [a for a in coco["annotations"] if a["category_id"] in [STAIRS_ID, SPECIAL_AREA_ID, BALCONY_ID, ROOF_CATEGORY_ID, HOUSE_BOUNDARY_CATEGORY_ID]]

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

        new_category_id = None
        if ann["category_id"] == ROOF_CATEGORY_ID:
            new_category_id = NEW_ROOF_CATEGORY_ID
        elif ann["category_id"] == HOUSE_BOUNDARY_CATEGORY_ID:
            new_category_id = NEW_HOUSE_BOUNDARY_CATEGORY_ID
        elif ann["category_id"] == STAIRS_ID:
            new_category_id = NEW_STAIRS_ID
        elif ann["category_id"] == BALCONY_ID:
            new_category_id = NEW_BALCONY_ID
        elif ann["category_id"] == SPECIAL_AREA_ID:
            if ann["attributes"]["type"] in ["entrance", "entrace"]:
                new_category_id = NEW_ENTRANCE_ID
            elif ann["attributes"]["type"] == "bath_room":
                new_category_id = NEW_BATHROOM_ID
            elif ann["attributes"]["type"] == "dirt_floor":
                new_category_id = NEW_DIRT_FLOOR_ID
            elif ann["attributes"]["type"] == "garage":
                new_category_id = NEW_GARAGE_ID
            else:
                print(f"Unknown special area type: {ann['attributes']['type']}")
                continue

        ann["category_id"] = new_category_id
        ann["noun_phrase"] = NAMES_MAP[new_category_id]
        ann_id += 1
        all_annotations.append(ann)

all_images = [img for img in all_images if img["id"] in all_images_in_annotations]

all_test_images = []
all_test_annotations = []
all_test_images_in_annotations = set()
for task in os.listdir(raw_test_data_dir):
    task_dir = os.path.join(raw_test_data_dir, task)
    coco = json.load(open(os.path.join(task_dir, "annotations", "instances_default.json")))

    chosen_anns = [a for a in coco["annotations"] if a["category_id"] in [STAIRS_ID, SPECIAL_AREA_ID, BALCONY_ID, ROOF_CATEGORY_ID, HOUSE_BOUNDARY_CATEGORY_ID]]

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

        new_category_id = None
        if ann["category_id"] == ROOF_CATEGORY_ID:
            new_category_id = NEW_ROOF_CATEGORY_ID
        elif ann["category_id"] == HOUSE_BOUNDARY_CATEGORY_ID:
            new_category_id = NEW_HOUSE_BOUNDARY_CATEGORY_ID
        elif ann["category_id"] == STAIRS_ID:
            new_category_id = NEW_STAIRS_ID
        elif ann["category_id"] == BALCONY_ID:
            new_category_id = NEW_BALCONY_ID
        elif ann["category_id"] == SPECIAL_AREA_ID:
            if ann["attributes"]["type"] in ["entrance", "entrace"]:
                new_category_id = NEW_ENTRANCE_ID
            elif ann["attributes"]["type"] == "bath_room":
                new_category_id = NEW_BATHROOM_ID
            elif ann["attributes"]["type"] == "dirt_floor":
                new_category_id = NEW_DIRT_FLOOR_ID
            elif ann["attributes"]["type"] == "garage":
                new_category_id = NEW_GARAGE_ID
            else:
                print(f"Unknown special area type: {ann['attributes']['type']}")
                continue

        ann["category_id"] = new_category_id
        ann["noun_phrase"] = NAMES_MAP[new_category_id]
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
                "id": cat_id,
                "name": name,
                "supercategory": ""
            } for cat_id, name in NAMES_MAP.items()],
            "images": images,
            "annotations": annotations,
        },
        open(path, "w")
    )

write_coco(os.path.join(train_dir, "_annotations.coco.json"), train_imgs, train_anns)
write_coco(os.path.join(val_dir, "_annotations.coco.json"), val_imgs, val_anns)
write_coco(os.path.join(test_dir, "_annotations.coco.json"), all_test_images, all_test_annotations)
