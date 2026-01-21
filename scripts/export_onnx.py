import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
from pathlib import Path
from PIL import Image
from torchvision.transforms import v2
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import FindStage, Sam3Processor
from sam3.model import box_ops
from sam3.model.data_misc import interpolate

device = "cpu" # for onnx export we use CPU for maximum compatibility

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--checkpoint-path', default="/home/tien.nguyen/workspace/project/sam3/experiments/roof_and_house_boundary_segmentation_20251230/checkpoints/checkpoint.pt")
parser.add_argument('--img-example-path', default="/home/tien.nguyen/workspace/project/common/data/nice/cropped_test_elevation_dataset_20251230/batch0/task3312/images/1247-Nice_Poc_Val_2-AE_elevation_crop_491.png")
parser.add_argument('--onnx-path', default=".")
args = parser.parse_args()

# 1. Load model & processor
model = build_sam3_image_model(
    bpe_path="/home/tien.nguyen/workspace/project/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
    device=device,
    eval_mode=True,
    checkpoint_path=args.checkpoint_path,
    enable_segmentation=True
)
processor = Sam3Processor(model)

model.eval()

# 2. Build a sample batch (same as your example)
image = Image.open(args.img_example_path).convert("RGB")
width, height = image.size
original_width = torch.tensor([width]).to(device)
original_height = torch.tensor([height]).to(device)

prompt = "roof"
text_outputs = model.backbone.forward_text([prompt], device=device)
image = v2.functional.to_image(image)
image = processor.transform(image).unsqueeze(0).to(device)

# 3. Wrap Sam3Model so the ONNX graph has clean inputs/outputs
class Sam3ONNXWrapper(torch.nn.Module):
    def __init__(self, sam3):
        super().__init__()
        self.sam3 = sam3
        self.find_stage = FindStage(
            img_ids=torch.tensor([0], device=device, dtype=torch.long),
            text_ids=torch.tensor([0], device=device, dtype=torch.long),
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )

    def forward(self, original_width, original_height, image, language_features, language_mask, language_embeds):
        backbone_out = self.sam3.backbone.forward_image(image)
        backbone_out["language_features"] = language_features
        backbone_out["language_mask"] = language_mask
        backbone_out["language_embeds"] = language_embeds
        outputs = self.sam3.forward_grounding(
            backbone_out=backbone_out,
            find_input=self.find_stage,
            geometric_prompt=self.sam3._get_dummy_prompt(),
            find_target=None,
        )
        out_bbox = outputs["pred_boxes"]
        out_logits = outputs["pred_logits"]
        out_masks = outputs["pred_masks"]
        out_probs = out_logits.sigmoid()
        presence_score = outputs["presence_logit_dec"].sigmoid().unsqueeze(1)
        out_probs = (out_probs * presence_score).squeeze(-1)

        keep = out_probs > 0.5
        out_probs = out_probs[keep]
        out_masks = out_masks[keep]
        out_bbox = out_bbox[keep]

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)

        img_h = original_height
        img_w = original_width
        scale_fct = torch.tensor([img_w, img_h, img_w, img_h]).to(device)
        boxes = boxes * scale_fct[None, :]

        out_masks = interpolate(
            out_masks.unsqueeze(1),
            (img_h, img_w),
            mode="bilinear",
            align_corners=False,
        ).sigmoid()

        return out_masks > 0.5, boxes, out_probs

wrapper = Sam3ONNXWrapper(model).to(device).eval()

# 5. Export to ONNX
output_dir = Path(args.onnx_path)
output_dir.mkdir(exist_ok=True)
onnx_path = output_dir / "sam3_static.onnx"

torch.onnx.export(
    wrapper,
    (original_width, original_height, image, text_outputs["language_features"], text_outputs["language_mask"], text_outputs["language_embeds"]),
    onnx_path,
    input_names=["original_width", "original_height", "image", "language_features", "language_mask", "language_embeds"],
    output_names=["masks", "boxes", "scores"],
    opset_version=21,
    verify=True,
    dynamo=True
)
print(f"Exported to {onnx_path}")