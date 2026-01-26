#!/usr/bin/env python3

import pathlib
import typing
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['EXPORTING_MODEL'] = 'true'
os.environ['DISABLE_BOX_ENCODING'] = 'true'
os.environ['DISABLE_POINT_ENCODING'] = 'true'

import imgviz
import numpy as np
import onnxruntime
import PIL.Image
from PIL import Image
import torch
from loguru import logger
from numpy.typing import NDArray
from osam._models.yoloworld.clip import tokenize
from torchvision.transforms import v2

from sam3.model.sam3_image import Sam3Image  # type: ignore[unresolved-import]
from sam3.model.sam3_image_processor import (  # type: ignore[unresolved-import]
    Sam3Processor,
)
from sam3.model_builder import build_sam3_image_model  # type: ignore[unresolved-import]


def get_replace_freqs_cis(module):
    if hasattr(module, "freqs_cis"):
        freqs_cos = module.freqs_cis.real.float()
        freqs_sin = module.freqs_cis.imag.float()
        # Replace the buffer
        module.register_buffer("freqs_cos", freqs_cos)
        module.register_buffer("freqs_sin", freqs_sin)
        del module.freqs_cis  # Remove complex version
    for child in module.children():
        get_replace_freqs_cis(child)


class _ImageEncoder(torch.nn.Module):
    def __init__(self, processor: Sam3Processor) -> None:
        super().__init__()
        self._processor: Sam3Processor = processor
        self._transform = v2.Compose(
            [
                # NOTE: Resize in .transform has difference between pytorch and onnx
                # v2.ToDtype(torch.uint8, scale=True),
                # v2.Resize(size=(resolution := 1008, resolution)),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        image = self._transform(image).unsqueeze(0)

        backbone_out = self._processor.model.backbone._forward_image_no_act_ckpt(image)
        del backbone_out["vision_features"]
        del backbone_out["sam2_backbone_out"]

        assert len(backbone_out["vision_pos_enc"]) == 3
        assert len(backbone_out["backbone_fpn"]) == 3
        return *backbone_out["vision_pos_enc"], *backbone_out["backbone_fpn"]


def _export_image_encoder(
    processor: Sam3Processor, image: PIL.Image.Image
) -> tuple[list[NDArray], list[NDArray]]:
    image = image.resize((1008, 1008), resample=PIL.Image.BILINEAR)

    onnx_file: pathlib.Path = pathlib.Path("models/sam3_image_encoder.onnx")
    if onnx_file.exists():
        logger.debug("onnx model already exists, skip export: {!r}", str(onnx_file))
    else:
        encoder: _ImageEncoder = _ImageEncoder(processor=processor)
        input_image: torch.Tensor = v2.functional.to_image(image).to("cuda")

        # with torch.no_grad():
        #     output = image_backbone(input_image)

        logger.debug("exporting onnx model: {!r}", str(onnx_file))
        torch.onnx.export(
            encoder,
            args=(input_image,),
            f=str(onnx_file),
            input_names=["image"],
            output_names=[
                "vision_pos_enc_0",
                "vision_pos_enc_1",
                "vision_pos_enc_2",
                "backbone_fpn_0",
                "backbone_fpn_1",
                "backbone_fpn_2",
            ],
            opset_version=21,
            verify=True,
            dynamo=True
        )
        logger.debug("exported onnx model: {!r}", str(onnx_file))

    session: onnxruntime.InferenceSession = onnxruntime.InferenceSession(onnx_file)
    output = session.run(None, {"image": np.asarray(image).transpose(2, 0, 1)})
    
    # encoder: _ImageEncoder = _ImageEncoder(processor=processor)
    # input_image: torch.Tensor = v2.functional.to_image(image).to("cuda")
    # output2 = encoder(input_image)
    # output = [o.cpu().detach().numpy() for o in output2]

    assert all(isinstance(o, np.ndarray) for o in output)
    output = typing.cast(list[NDArray], output)
    logger.debug("finished onnx runtime inference")

    vision_pos_enc: list[NDArray] = output[:3]
    backbone_fpn: list[NDArray] = output[3:]
    return vision_pos_enc, backbone_fpn


class _LanguageEncoder(torch.nn.Module):
    def __init__(self, processor: Sam3Processor) -> None:
        super().__init__()
        self._processor: Sam3Processor = processor

    def forward(
        self, tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        model: Sam3Image = self._processor.model

        # VETextEncoder.forward
        text_attention_mask = (tokens != 0).bool()
        #
        inputs_embeds = model.backbone.language_backbone.encoder.token_embedding(tokens)
        _, text_memory = model.backbone.language_backbone.encoder(tokens)
        #
        assert text_memory.shape[1] == inputs_embeds.shape[1]
        text_attention_mask = text_attention_mask.ne(1)
        text_memory = text_memory.transpose(0, 1)
        text_memory_resized = model.backbone.language_backbone.resizer(text_memory)
        return text_attention_mask, text_memory_resized, inputs_embeds.transpose(0, 1)


def _export_language_encoder(processor: Sam3Processor) -> list[NDArray]:
    tokens = tokenize(texts=["roof"], context_length=32)

    onnx_file: pathlib.Path = pathlib.Path("models/sam3_language_encoder.onnx")
    if onnx_file.exists():
        logger.debug("onnx model already exists, skip export: {!r}", str(onnx_file))
    else:
        encoder: _LanguageEncoder = _LanguageEncoder(processor=processor)
        tokens_input: torch.Tensor = torch.from_numpy(tokens).to("cuda")

        # with torch.no_grad():
        #     output = encoder(tokens=tokens_input)

        logger.debug("exporting onnx model: {!r}", str(onnx_file))
        torch.onnx.export(
            encoder,
            args=(tokens_input,),
            f=str(onnx_file),
            input_names=["tokens"],
            output_names=["text_attention_mask", "text_memory", "text_embeds"],
            opset_version=21,
            verify=True,
            dynamo=True
        )
        logger.debug("exported onnx model: {!r}", str(onnx_file))

    session: onnxruntime.InferenceSession = onnxruntime.InferenceSession(onnx_file)
    output = session.run(None, {"tokens": tokens})
    
    # encoder: _LanguageEncoder = _LanguageEncoder(processor=processor)
    # tokens_input: torch.Tensor = torch.from_numpy(tokens).to("cuda")
    # output = encoder(tokens_input)
    # output = [o.cpu().detach().numpy() for o in output]
    
    assert all(isinstance(o, np.ndarray) for o in output)
    output = typing.cast(list[NDArray], output)
    logger.debug("finished onnx runtime inference")

    return output


class _Decoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._model: Sam3Image = build_sam3_image_model(
            bpe_path="/home/tien.nguyen/workspace/project/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
            device="cuda",
            eval_mode=True,
            checkpoint_path="/home/tien.nguyen/workspace/project/sam3/experiments/roof_and_house_boundary_segmentation_20251230/checkpoints/checkpoint.pt",
            enable_segmentation=True
        )
        self._processor: Sam3Processor = Sam3Processor(self._model)

    def forward(
        self,
        original_height: torch.Tensor,
        original_width: torch.Tensor,
        vision_pos_enc_0: torch.Tensor,
        vision_pos_enc_1: torch.Tensor,
        vision_pos_enc_2: torch.Tensor,
        backbone_fpn_0: torch.Tensor,
        backbone_fpn_1: torch.Tensor,
        backbone_fpn_2: torch.Tensor,
        language_mask: torch.Tensor,
        language_features: torch.Tensor,
        language_embeds: torch.Tensor,
        # box_coords: torch.Tensor,
        # box_labels: torch.Tensor,
        # box_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        geometric_prompt = self._processor.model._get_dummy_prompt()
        # geometric_prompt.box_embeddings = box_coords
        # geometric_prompt.box_labels = box_labels
        # geometric_prompt.box_mask = box_masks
        state = {
            "original_height": original_height,
            "original_width": original_width,
            "backbone_out": {
                "vision_pos_enc": [
                    vision_pos_enc_0,
                    vision_pos_enc_1,
                    vision_pos_enc_2,
                ],
                "backbone_fpn": [
                    backbone_fpn_0,
                    backbone_fpn_1,
                    backbone_fpn_2,
                ],
                "language_mask": language_mask,
                "language_features": language_features,
                "language_embeds": language_embeds,
            },
            "geometric_prompt": geometric_prompt,
        }
        result = self._processor._forward_grounding(state)
        return result["boxes"], result["scores"], result["masks"]


def _export_decoder(
    original_height: int,
    original_width: int,
    vision_pos_enc_0: NDArray,
    vision_pos_enc_1: NDArray,
    vision_pos_enc_2: NDArray,
    backbone_fpn_0: NDArray,
    backbone_fpn_1: NDArray,
    backbone_fpn_2: NDArray,
    language_mask: NDArray,
    language_features: NDArray,
    language_embeds: NDArray,
    # box_coords: NDArray,
    # box_labels: NDArray,
    # box_masks: NDArray,
) -> list[NDArray]:
    onnx_file: pathlib.Path = pathlib.Path("models/sam3_decoder.onnx")
    if onnx_file.exists():
        logger.debug("onnx model already exists, skip export: {!r}", str(onnx_file))
    else:
        logger.debug("exporting onnx model: {!r}", str(onnx_file))
        decoder: _Decoder = _Decoder()

        # XXX: this inference is needed to make export work with if-condition with
        # torch.compiler.is_dynamo_compiling
        # with torch.no_grad():
        #     output = decoder(
        #         original_height=torch.tensor(original_height)[None].to("cuda"),
        #         original_width=torch.tensor(original_width)[None].to("cuda"),
        #         vision_pos_enc_0=torch.tensor(vision_pos_enc_0).to("cuda"),
        #         vision_pos_enc_1=torch.tensor(vision_pos_enc_1).to("cuda"),
        #         vision_pos_enc_2=torch.tensor(vision_pos_enc_2).to("cuda"),
        #         backbone_fpn_0=torch.tensor(backbone_fpn_0).to("cuda"),
        #         backbone_fpn_1=torch.tensor(backbone_fpn_1).to("cuda"),
        #         backbone_fpn_2=torch.tensor(backbone_fpn_2).to("cuda"),
        #         language_mask=torch.tensor(language_mask).to("cuda"),
        #         language_features=torch.tensor(language_features).to("cuda"),
        #         language_embeds=torch.tensor(language_embeds).to("cuda"),
        #         box_coords=torch.tensor(box_coords).to("cuda"),
        #         box_labels=torch.tensor(box_labels).to("cuda"),
        #     )

        torch.onnx.export(
            decoder,
            args=(
                torch.tensor(original_height).to("cuda"),
                torch.tensor(original_width).to("cuda"),
                torch.tensor(vision_pos_enc_0).to("cuda"),
                torch.tensor(vision_pos_enc_1).to("cuda"),
                torch.tensor(vision_pos_enc_2).to("cuda"),
                torch.tensor(backbone_fpn_0).to("cuda"),
                torch.tensor(backbone_fpn_1).to("cuda"),
                torch.tensor(backbone_fpn_2).to("cuda"),
                torch.tensor(language_mask).to("cuda"),
                torch.tensor(language_features).to("cuda"),
                torch.tensor(language_embeds).to("cuda"),
                # torch.tensor(box_coords).to("cuda"),
                # torch.tensor(box_labels).to("cuda"),
                # torch.tensor(box_masks).to("cuda"),
            ),
            f=onnx_file,
            input_names=[
                "original_height",
                "original_width",
                "vision_pos_enc_0",
                "vision_pos_enc_1",
                "vision_pos_enc_2",
                "backbone_fpn_0",
                "backbone_fpn_1",
                "backbone_fpn_2",
                "language_mask",
                "language_features",
                "language_embeds",
                # "box_coords",
                # "box_labels",
                # "box_masks",
            ],
            output_names=["boxes", "scores", "masks"],
            opset_version=21,
            dynamo=True,
            verify=True
        )
        logger.debug("exported onnx model: {!r}", str(onnx_file))

    session = onnxruntime.InferenceSession(onnx_file)
    output = session.run(
        None,
        {
            "original_height": np.array(original_height),
            "original_width": np.array(original_width),
            "vision_pos_enc_0": vision_pos_enc_0,
            "vision_pos_enc_1": vision_pos_enc_1,
            "vision_pos_enc_2": vision_pos_enc_2,
            "backbone_fpn_0": backbone_fpn_0,
            "backbone_fpn_1": backbone_fpn_1,
            "backbone_fpn_2": backbone_fpn_2,
            "language_mask": language_mask,
            "language_features": language_features,
            "language_embeds": language_embeds,
            # "box_coords": box_coords,
            # "box_labels": box_labels,
            # "box_masks": box_masks,
        },
    )

    assert all(isinstance(o, np.ndarray) for o in output)
    output = typing.cast(list[NDArray], output)
    logger.debug("decoder onnx runtime inference done")

    return output


def main():
    model: Sam3Image = build_sam3_image_model(
        bpe_path="/home/tien.nguyen/workspace/project/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
        device="cuda",
        eval_mode=True,
        checkpoint_path="/home/tien.nguyen/workspace/project/sam3/experiments/roof_box_and_house_boundary_segmentation_windoor_slope_factor_box_detection_20260119/checkpoints/checkpoint.pt",
        enable_segmentation=True
    )
    get_replace_freqs_cis(model)
    processor: Sam3Processor = Sam3Processor(model)
    for p in model.parameters():
        p.requires_grad_(False)

    image: PIL.Image.Image = PIL.Image.open("/home/tien.nguyen/workspace/project/common/data/nice/cropped_test_elevation_dataset_20251230/batch0/task3312/images/1247-Nice_Poc_Val_2-AE_elevation_crop_491.png")
    # image: PIL.Image.Image = PIL.Image.open("2011_000006.jpg")

    # image_encoder {{
    # state = processor.set_image(image)

    os.makedirs("models", exist_ok=True)
    vision_pos_enc, backbone_fpn = _export_image_encoder(processor, image)
    # }}

    # language_encoder {{
    # result = processor.set_text_prompt(prompt="person", state=state)

    language_mask, language_features, language_embeds = _export_language_encoder(
        processor=processor
    )
    # }}

    # decoder {{
    # result = processor._forward_grounding(state)
    # boxes, scores, masks = result["boxes"], result["scores"], result["masks"]

    box_coords = np.array([[[0.1620, 0.4010, 0.0640, 0.0180]]], dtype=np.float32)
    box_labels = np.array([[1]], dtype=np.int64)
    box_masks = np.array([[True]], dtype=np.bool_)

    boxes, scores, masks = _export_decoder(
        original_height=image.height,
        original_width=image.width,
        vision_pos_enc_0=vision_pos_enc[0],
        vision_pos_enc_1=vision_pos_enc[1],
        vision_pos_enc_2=vision_pos_enc[2],
        backbone_fpn_0=backbone_fpn[0],
        backbone_fpn_1=backbone_fpn[1],
        backbone_fpn_2=backbone_fpn[2],
        language_mask=language_mask,
        language_features=language_features,
        language_embeds=language_embeds,
        # box_coords=box_coords,
        # box_labels=box_labels,
        # box_masks=box_masks,
    )
    # }}

    print(boxes)
    print(scores)
    print(masks)

    viz = imgviz.instances2rgb(
        image=np.asarray(image),
        masks=masks[:, 0, :, :],
        bboxes=boxes[:, [1, 0, 3, 2]],
        labels=np.arange(len(boxes)) + 1,
        captions=[f"{s:.2f}" for s in scores],
    )
    viz_pil = Image.fromarray(viz)
    viz_pil.save("exported_debug.png")


if __name__ == "__main__":
    main()
