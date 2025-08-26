from typing import Any, Dict, Optional

from verl.utils import hf_processor
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.dataset.vision_utils import (
    process_image,
    process_video,
)


def build_multi_modal_inputs(
    prompt: str,
    raw_mm_data: Optional[Dict[str, Any]],
    config: Dict[str, Any],
    **kwargs,
) -> Dict[str, Any]:
    """
    Build multi-modal inputs for model
    Adapted from: verl/utils/dataset/rl_dataset.py
    """
    local_path = copy_local_path_from_hdfs(config.model_path)
    processor = hf_processor(local_path, use_fast=True)
    # processor_type = processor.image_processor.__class__.__name__ if processor is not None else None

    if prompt is None:
        raise ValueError("Prompt is required for build multi-modal inputs")

    raw_images, raw_videos = raw_mm_data["image"], raw_mm_data["video"]

    multi_modal_data = {}
    images, videos = None, None
    if raw_images is not None:
        images = [process_image(image) for image in raw_images]
        multi_modal_data["image"] = images
    if raw_videos is not None:
        videos = [process_video(video) for video in raw_videos]
        multi_modal_data["video"] = [video.numpy() for video in videos]

    model_inputs = processor(text=[prompt], images=images, videos=videos, return_tensors="pt")

    model_inputs.pop("input_ids", None)  # TODO: check
    model_inputs.pop("attention_mask", None)

    if "second_per_grid_ts" in model_inputs:
        model_inputs.pop("second_per_grid_ts")

    # There's a trap here, multi_modal_inputs has to be a dict, not BatchFeature
    multi_modal_inputs = dict(model_inputs)

    # TODO: check position_ids 

    return {
        "prompt": prompt,
        "multi_modal_inputs": multi_modal_inputs,
        "multi_modal_data": multi_modal_data,
    }
